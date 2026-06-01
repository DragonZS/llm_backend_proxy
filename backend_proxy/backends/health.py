"""Background health watcher.

A single asyncio task periodically pings each backend's ``/health``. Each
backend has a corresponding bool flag that the pool consults in ``pick()``.

Hysteresis (``unhealthy_threshold`` / ``healthy_threshold``) prevents a
single transient probe failure from flipping the backend to unhealthy —
which is fatal for single-backend deployments where any 503 immediately
becomes a user-visible 503. With the defaults (2 / 1) two consecutive
failures are required to mark unhealthy, but a single success brings it
back instantly. A flap is the worst-case outcome, not an outage.

The watcher does not own circuit breaker state (that is request-driven),
but a sustained unhealthy run will eventually flip the breaker into OPEN
so the pool can short-circuit straightaway instead of waiting for a real
client request to fail.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from ..config.schema import HealthCheckConfig
from .base import Backend
from .circuit_breaker import CircuitBreaker

log = logging.getLogger("backend_proxy.health")


class HealthWatcher:
    def __init__(
        self,
        backends: List[Backend],
        breakers: Dict[str, CircuitBreaker],
        *,
        config: Optional[HealthCheckConfig] = None,
    ) -> None:
        self._backends = backends
        self._breakers = breakers
        self._cfg = config or HealthCheckConfig()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Optimistic start: assume healthy until a probe says otherwise.
        # Otherwise the first request after startup races the first probe and
        # may see healthy=True (default-True initialisation) which is fine, or
        # healthy=False (which would 503 the first request). Keeping True.
        self.healthy: Dict[str, bool] = {b.name: True for b in backends}
        # Per-backend running counters of *consecutive* probe outcomes.
        self._consec_fail: Dict[str, int] = {b.name: 0 for b in backends}
        self._consec_ok:   Dict[str, int] = {b.name: 0 for b in backends}

        # Push the configured probe timeout to backends that support it
        # (currently OpenAICompatBackend & subclasses). This keeps probes off
        # the inference timeout (which can be 600s) and off the inference
        # connection pool's keepalive (which vLLM closes mid-stream).
        for b in backends:
            setter = getattr(b, "set_health_probe", None)
            if callable(setter):
                try:
                    setter(timeout_s=self._cfg.probe_timeout_s)
                except Exception:  # noqa: BLE001
                    pass

    def start(self) -> None:
        if not self._cfg.enabled:
            log.info("health watcher disabled by config; backends always considered healthy")
            return
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="backend-proxy.health")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            await asyncio.gather(*[self._probe(b) for b in self._backends],
                                 return_exceptions=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._cfg.interval_s)
            except asyncio.TimeoutError:
                pass

    async def _probe(self, b: Backend) -> None:
        try:
            ok = await b.health()
        except Exception as e:  # noqa: BLE001
            log.debug("health probe %s raised %s", b.name, e)
            ok = False

        if ok:
            self._consec_ok[b.name] += 1
            self._consec_fail[b.name] = 0
        else:
            self._consec_fail[b.name] += 1
            self._consec_ok[b.name] = 0

        prev = self.healthy[b.name]
        if prev:
            # Currently healthy → flip only after N consecutive failures.
            if self._consec_fail[b.name] >= self._cfg.unhealthy_threshold:
                self.healthy[b.name] = False
                log.warning(
                    "backend %s is unhealthy (%d consecutive failed probes)",
                    b.name, self._consec_fail[b.name],
                )
                br = self._breakers.get(b.name)
                if br is not None:
                    br.record_failure()
            elif self._consec_fail[b.name] == 1:
                # First failure since healthy — emit a debug-level note so
                # operators can correlate with upstream noise but no warning
                # spam.
                log.debug(
                    "backend %s probe failed once; need %d consecutive to mark unhealthy",
                    b.name, self._cfg.unhealthy_threshold,
                )
        else:
            # Currently unhealthy → flip back after N consecutive successes.
            if self._consec_ok[b.name] >= self._cfg.healthy_threshold:
                self.healthy[b.name] = True
                log.info("backend %s is healthy again", b.name)
