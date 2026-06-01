"""Backend pool with routing strategies, health watcher, and circuit breakers.

The pool exposes:

    pool.pick(model=...)      -> Backend
    async with pool.use(b):   -> increments _inflight on b for the duration
    await pool.list_models_aggregated()
    await pool.start() / await pool.aclose()

Selection algorithm:

    candidates = [b for b in backends if breaker(b).is_available() and healthy(b)]
    pick = primary_strategy.pick(candidates, model=model)
    if pick is None and fallback_strategy is not None:
        pick = fallback_strategy.pick(candidates, model=None)
    if pick is None:
        raise NoBackendAvailable
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from ..config.schema import BackendConfig, RoutingConfig
from ..core.errors import NoBackendAvailable, UpstreamError
from .base import Backend, PassthroughResponse
from .circuit_breaker import CircuitBreaker
from .factory import make_backend
from .health import HealthWatcher
from .openai_compat import OpenAICompatBackend
from .rate_limit import RateLimited, TokenBucket
from .strategies import Strategy, make_strategy

log = logging.getLogger("backend_proxy.pool")


class BackendPool:
    def __init__(self, backends: Sequence[Backend], routing: RoutingConfig) -> None:
        if not backends:
            raise ValueError("BackendPool requires at least one backend")
        self._backends: List[Backend] = list(backends)
        self._by_name: Dict[str, Backend] = {b.name: b for b in self._backends}
        self.routing = routing
        self.primary = make_strategy(routing.strategy)
        self.fallback = make_strategy(routing.fallback) if routing.fallback != "none" else None

        self._breakers: Dict[str, CircuitBreaker] = {
            b.name: CircuitBreaker(
                failures_threshold=routing.circuit_breaker.failures,
                cooldown_s=routing.circuit_breaker.cooldown_s,
            ) for b in self._backends
        }
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._buckets: Dict[str, TokenBucket] = {}
        for b in self._backends:
            cap = b.cfg.max_concurrency if isinstance(b, OpenAICompatBackend) else 64
            self._semaphores[b.name] = asyncio.Semaphore(cap)
            rl = getattr(b.cfg, "rate_limit", None)
            if rl is not None and rl.rps > 0:
                self._buckets[b.name] = TokenBucket(rps=rl.rps, burst=rl.burst)
            # Counter consulted by LeastBusy
            setattr(b, "_inflight", 0)

        self._health = HealthWatcher(
            self._backends, self._breakers,
            config=getattr(routing, "health_check", None),
        )

    # ---- lifecycle ---------------------------------------------------
    @classmethod
    def from_configs(
        cls, cfgs: Sequence[BackendConfig], routing: RoutingConfig
    ) -> "BackendPool":
        return cls([make_backend(c) for c in cfgs], routing)

    async def start(self) -> None:
        # Run an initial health probe synchronously so /readyz immediately
        # reflects reality, then start the periodic watcher.
        for b in self._backends:
            try:
                self._health.healthy[b.name] = await b.health()
            except Exception:  # noqa: BLE001
                self._health.healthy[b.name] = False
        self._health.start()

    async def aclose(self) -> None:
        await self._health.stop()
        await asyncio.gather(
            *[b.aclose() for b in self._backends], return_exceptions=True,
        )

    # ---- introspection ----------------------------------------------
    @property
    def backends(self) -> List[Backend]:
        return list(self._backends)

    @property
    def healthy_map(self) -> Dict[str, bool]:
        return dict(self._health.healthy)

    def get_by_name(self, name: str) -> Optional[Backend]:
        return self._by_name.get(name)

    # ---- selection ---------------------------------------------------
    def _candidates(self) -> List[Backend]:
        return [
            b for b in self._backends
            if self._breakers[b.name].is_available()
            and self._health.healthy.get(b.name, True)
        ]

    def pick(self, *, model: Optional[str] = None) -> Backend:
        candidates = self._candidates()
        if not candidates:
            # SAFETY NET: nothing passed both filters. If at least one backend
            # has its breaker still closed (i.e. real client requests have not
            # been failing), give it a chance — the health probe may simply be
            # flapping. The breaker is the authoritative signal because it's
            # driven by real traffic, not by a periodic probe that can hit a
            # closed keepalive socket and falsely report unhealthy.
            probe_bypass = [
                b for b in self._backends
                if self._breakers[b.name].is_available()
            ]
            if probe_bypass:
                log.warning(
                    "all backends marked unhealthy by probe but breakers are closed; "
                    "trying %s anyway (probe-bypass)", probe_bypass[0].name,
                )
                return probe_bypass[0]
            raise NoBackendAvailable("no healthy backend available")
        chosen = self.primary.pick(candidates, model=model)
        if chosen is None and self.fallback is not None:
            chosen = self.fallback.pick(candidates, model=None)
        if chosen is None:
            # ModelAffinity returned None and there's no fallback — pick first
            chosen = candidates[0]
        return chosen

    @contextlib.asynccontextmanager
    async def use(self, backend: Backend) -> AsyncIterator[Backend]:
        """Track in-flight requests + forward circuit-breaker callbacks +
        enforce per-backend rate limit (token bucket)."""
        bucket = self._buckets.get(backend.name)
        if bucket is not None:
            mode = backend.cfg.rate_limit.mode  # type: ignore[union-attr]
            try:
                await bucket.acquire(mode=mode)
            except RateLimited as e:
                # Surface to API layer as a 429.
                raise UpstreamError(
                    str(e), status_code=429, code="rate_limited",
                    extras={"backend": backend.name},
                ) from None

        sem = self._semaphores[backend.name]
        await sem.acquire()
        backend._inflight = getattr(backend, "_inflight", 0) + 1  # type: ignore[attr-defined]
        try:
            yield backend
            self._breakers[backend.name].record_success()
        except UpstreamError as exc:
            # 4xx is a client-side error (bad payload, unknown model, malformed
            # tool_choice, …) — the backend behaved correctly by rejecting it,
            # so it must NOT count against the circuit breaker. Otherwise a
            # client that repeatedly sends a malformed request can trip the
            # breaker and DoS every other tenant. Only 5xx / connection
            # failures indicate the backend itself is unhealthy.
            sc = exc.status_code
            if sc is not None and 400 <= sc < 500:
                self._breakers[backend.name].record_success()
            else:
                self._breakers[backend.name].record_failure()
            raise
        except Exception:
            self._breakers[backend.name].record_failure()
            raise
        finally:
            backend._inflight = max(0, getattr(backend, "_inflight", 1) - 1)  # type: ignore[attr-defined]
            sem.release()

    # ---- aggregation -------------------------------------------------
    async def list_models_aggregated(self) -> Dict[str, Any]:
        results = await asyncio.gather(
            *[b.list_models() for b in self._backends],
            return_exceptions=True,
        )
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for res in results:
            if isinstance(res, BaseException):
                continue
            for m in res:
                mid = m.get("id")
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                merged.append(m)
        return {"object": "list", "data": merged}
