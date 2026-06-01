"""Live config reload.

Two ingress points:

1. ``POST /admin/reload`` — programmatic, requires the optional admin token.
2. ``SIGHUP`` — invoked by humans / systemctl reload.

Both call ``app.state.reloader.reload()`` which:

* re-loads the YAML at ``app.state.config_path``,
* shuts down the existing pool's health watcher and httpx clients,
* installs a new pool / resolver,
* keeps the FastAPI server itself running (no socket churn).

In-flight requests against the previous pool finish naturally because we hold
references via the closure they captured; only the new state is publicly
visible.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
from typing import Optional

from fastapi import FastAPI

from ..agents import AgentResolver, builtin_adapter_config, builtin_agents
from ..backends import BackendPool
from ..config import load_config

log = logging.getLogger("backend_proxy.reload")


class ConfigReloader:
    def __init__(self, app: FastAPI, *, config_path: Optional[str]) -> None:
        self.app = app
        self.config_path = config_path
        self._lock = asyncio.Lock()

    async def reload(self) -> dict:
        async with self._lock:
            log.info("reloading config from %s", self.config_path or "<env>")
            new_cfg = load_config(self.config_path)

            new_pool = BackendPool.from_configs(new_cfg.backends, new_cfg.routing)
            agents = new_cfg.agents or builtin_agents()
            adapter_cfg = dict(new_cfg.adapters)
            for k, v in builtin_adapter_config().items():
                adapter_cfg.setdefault(k, v)
            new_resolver = AgentResolver(agents, adapter_cfg)

            await new_pool.start()
            old_pool: BackendPool = self.app.state.pool

            # Hot-swap then drain the old pool.
            self.app.state.config = new_cfg
            self.app.state.pool = new_pool
            self.app.state.resolver = new_resolver

            asyncio.create_task(_drain(old_pool, drain_window_s=new_cfg.server.drain_window_s))
            return {
                "status": "reloaded",
                "backends": [b.name for b in new_pool.backends],
                "agents": [a.name for a in agents],
            }


async def _drain(old_pool: BackendPool, *, drain_window_s: float = 2.0) -> None:
    # Give in-flight requests up to a moment to finish, then close the clients.
    await asyncio.sleep(drain_window_s)
    try:
        await old_pool.aclose()
    except Exception:  # noqa: BLE001
        log.warning("error closing old pool", exc_info=True)


def install_sighup(app: FastAPI) -> None:
    """Install a SIGHUP handler that schedules a reload on the running loop.

    Multi-worker uvicorn: each worker gets its own handler — that's fine, they
    each re-read the same YAML."""

    if threading.current_thread() is not threading.main_thread():
        return  # signal handlers only work on the main thread

    def _handler(signum, frame):  # noqa: ARG001
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(app.state.reloader.reload())
            log.info("SIGHUP received — reload scheduled")
        except Exception:  # noqa: BLE001
            log.exception("failed to schedule SIGHUP reload")

    try:
        signal.signal(signal.SIGHUP, _handler)
    except (AttributeError, ValueError):
        # SIGHUP may be unavailable (Windows) or signal.signal may fail in
        # nested loops — fall back to admin endpoint only.
        log.debug("SIGHUP handler not installed", exc_info=True)
