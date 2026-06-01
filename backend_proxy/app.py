"""FastAPI app factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from . import __version__
from .api import admin as admin_router
from .api import anthropic_messages as anthropic_router
from .api import openai_models as models_router
from .api import openai_responses as responses_router
from .api import passthrough as passthrough_router
from .agents import AgentResolver, builtin_adapter_config, builtin_agents
from .backends import BackendPool
from .config import ProxyConfig, load_config
from .config.reload import ConfigReloader, install_sighup
from .core import configure_logging, get_logger
from .core.errors import ProxyError, render_error
from .core.metrics import Metrics
from .core.middleware import ObservabilityMiddleware

log = get_logger("backend_proxy.app")


def create_app(config: ProxyConfig | None = None, *, config_path: str | None = None) -> FastAPI:
    if config is None:
        config = load_config(config_path)
    configure_logging(config.server.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = BackendPool.from_configs(config.backends, config.routing)
        agents = config.agents or builtin_agents()
        adapter_cfg = dict(config.adapters)
        for k, v in builtin_adapter_config().items():
            adapter_cfg.setdefault(k, v)
        resolver = AgentResolver(agents, adapter_cfg)
        app.state.config = config
        app.state.config_path = config_path
        app.state.pool = pool
        app.state.resolver = resolver
        app.state.metrics = Metrics()
        app.state.reloader = ConfigReloader(app, config_path=config_path)
        install_sighup(app)
        await pool.start()
        log.info(
            "backend-proxy %s up — %d backend(s): %s; %d agent profile(s): %s",
            __version__, len(pool.backends),
            ", ".join(b.name for b in pool.backends),
            len(agents), ", ".join(a.name for a in agents),
        )
        try:
            yield
        finally:
            await pool.aclose()
            log.info("backend-proxy shutdown complete")

    app = FastAPI(
        title="backend-proxy",
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    if config.server.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(ObservabilityMiddleware)

    @app.exception_handler(ProxyError)
    async def _proxy_error(request: Request, exc: ProxyError):  # noqa: ARG001
        return render_error(exc)

    # ORDER MATTERS: specific routes first, catch-all last.
    app.include_router(admin_router.router)
    app.include_router(models_router.router)
    app.include_router(responses_router.router)
    app.include_router(anthropic_router.router)
    app.include_router(passthrough_router.router)
    return app
