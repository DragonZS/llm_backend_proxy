"""FastAPI dependencies for accessing app-scoped state."""

from __future__ import annotations

from fastapi import Request

from ..agents.resolver import AgentResolver
from ..backends import BackendPool
from ..config import ProxyConfig
from ..core.context import RequestContext


def get_pool(request: Request) -> BackendPool:
    return request.app.state.pool


def get_config(request: Request) -> ProxyConfig:
    return request.app.state.config


def get_resolver(request: Request) -> AgentResolver:
    return request.app.state.resolver


def make_context(request: Request) -> RequestContext:
    return RequestContext.new(
        method=request.method,
        path=request.url.path,
        headers={k.lower(): v for k, v in request.headers.items()},
    )
