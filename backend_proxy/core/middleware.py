"""Middleware for trace_id stamping + per-request structured log line +
metrics emission."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import Metrics

log = logging.getLogger("backend_proxy.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """trace_id stamping + access log + metrics emission. Reads metrics off
    ``app.state.metrics`` so the lifespan can construct them after the
    middleware is installed."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Honour upstream trace_id (e.g. behind a gateway) when present.
        trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.trace_id = trace_id
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
            status = resp.status_code
        except Exception:
            elapsed = time.perf_counter() - t0
            log.exception(
                "trace=%s %s %s -> 500 %.1fms (unhandled)",
                trace_id, request.method, request.url.path, elapsed * 1000,
            )
            raise

        elapsed = time.perf_counter() - t0
        agent = (resp.headers.get("x-backend-proxy-agent") or "-")
        backend = (resp.headers.get("x-backend-proxy-backend") or "-")
        # Always echo trace id back to caller.
        resp.headers["x-request-id"] = trace_id

        log.info(
            "trace=%s %s %s -> %d %.1fms agent=%s backend=%s",
            trace_id, request.method, request.url.path,
            status, elapsed * 1000, agent, backend,
        )

        metrics: Metrics | None = getattr(request.app.state, "metrics", None)
        if metrics is not None and metrics.enabled:
            route = request.url.path
            metrics.requests_total.labels(
                agent=agent, backend=backend, route=route,
                status=str(status),
            ).inc()
            metrics.request_duration.labels(
                agent=agent, backend=backend, route=route,
            ).observe(elapsed)
        return resp
