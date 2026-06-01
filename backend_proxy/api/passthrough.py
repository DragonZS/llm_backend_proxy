"""Catch-all router that passes any unmatched request straight to the chosen
backend. M0 picks the first backend; M2 will route by model affinity / strategy."""

from __future__ import annotations

import orjson
from fastapi import APIRouter, Depends, Request, Response

from ..backends import BackendPool
from ..backends.retry import retry_passthrough
from ..config.schema import RetryConfig
from ..core.errors import ProxyError, render_error
from .deps import get_pool

router = APIRouter(tags=["passthrough"])


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def passthrough(
    path: str,
    request: Request,
    pool: BackendPool = Depends(get_pool),
) -> Response:
    body = await request.body() if request.method not in ("GET", "HEAD", "OPTIONS") else None

    # Try to detect a model field for routing — silent best effort.
    model: str | None = None
    if body and request.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = orjson.loads(body)
            if isinstance(payload, dict) and isinstance(payload.get("model"), str):
                model = payload["model"]
        except orjson.JSONDecodeError:
            pass

    # Resolve the effective retry config from routing-level default.
    retry_cfg = getattr(pool.routing, "retry", None) or RetryConfig()

    try:
        resp, backend = await retry_passthrough(
            pool, model=model,
            method=request.method, path="/" + path,
            body=body,
            headers={k: v for k, v in request.headers.items()},
            retry_cfg=retry_cfg,
        )
    except ProxyError as exc:
        return render_error(exc)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp.headers,
    )
