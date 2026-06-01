"""Operational endpoints: liveness, readiness, version, metrics, reload."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse

from .. import __version__
from ..backends import BackendPool
from .deps import get_pool

router = APIRouter(tags=["admin"])


@router.get("/healthz", response_class=ORJSONResponse)
async def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/readyz", response_class=ORJSONResponse)
async def readyz(pool: BackendPool = Depends(get_pool)) -> dict:
    statuses = {}
    for b in pool.backends:
        try:
            statuses[b.name] = await b.health()
        except Exception:  # noqa: BLE001
            statuses[b.name] = False
    return {
        "status": "ok" if any(statuses.values()) else "degraded",
        "backends": statuses,
    }


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    m = request.app.state.metrics
    if m.enabled:
        for b in request.app.state.pool.backends:
            m.inflight.labels(backend=b.name).set(getattr(b, "_inflight", 0))
    body, ctype = m.render()
    return Response(content=body, media_type=ctype)


@router.post("/admin/reload", response_class=ORJSONResponse)
async def admin_reload(request: Request,
                       authorization: str | None = Header(default=None)) -> dict:
    """Reload the YAML config at ``BACKEND_PROXY_CONFIG``. Optionally protected
    by ``BACKEND_PROXY_ADMIN_TOKEN`` (deprecated alias: ``VLLM_PROXY_ADMIN_TOKEN``)
    — if set, the caller must send ``Authorization: Bearer <token>``."""
    expected = (
        os.environ.get("BACKEND_PROXY_ADMIN_TOKEN")
        or os.environ.get("VLLM_PROXY_ADMIN_TOKEN")
    )
    if expected:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid admin token")
    reloader = request.app.state.reloader
    return await reloader.reload()
