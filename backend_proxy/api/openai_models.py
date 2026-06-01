"""GET /v1/models — aggregate across all configured backends."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse

from ..backends import BackendPool
from .deps import get_pool

router = APIRouter(tags=["models"])


@router.get("/v1/models", response_class=ORJSONResponse)
async def list_models(pool: BackendPool = Depends(get_pool)) -> dict:
    return await pool.list_models_aggregated()
