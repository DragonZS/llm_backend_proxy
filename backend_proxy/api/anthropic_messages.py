"""POST /v1/messages — Anthropic-compatible endpoint for Claude Code.

Mirrors ``/v1/responses`` structurally; the difference is just which adapters
the agent profile binds.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from ..backends import BackendPool
from ..backends.retry import retry_request_json, retry_stream_start
from ..config.schema import RetryConfig
from ..core.context import RequestContext
from ..core.errors import BadRequestError, NoBackendAvailable, ProxyError, render_error
from ..streaming import parse_sse_lines, with_heartbeat, with_timeout
from .deps import get_pool, get_resolver, make_context

router = APIRouter(tags=["anthropic-messages"])
log = logging.getLogger("backend_proxy.api.messages")


async def _serialise_anthropic_events(events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[bytes]:
    """Anthropic streams use ``event: <type>\\ndata: {…}\\n\\n`` framing."""
    async for obj in events:
        ev = obj.get("type") or "message"
        yield (f"event: {ev}\n").encode() + b"data: " + orjson.dumps(obj) + b"\n\n"


@router.post("/v1/messages")
async def create_message(
    request: Request,
    pool: BackendPool = Depends(get_pool),
    ctx: RequestContext = Depends(make_context),
):
    raw = await request.body()
    try:
        body: Dict[str, Any] = orjson.loads(raw or b"{}")
    except orjson.JSONDecodeError as e:
        return render_error(BadRequestError(f"bad json: {e}"))
    if not isinstance(body, dict):
        return render_error(BadRequestError("request body must be a JSON object"))

    streaming = bool(body.get("stream"))
    ctx.stream = streaming
    ctx.model = body.get("model") if isinstance(body.get("model"), str) else None

    resolver = get_resolver(request)
    agent = resolver.resolve(
        path=request.url.path,
        headers={k: v for k, v in request.headers.items()},
        query_params=dict(request.query_params),
    )
    if agent is None:
        return render_error(NoBackendAvailable("no agent profile matched /v1/messages"))
    resolver.update_context(ctx, agent)
    pipeline = agent.pipeline

    try:
        chat_req = await pipeline.apply_request(body, ctx)
    except ProxyError as exc:
        return render_error(exc)

    chat_req["stream"] = streaming

    # Resolve the effective retry config from routing-level default.
    retry_cfg = getattr(pool.routing, "retry", None) or RetryConfig()

    auth_headers = {}
    if "authorization" in {k.lower() for k in ctx.client_headers}:
        auth_headers["authorization"] = ctx.client_headers.get("authorization", "")
    if "x-api-key" in {k.lower() for k in ctx.client_headers}:
        # Anthropic clients use x-api-key — map to Authorization for vLLM.
        auth_headers.setdefault("authorization", "Bearer " + ctx.client_headers.get("x-api-key", ""))

    if not streaming:
        try:
            chat_full, backend = await retry_request_json(
                pool, model=chat_req.get("model"),
                method="POST", path="/v1/chat/completions",
                json=chat_req, headers=auth_headers,
                retry_cfg=retry_cfg,
            )
            ctx.backend_name = backend.name
        except ProxyError as exc:
            return render_error(exc)

        log.info("[%s] /v1/messages model=%s stream=%s backend=%s",
                 ctx.trace_id, chat_req.get("model"), streaming, ctx.backend_name)

        result = await pipeline.apply_response(chat_full, ctx)
        return ORJSONResponse(result, headers={
            "x-request-id": ctx.trace_id,
            "x-backend-proxy-agent": ctx.agent_name or "-",
            "x-backend-proxy-backend": ctx.backend_name or "-",
        })

    # Streaming path
    try:
        upstream_bytes, backend, slot = await retry_stream_start(
            pool, model=chat_req.get("model"),
            method="POST", path="/v1/chat/completions",
            json=chat_req, headers=auth_headers,
            retry_cfg=retry_cfg,
        )
        ctx.backend_name = backend.name
    except ProxyError as exc:
        return render_error(exc)

    log.info("[%s] /v1/messages model=%s stream=%s backend=%s",
             ctx.trace_id, chat_req.get("model"), streaming, ctx.backend_name)

    # Apply stream timeouts to the raw upstream byte stream.
    stream_cfg = request.app.state.config.server.stream_timeout
    abs_timeout = stream_cfg.absolute_timeout_s or request.app.state.config.server.request_timeout
    if stream_cfg.idle_timeout_s > 0 or abs_timeout > 0:
        upstream_bytes = with_timeout(
            upstream_bytes,
            idle_timeout_s=stream_cfg.idle_timeout_s,
            absolute_timeout_s=abs_timeout,
        )

    parsed = parse_sse_lines(upstream_bytes)
    transformed = pipeline.apply_stream(parsed, ctx)

    async def _gen() -> AsyncIterator[bytes]:
        try:
            async for piece in _serialise_anthropic_events(transformed):
                yield piece
        finally:
            await slot.__aexit__(None, None, None)

    body_iter = _gen()
    hb = request.app.state.config.server.heartbeat
    if hb.enabled:
        body_iter = with_heartbeat(body_iter, interval=hb.interval_s, payload=hb.payload)

    return StreamingResponse(body_iter, media_type="text/event-stream", headers={
        "cache-control": "no-cache",
        "x-request-id": ctx.trace_id,
        "x-backend-proxy-trace": ctx.trace_id,
        "x-backend-proxy-agent": ctx.agent_name or "-",
        "x-backend-proxy-backend": ctx.backend_name or "-",
    })
