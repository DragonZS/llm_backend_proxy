"""POST /v1/responses — runs an adapter pipeline that translates a Responses
request into a Chat-Completions one, calls the upstream, and translates the
result back. Streaming and non-streaming both supported.

In M1 the pipeline is hard-wired to the "Codex" set of adapters. M3 will
swap in agent-profile resolution at the top of the handler.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from ..adapters import AdapterPipeline
from ..adapters.chat_to_responses import (
    _close_text_item,
    _close_tool_calls,
    _new_state,
    chat_chunk_to_responses_events,
)
from ..adapters.stuck_rescue import (
    build_nudge_messages,
    looks_stuck,
)
from ..backends import BackendPool
from ..backends.retry import retry_request_json, retry_stream_start
from ..config.schema import RetryConfig
from ..core.context import RequestContext
from ..core.errors import BadRequestError, NoBackendAvailable, ProxyError, render_error
from ..streaming import parse_sse_lines, serialise_events, with_heartbeat, with_timeout
from .deps import get_pool, get_resolver, make_context

router = APIRouter(tags=["responses"])
log = logging.getLogger("backend_proxy.api.responses")


@router.post("/v1/responses")
async def create_response(
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
        return render_error(NoBackendAvailable("no agent profile matched /v1/responses"))
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
    if "authorization" in ctx.client_headers:
        auth_headers["authorization"] = ctx.client_headers["authorization"]

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

        log.info(
            "[%s] /v1/responses model=%s stream=%s backend=%s msgs=%d",
            ctx.trace_id, chat_req.get("model"), streaming, ctx.backend_name,
            len(chat_req.get("messages") or []),
        )

        result = await pipeline.apply_response(chat_full, ctx)
        return ORJSONResponse(result, headers={
            "x-request-id": ctx.trace_id,
            "x-backend-proxy-agent": ctx.agent_name or "-",
            "x-backend-proxy-backend": ctx.backend_name or "-",
        })

    # Streaming path: parse upstream SSE chunks, run them through stream
    # adapters, re-serialise to the client.
    #
    # OPTIONAL: when BACKEND_PROXY_AUTO_NUDGE=1, route the streaming
    # request through the inline-rescue path. The client still gets
    # token-by-token streaming on the happy path; only when the model
    # finishes a turn with finish=stop / 0 tool_calls / a future-tense
    # announcement does the proxy transparently issue a second upstream
    # call and splice its events into the same SSE connection. Default
    # is OFF — set BACKEND_PROXY_AUTO_NUDGE=1 to opt in.
    nudge_enabled = os.environ.get("BACKEND_PROXY_AUTO_NUDGE") == "1"
    if nudge_enabled:
        return await _streaming_with_nudge(
            request, ctx, pool, pipeline, chat_req, auth_headers, retry_cfg, log,
        )

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

    log.info(
        "[%s] /v1/responses model=%s stream=%s backend=%s msgs=%d",
        ctx.trace_id, chat_req.get("model"), streaming, ctx.backend_name,
        len(chat_req.get("messages") or []),
    )
    # Debug observability: log how many tools the upstream is being told about
    # and the names. When Codex sends 0 function tools (only `namespace`
    # / `web_search` etc., which we drop), the model has nothing to call and
    # WILL produce a chatty reply with no tool_calls. That's the most common
    # cause of "agent talks but doesn't act" — surface it explicitly so
    # operators don't have to guess.
    _tools = chat_req.get("tools") or []
    if not _tools:
        log.warning(
            "[%s] no function tools forwarded to upstream — "
            "model cannot call a tool and will only produce text. "
            "If the client thinks it sent tools, they were either "
            "non-`function` types (dropped) or malformed.",
            ctx.trace_id,
        )
    else:
        log.info(
            "[%s] forwarding %d tool(s) to upstream: %s",
            ctx.trace_id, len(_tools),
            ", ".join((t.get("function") or {}).get("name", "?") for t in _tools),
        )

    # OPT-IN deep dump: when env BACKEND_PROXY_DUMP_REQ=1 is set, write the
    # full chat_req for THIS trace to /tmp/proxy_req_<trace>.json so an
    # operator can replay it directly against vLLM and compare. Off by
    # default to avoid leaking conversation history into disk.
    import os as _os, orjson as _oj  # noqa: PLC0415
    if _os.environ.get("BACKEND_PROXY_DUMP_REQ") == "1":
        _dump_path = f"/tmp/proxy_req_{ctx.trace_id}.json"
        try:
            with open(_dump_path, "wb") as f:
                f.write(_oj.dumps(chat_req, option=_oj.OPT_INDENT_2))
            log.info("[%s] dumped chat_req to %s", ctx.trace_id, _dump_path)
        except Exception as _e:  # noqa: BLE001
            log.warning("[%s] failed to dump chat_req: %s", ctx.trace_id, _e)

    # Apply stream timeouts to the raw upstream byte stream.
    stream_cfg = request.app.state.config.server.stream_timeout
    abs_timeout = stream_cfg.absolute_timeout_s or request.app.state.config.server.request_timeout
    if stream_cfg.idle_timeout_s > 0 or abs_timeout > 0:
        upstream_bytes = with_timeout(
            upstream_bytes,
            idle_timeout_s=stream_cfg.idle_timeout_s,
            absolute_timeout_s=abs_timeout,
        )

    parsed: AsyncIterator[Dict[str, Any]] = parse_sse_lines(upstream_bytes)
    transformed = pipeline.apply_stream(parsed, ctx)

    async def _gen() -> AsyncIterator[bytes]:
        # Track whether the upstream produced a clean terminator so we can
        # surface a diagnostic line when codex (or any client) reports a
        # disconnect. The stream adapter sets `ctx.stream_aborted` when it
        # had to synthesise an `incomplete` completion event.
        try:
            async for piece in serialise_events(transformed):
                yield piece
        finally:
            aborted = getattr(ctx, "stream_aborted", False)
            reason = getattr(ctx, "stream_abort_reason", None)
            # Telemetry stashed by the chat_to_responses_stream adapter.
            fin = ctx.extras.get("upstream_finish_reason")
            n_tool = ctx.extras.get("upstream_n_tool_calls", 0)
            n_text = ctx.extras.get("upstream_n_text_chars", 0)
            n_reas = ctx.extras.get("upstream_n_reasoning_chars", 0)

            if aborted:
                log.warning(
                    "[%s] stream salvaged: backend=%s reason=%s "
                    "(finish=%s text=%d reasoning=%d tools=%d) — "
                    "client got synthetic response.completed with status=incomplete",
                    ctx.trace_id, ctx.backend_name, reason,
                    fin, n_text, n_reas, n_tool,
                )
            else:
                # Surface the "agent will appear stuck" pattern: model produced
                # output but no tool call, and what it did produce was reasoning
                # rather than visible content. This is the canonical "Codex shows
                # an empty turn and the user says '断了'" symptom — log at
                # WARNING so it's discoverable without deep tracing.
                stuck_pattern = (
                    n_tool == 0 and n_text == 0 and n_reas > 0
                ) or (
                    n_tool == 0 and fin == "length"
                )
                level = log.warning if stuck_pattern else log.info
                level(
                    "[%s] stream closed cleanly: backend=%s "
                    "finish=%s text=%d reasoning=%d tools=%d%s",
                    ctx.trace_id, ctx.backend_name,
                    fin, n_text, n_reas, n_tool,
                    "  ⚠️ NO_TOOL_CALL_BUT_REASONING — model talked itself out of acting; "
                    "client will see an idle turn. Consider raising max_tokens or "
                    "adjusting the system prompt to be more action-oriented."
                    if stuck_pattern else "",
                )
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


# ---------------------------------------------------------------------------
# Streaming-with-nudge path (BACKEND_PROXY_AUTO_NUDGE=1)
# ---------------------------------------------------------------------------

async def _streaming_with_nudge(
    request: Request,
    ctx: RequestContext,
    pool: BackendPool,
    pipeline: AdapterPipeline,
    chat_req: Dict[str, Any],
    auth_headers: Dict[str, str],
    retry_cfg: RetryConfig,
    log: logging.Logger,
):
    """Inline streaming nudge: stream the FIRST upstream response token-by-
    token to the client. If the stream ends with finish=stop / 0 tool_calls
    / a future-tense announcement (the 'agent talks but doesn't act'
    pattern), withhold the terminal ``response.completed`` and instead
    transparently issue a SECOND upstream call with a nudge, splicing the
    second response's events into the same SSE connection. The client sees
    one continuous turn — the model 'remembered' to act after talking.

    Trade-offs vs. the buffered approach:
      * Token-by-token streaming preserved on the happy path (the common case)
      * On rescue, the user sees a brief pause while the second call starts —
        the heartbeat keeps the connection alive
      * One state object is shared across both upstream calls so output_index,
        resp_id, item_id, and tool_call ordering all stay consistent
    """
    # Open the first upstream stream.
    try:
        upstream_bytes, backend, slot = await retry_stream_start(
            pool, model=chat_req.get("model"),
            method="POST", path="/v1/chat/completions",
            json=chat_req, headers=auth_headers, retry_cfg=retry_cfg,
        )
        ctx.backend_name = backend.name
    except ProxyError as exc:
        return render_error(exc)

    log.info(
        "[%s] /v1/responses (auto-nudge inline) model=%s backend=%s msgs=%d",
        ctx.trace_id, chat_req.get("model"), ctx.backend_name,
        len(chat_req.get("messages") or []),
    )

    stream_cfg = request.app.state.config.server.stream_timeout
    abs_timeout = stream_cfg.absolute_timeout_s or request.app.state.config.server.request_timeout
    if stream_cfg.idle_timeout_s > 0 or abs_timeout > 0:
        upstream_bytes = with_timeout(
            upstream_bytes,
            idle_timeout_s=stream_cfg.idle_timeout_s,
            absolute_timeout_s=abs_timeout,
        )

    body_iter = _inline_rescue_stream(
        request, ctx, pool, chat_req, auth_headers, retry_cfg, log,
        upstream_bytes, slot,
    )
    hb = request.app.state.config.server.heartbeat
    if hb.enabled:
        body_iter = with_heartbeat(body_iter, interval=hb.interval_s, payload=hb.payload)

    return StreamingResponse(body_iter, media_type="text/event-stream", headers={
        "cache-control": "no-cache",
        "x-request-id": ctx.trace_id,
        "x-backend-proxy-trace": ctx.trace_id,
        "x-backend-proxy-agent": ctx.agent_name or "-",
        "x-backend-proxy-backend": ctx.backend_name or "-",
        "x-backend-proxy-mode": "auto-nudge-inline",
    })


def _emit_bytes(obj: Dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(obj) + b"\n\n"


async def _consume_one_stream(
    upstream_bytes: AsyncIterator[bytes],
    state: Dict[str, Any],
    *,
    suppress_completed: bool,
) -> AsyncIterator[bytes]:
    """Consume one upstream chat-completions SSE stream, drive the shared
    Responses-API ``state``, and yield bytes ready for the client.

    When ``suppress_completed`` is True, the terminal ``response.completed``
    event from ``chat_chunk_to_responses_events`` is intercepted (and
    ``state['completed']`` rolled back) so the caller can decide whether to
    splice in a rescue stream before finalising. Other terminal-aspect
    events (output_text.done, content_part.done, output_item.done,
    function_call_arguments.done, etc.) ARE still emitted — once the text
    block is closed we won't add more text to it; the rescue contributes
    a NEW message item with a fresh output_index.
    """
    async for chunk_dict in parse_sse_lines(upstream_bytes):
        for evt_name, obj in chat_chunk_to_responses_events(chunk_dict, state):
            if suppress_completed and evt_name == "response.completed":
                # Roll back: we'll either re-emit completion later (after
                # a rescue) or, if no rescue is needed, emit it as the very
                # last event from the caller.
                state["completed"] = False
                continue
            yield _emit_bytes(obj)


def _seal_first_phase_for_rescue(
    state: Dict[str, Any],
) -> AsyncIterator[bytes]:
    """Generator that yields the events needed to cleanly close out the
    first phase's items so the rescue can append a fresh message item.

    Currently the Chat-Completions stream's ``finish_reason`` already
    triggered ``_close_text_item``/``_close_tool_calls`` inside
    ``chat_chunk_to_responses_events``, so this helper is a no-op
    placeholder kept for symmetry — if future state changes require
    explicit sealing, do it here.
    """
    if False:  # pragma: no cover - intentional empty generator
        yield b""


def _reset_state_for_rescue_phase(state: Dict[str, Any]) -> None:
    """Reset the ``state`` fields that govern message-item lifecycle so
    the rescue phase can produce a NEW assistant message item (with a
    fresh ``item_id`` and the next ``output_index``) without colliding
    with the just-closed first-phase item. Tool-call slots and the
    ``resp_id`` are preserved so any tool calls the rescue makes are
    appended to the same response and so the client keeps the same
    ``response.id``."""
    # Reset message-item state
    state["text"] = ""
    state["phase"] = None
    state["msg_opened"] = False
    state["msg_closed"] = False
    state["item_id"] = _new_state()["item_id"]  # fresh id for the next item
    # Don't reset state['started']: response.created already fired.
    # Don't reset state['resp_id']: same response continues.
    # Don't reset state['next_output_index']: it already advanced past the
    #   first message item. Tool-call slots from phase one (if any) keep
    #   their output_index.
    # Don't reset tool_calls: any tool_calls from phase one stay valid.
    state["completed"] = False
    state["telemetry_finish_reason"] = None


async def _inline_rescue_stream(
    request: Request,
    ctx: RequestContext,
    pool: BackendPool,
    chat_req: Dict[str, Any],
    auth_headers: Dict[str, str],
    retry_cfg: RetryConfig,
    log: logging.Logger,
    upstream_bytes: AsyncIterator[bytes],
    slot,
) -> AsyncIterator[bytes]:
    """Async generator powering the inline streaming nudge."""
    state = _new_state()
    rescue_attempted = False
    rescue_succeeded = False

    try:
        # ---------- phase 1: stream the original response ----------
        try:
            async for piece in _consume_one_stream(
                upstream_bytes, state, suppress_completed=True,
            ):
                yield piece
        except BaseException as exc:  # noqa: BLE001
            # Match the non-nudge handler's salvage behaviour: emit a
            # terminal response.completed with status=incomplete so the
            # client doesn't hang.
            log.warning(
                "[%s] auto-nudge: phase-1 stream errored (%s) — emitting salvage",
                ctx.trace_id, type(exc).__name__,
            )
            yield _emit_bytes({
                "type": "response.completed",
                "response": {
                    "id": state["resp_id"], "object": "response",
                    "status": "incomplete",
                    "incomplete_details": {"reason": f"upstream_stream_error:{type(exc).__name__}"},
                    "model": chat_req.get("model"),
                    "output": _state_to_output(state),
                    "usage": None,
                },
            })
            yield b"data: [DONE]\n\n"
            return

        # ---------- decision: do we need a rescue? ----------
        finish = state.get("telemetry_finish_reason")
        n_tool = len(state.get("tool_calls") or {})
        # state["text"] holds the visible reasoning + content concatenation;
        # for stuck detection the visible content portion is what matters.
        # Use raw chars from telemetry so reasoning-only output isn't
        # mistakenly classified as an announcement.
        n_text = state.get("telemetry_n_text_chars", 0)
        text_for_match = state.get("text", "") if n_text > 0 else ""

        stuck = looks_stuck(finish, text_for_match, n_tool)
        if stuck:
            rescue_attempted = True
            log.warning(
                "[%s] auto-nudge: detected stuck pattern after phase-1 "
                "(finish=%s tools=%d text-prefix=%r) — issuing rescue stream",
                ctx.trace_id, finish, n_tool, text_for_match[:120],
            )

            # ---------- phase 2: rescue stream ----------
            nudged_msgs = build_nudge_messages(
                chat_req["messages"], text_for_match,
            )
            nudged_req = {**chat_req, "messages": nudged_msgs, "stream": True}

            # Reset the lifecycle bits so a NEW message item can open.
            _reset_state_for_rescue_phase(state)

            try:
                rescue_bytes, _b, rescue_slot = await retry_stream_start(
                    pool, model=nudged_req.get("model"),
                    method="POST", path="/v1/chat/completions",
                    json=nudged_req, headers=auth_headers,
                    retry_cfg=retry_cfg,
                )
            except ProxyError as exc:
                log.warning(
                    "[%s] auto-nudge: rescue stream failed to start: %s — "
                    "finalising original response",
                    ctx.trace_id, exc,
                )
                rescue_slot = None
            else:
                # Apply same stream-timeout knobs as phase 1
                stream_cfg = request.app.state.config.server.stream_timeout
                abs_timeout = (
                    stream_cfg.absolute_timeout_s
                    or request.app.state.config.server.request_timeout
                )
                if stream_cfg.idle_timeout_s > 0 or abs_timeout > 0:
                    rescue_bytes = with_timeout(
                        rescue_bytes,
                        idle_timeout_s=stream_cfg.idle_timeout_s,
                        absolute_timeout_s=abs_timeout,
                    )

                try:
                    async for piece in _consume_one_stream(
                        rescue_bytes, state, suppress_completed=False,
                    ):
                        yield piece
                    rescue_succeeded = True
                    log.info(
                        "[%s] auto-nudge: rescue stream consumed cleanly "
                        "(final_finish=%s, total_tools=%d)",
                        ctx.trace_id,
                        state.get("telemetry_finish_reason"),
                        len(state.get("tool_calls") or {}),
                    )
                except BaseException as exc:  # noqa: BLE001
                    log.warning(
                        "[%s] auto-nudge: rescue stream errored (%s) — "
                        "finalising what we have",
                        ctx.trace_id, type(exc).__name__,
                    )
                finally:
                    try:
                        await rescue_slot.__aexit__(None, None, None)
                    except Exception:  # noqa: BLE001
                        pass

        # ---------- finalise ----------
        # If rescue ran and emitted its own response.completed (suppress=False),
        # we're done. Otherwise we need to emit a terminal response.completed.
        if not state.get("completed"):
            yield _emit_bytes({
                "type": "response.completed",
                "response": {
                    "id": state["resp_id"], "object": "response",
                    "status": "completed",
                    "model": chat_req.get("model"),
                    "output": _state_to_output(state),
                    "usage": None,
                },
            })
        yield b"data: [DONE]\n\n"
    finally:
        try:
            await slot.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
        if rescue_attempted:
            log.info(
                "[%s] auto-nudge summary: rescue %s",
                ctx.trace_id,
                "succeeded" if rescue_succeeded else "skipped/failed",
            )


def _state_to_output(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct the cumulative ``output[]`` array for a terminal
    response.completed event from the streaming state."""
    items: List[tuple] = []
    if state.get("msg_opened"):
        items.append((state["msg_output_index"], {
            "id": state["item_id"], "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state.get("text", "")}],
        }))
    for slot in (state.get("tool_calls") or {}).values():
        items.append((slot["output_index"], {
            "id": slot["item_id"], "type": "function_call",
            "status": "completed",
            "call_id": slot["call_id"],
            "name": slot["name"],
            "arguments": slot["arguments"],
        }))
    items.sort(key=lambda kv: kv[0])
    return [item for _, item in items]
