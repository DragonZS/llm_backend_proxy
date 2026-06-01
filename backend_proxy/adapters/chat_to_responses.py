"""Translate Chat-Completions output (full and streaming) into Responses-API events.

State machine for streaming reasoning:

    None  -> "thinking"   (first reasoning chunk: prepend "<think>\\n")
    "thinking" -> "content" (first real content chunk: emit "\\n</think>\\n\\n" first)

If a finish_reason arrives while still inside the think block, we close the
``</think>`` tag in the same delta event so the client never sees a half-open
think block.

Tool calls
----------
The Chat-Completions API emits tool calls under ``message.tool_calls`` (full)
or ``delta.tool_calls`` (stream). The Responses API surfaces each call as a
separate output item of ``type: "function_call"`` with ``call_id``,
``name``, and ``arguments``. Both directions are translated:

* full   : ``tool_calls`` -> additional ``output[]`` entries of type
  ``function_call`` alongside any text message.
* stream : per-call lifecycle events
    response.output_item.added              (function_call shell)
    response.function_call_arguments.delta  (arguments tokens)
    response.function_call_arguments.done   (final arguments string)
    response.output_item.done               (completed function_call)

Without these events, codex / cursor-style clients see ``response.completed``
with no function_call item and treat the turn as an empty answer — which is
exactly the "model said it would write code, then nothing came back" symptom.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Tuple

from ..core.context import RequestContext
from ..utils.ids import new_message_id, new_response_id
from .base import ResponseAdapter, StreamAdapter


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------

class ChatToResponsesAdapter(ResponseAdapter):
    """Wrap a Chat-Completions JSON body in the Responses-API envelope."""
    name = "chat_to_responses"

    async def transform(self, chat: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        return chat_full_to_responses(chat)


def _function_call_item(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Chat-API tool_call dict into a Responses-API function_call item."""
    fn = tc.get("function") or {}
    return {
        "id": new_message_id(),
        "type": "function_call",
        "status": "completed",
        "call_id": tc.get("id") or "call_0",
        "name": fn.get("name") or "",
        "arguments": fn.get("arguments") or "",
    }


def _responses_usage(chat_usage: Any) -> Any:
    """Map Chat-Completions usage names to Responses API usage names."""
    if not isinstance(chat_usage, dict):
        return chat_usage
    if "input_tokens" in chat_usage or "output_tokens" in chat_usage:
        return chat_usage
    out: Dict[str, Any] = {}
    if "prompt_tokens" in chat_usage:
        out["input_tokens"] = chat_usage["prompt_tokens"]
    if "completion_tokens" in chat_usage:
        out["output_tokens"] = chat_usage["completion_tokens"]
    if "total_tokens" in chat_usage:
        out["total_tokens"] = chat_usage["total_tokens"]
    for key, value in chat_usage.items():
        if key not in ("prompt_tokens", "completion_tokens", "total_tokens"):
            out[key] = value
    return out


def chat_full_to_responses(chat: Dict[str, Any]) -> Dict[str, Any]:
    msg = (chat.get("choices") or [{}])[0].get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    tool_calls = msg.get("tool_calls") or []

    if reasoning and content:
        text = f"<think>\n{reasoning}\n</think>\n\n{content}"
    elif reasoning:
        text = f"<think>\n{reasoning}\n</think>\n"
    else:
        text = content

    rid = new_response_id()
    output: List[Dict[str, Any]] = []

    # Emit a text message item only when there is actual text. Tool-only
    # responses (the model went straight to a function_call without any
    # accompanying text) should not carry an empty message item.
    if text:
        output.append({
            "id": new_message_id(),
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        })
    for tc in tool_calls:
        output.append(_function_call_item(tc))

    return {
        "id": rid,
        "object": "response",
        "status": "completed",
        "model": chat.get("model"),
        "created_at": chat.get("created", int(time.time())),
        "output": output,
        "usage": _responses_usage(chat.get("usage")),
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class ChatToResponsesStreamAdapter(StreamAdapter):
    """Convert a stream of chat.completion.chunk dicts to Responses SSE events.

    Crucially, this adapter GUARANTEES a terminal ``response.completed`` event
    is emitted even when the upstream stream ends without a ``finish_reason``
    (network blip, vLLM restart mid-decode, httpx ReadError, etc.) — without
    it, codex-style clients hang on "stream disconnected before completion".
    A synthetic completion is emitted with ``status="incomplete"`` and
    ``incomplete_details.reason`` recording why we had to fabricate it, so the
    caller can distinguish a clean finish from a salvaged one.
    """
    name = "chat_to_responses_stream"

    async def transform(
        self, chunks: AsyncIterator[Dict[str, Any]], ctx: RequestContext
    ) -> AsyncIterator[Dict[str, Any]]:
        state = _new_state()
        reason = "upstream_ended_without_finish"
        try:
            async for chunk in chunks:
                for _evt, obj in chat_chunk_to_responses_events(chunk, state):
                    yield obj
        except BaseException as exc:  # noqa: BLE001
            # Cancellation, ReadError, RemoteProtocolError, etc. We still want
            # to emit a salvage event so the client side terminates cleanly,
            # then re-raise so upper layers see the failure.
            reason = f"upstream_stream_error:{type(exc).__name__}"
            try:
                for _evt, obj in _force_close_events(state, reason=reason):
                    yield obj
            finally:
                # Stash the failure on the context so the API handler can log it.
                try:
                    ctx.stream_aborted = True  # type: ignore[attr-defined]
                    ctx.stream_abort_reason = reason  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            raise
        # Normal completion path — only synthesise if upstream forgot to.
        if not state["completed"]:
            for _evt, obj in _force_close_events(state, reason=reason):
                yield obj
        # Stash telemetry for the api layer to log. This is the single most
        # useful debug signal for "agent stuck — model said it would act but
        # did nothing": when finish_reason=length we know vLLM ran out of
        # tokens mid-tool-call and the parser failed to extract; when
        # n_tool_calls=0 with reasoning>0 we know the model went into
        # chain-of-thought and forgot to actually emit a tool call.
        try:
            ctx.extras["upstream_finish_reason"] = state.get("telemetry_finish_reason")
            ctx.extras["upstream_n_text_chars"] = state.get("telemetry_n_text_chars")
            ctx.extras["upstream_n_reasoning_chars"] = state.get("telemetry_n_reasoning_chars")
            ctx.extras["upstream_n_tool_calls"] = len(state.get("tool_calls") or {})
        except Exception:  # noqa: BLE001
            pass


def _new_state() -> Dict[str, Any]:
    return {
        "started": False,
        "completed": False,           # has response.completed been emitted?
        "text": "",
        "phase": None,                # None | "thinking" | "content"
        "resp_id": new_response_id(),
        "item_id": new_message_id(),  # reserved for the (optional) text message
        "msg_opened": False,          # have we emitted output_item.added for text?
        "msg_closed": False,
        # tool-call streaming state, keyed by Chat-API delta `index`
        "tool_calls": {},             # idx -> {output_index, item_id, call_id, name, arguments}
        "next_output_index": 0,       # 0 reserved for the text message (if any)
        # Telemetry: surfaced in the api log so an agent stuck in "model talked
        # but did nothing" is debuggable. Counts what the upstream actually
        # produced regardless of whether the client recognised it.
        "telemetry_finish_reason": None,
        "telemetry_n_text_chars": 0,
        "telemetry_n_reasoning_chars": 0,
    }


def _force_close_events(
    state: Dict[str, Any], *, reason: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Emit whatever events are needed to close any half-open items and a
    terminal ``response.completed`` with ``status="incomplete"``.

    Idempotent: if ``state["completed"]`` is already True, returns ``[]``."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if state["completed"]:
        return out

    # If we never even started, the client is waiting for response.created
    # too — emit a minimal one so the envelope is well-formed.
    if not state["started"]:
        state["started"] = True
        out.append(("response.created", {
            "type": "response.created",
            "response": {
                "id": state["resp_id"], "object": "response",
                "status": "in_progress", "model": None, "output": [],
            },
        }))

    # Close <think> if we were inside it.
    if state["phase"] == "thinking":
        _open_text_item(state, None, out)
        tail = "\n</think>\n"
        state["text"] += tail
        out.append(("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": state["item_id"],
            "output_index": state["msg_output_index"], "content_index": 0,
            "delta": tail,
        }))
        state["phase"] = "content"

    _close_text_item(state, out)
    _close_tool_calls(state, out)

    state["completed"] = True
    out.append(("response.completed", {
        "type": "response.completed",
        "response": {
            "id": state["resp_id"], "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": reason},
            "model": None,
            "output": _final_output_array(state),
            "usage": None,
        },
    }))
    return out


def _open_text_item(state: Dict[str, Any], chunk_model: Any, out: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Lazily open the assistant-message item the first time we have content
    or reasoning to emit. Tool-only turns skip this entirely."""
    if state["msg_opened"]:
        return
    state["msg_opened"] = True
    state["msg_output_index"] = state["next_output_index"]
    state["next_output_index"] += 1
    out.append(("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": state["msg_output_index"],
        "item": {
            "id": state["item_id"], "type": "message", "role": "assistant",
            "status": "in_progress", "content": [],
        },
    }))
    out.append(("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": state["item_id"],
        "output_index": state["msg_output_index"], "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    }))


def _close_text_item(state: Dict[str, Any], out: List[Tuple[str, Dict[str, Any]]]) -> None:
    if not state["msg_opened"] or state["msg_closed"]:
        return
    state["msg_closed"] = True
    out.append(("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": state["item_id"],
        "output_index": state["msg_output_index"], "content_index": 0,
        "text": state["text"],
    }))
    out.append(("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": state["item_id"],
        "output_index": state["msg_output_index"], "content_index": 0,
        "part": {"type": "output_text", "text": state["text"]},
    }))
    out.append(("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": state["msg_output_index"],
        "item": {
            "id": state["item_id"], "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state["text"]}],
        },
    }))


def _handle_tool_calls_delta(
    tcs: List[Dict[str, Any]], state: Dict[str, Any],
    out: List[Tuple[str, Dict[str, Any]]],
) -> None:
    """Translate ``delta.tool_calls`` chunks into per-call Responses events.

    Chat-API streams tool_calls as fragments keyed by an integer ``index``.
    The first chunk for an index typically carries ``id`` + ``function.name``
    (and possibly an empty ``arguments``); subsequent chunks for that index
    carry only ``function.arguments`` slices to be concatenated."""
    for tc in tcs or []:
        # Some chat templates / tool parsers don't emit a per-call `index` on
        # delta chunks; falling back to a fixed 0 would collapse every parallel
        # tool call onto a single slot — the client then sees only the last
        # one. Allocate a stable per-call counter when index is missing.
        idx = tc.get("index")
        if idx is None:
            tc_id = tc.get("id") or ""
            anon = state.setdefault("_anon_tc", {"by_id": {}, "next": 1_000_000})
            if tc_id and tc_id in anon["by_id"]:
                idx = anon["by_id"][tc_id]
            else:
                idx = anon["next"]
                anon["next"] += 1
                if tc_id:
                    anon["by_id"][tc_id] = idx
        slot = state["tool_calls"].get(idx)
        if slot is None:
            slot = {
                "output_index": state["next_output_index"],
                "item_id": new_message_id(),
                "call_id": tc.get("id") or f"call_{idx}",
                "name": (tc.get("function") or {}).get("name") or "",
                "arguments": "",
                "opened": False,
                "closed": False,
            }
            state["next_output_index"] += 1
            state["tool_calls"][idx] = slot
        else:
            # Refine fields if a later chunk fills them in.
            if tc.get("id") and slot["call_id"].startswith("call_"):
                slot["call_id"] = tc["id"]
            new_name = (tc.get("function") or {}).get("name")
            if new_name and not slot["name"]:
                slot["name"] = new_name

        # Emit the shell event the first time we know enough to identify the call.
        if not slot["opened"] and slot["name"]:
            slot["opened"] = True
            out.append(("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": slot["output_index"],
                "item": {
                    "id": slot["item_id"], "type": "function_call",
                    "status": "in_progress",
                    "call_id": slot["call_id"], "name": slot["name"],
                    "arguments": "",
                },
            }))

        # Stream the arguments fragment, if any.
        frag = (tc.get("function") or {}).get("arguments")
        if frag:
            slot["arguments"] += frag
            if slot["opened"]:
                out.append(("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": slot["item_id"],
                    "output_index": slot["output_index"],
                    "delta": frag,
                }))


def _close_tool_calls(state: Dict[str, Any], out: List[Tuple[str, Dict[str, Any]]]) -> None:
    for slot in state["tool_calls"].values():
        if slot["closed"]:
            continue
        # If the upstream never sent a name (shouldn't happen), open lazily
        # so the client at least sees a function_call item.
        if not slot["opened"]:
            slot["opened"] = True
            out.append(("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": slot["output_index"],
                "item": {
                    "id": slot["item_id"], "type": "function_call",
                    "status": "in_progress",
                    "call_id": slot["call_id"], "name": slot["name"],
                    "arguments": "",
                },
            }))
        slot["closed"] = True
        out.append(("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": slot["item_id"],
            "output_index": slot["output_index"],
            "arguments": slot["arguments"],
        }))
        out.append(("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": slot["output_index"],
            "item": {
                "id": slot["item_id"], "type": "function_call",
                "status": "completed",
                "call_id": slot["call_id"],
                "name": slot["name"],
                "arguments": slot["arguments"],
            },
        }))


def _final_output_array(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reassemble the full output[] for the terminal response.completed event."""
    items: List[Tuple[int, Dict[str, Any]]] = []
    if state["msg_opened"]:
        items.append((state["msg_output_index"], {
            "id": state["item_id"], "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state["text"]}],
        }))
    for slot in state["tool_calls"].values():
        items.append((slot["output_index"], {
            "id": slot["item_id"], "type": "function_call",
            "status": "completed",
            "call_id": slot["call_id"],
            "name": slot["name"],
            "arguments": slot["arguments"],
        }))
    items.sort(key=lambda kv: kv[0])
    return [item for _, item in items]


def chat_chunk_to_responses_events(
    chat_chunk: Dict[str, Any], state: Dict[str, Any]
) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    choices = chat_chunk.get("choices") or []
    if not choices:
        return out
    delta = choices[0].get("delta") or {}
    finish = choices[0].get("finish_reason")

    # response.created — emit exactly once, before any item events.
    if not state["started"]:
        state["started"] = True
        out.append(("response.created", {
            "type": "response.created",
            "response": {
                "id": state["resp_id"], "object": "response", "status": "in_progress",
                "model": chat_chunk.get("model"), "output": [],
            },
        }))

    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
    content_delta = delta.get("content")
    tool_calls_delta = delta.get("tool_calls")

    if reasoning_delta:
        state["telemetry_n_reasoning_chars"] = state.get("telemetry_n_reasoning_chars", 0) + len(reasoning_delta)
    if content_delta:
        state["telemetry_n_text_chars"] = state.get("telemetry_n_text_chars", 0) + len(content_delta)
    if finish:
        state["telemetry_finish_reason"] = finish

    pieces: List[str] = []
    if reasoning_delta:
        if state["phase"] is None:
            pieces.append("<think>\n")
            state["phase"] = "thinking"
        pieces.append(reasoning_delta)
    if content_delta:
        if state["phase"] == "thinking":
            pieces.append("\n</think>\n\n")
        state["phase"] = "content"
        pieces.append(content_delta)

    if pieces:
        _open_text_item(state, chat_chunk.get("model"), out)
        for piece in pieces:
            state["text"] += piece
            out.append(("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": state["item_id"],
                "output_index": state["msg_output_index"], "content_index": 0,
                "delta": piece,
            }))

    if tool_calls_delta:
        _handle_tool_calls_delta(tool_calls_delta, state, out)

    if finish:
        # If we were still inside <think>, close it cleanly.
        if state["phase"] == "thinking":
            _open_text_item(state, chat_chunk.get("model"), out)
            tail = "\n</think>\n"
            state["text"] += tail
            out.append(("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": state["item_id"],
                "output_index": state["msg_output_index"], "content_index": 0,
                "delta": tail,
            }))
            state["phase"] = "content"

        _close_text_item(state, out)
        _close_tool_calls(state, out)

        state["completed"] = True
        out.append(("response.completed", {
            "type": "response.completed",
            "response": {
                "id": state["resp_id"], "object": "response", "status": "completed",
                "model": chat_chunk.get("model"),
                "output": _final_output_array(state),
                "usage": _responses_usage(chat_chunk.get("usage")),
            },
        }))
    return out
