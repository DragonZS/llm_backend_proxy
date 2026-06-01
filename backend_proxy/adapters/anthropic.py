"""Adapters bridging the Anthropic Messages API to OpenAI Chat Completions.

Anthropic request shape (subset we care about)::

    {
      "model": "claude-sonnet-4",
      "system": "you are helpful",            # or list[{type:text, text:...}]
      "messages": [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": [{"type":"text","text":"hi"},
                                     {"type":"tool_result", "tool_use_id":"x", "content":"42"}]}
      ],
      "tools": [{"name":"f", "description":"...", "input_schema": {...}}],
      "tool_choice": {"type": "auto"} | {"type": "tool", "name": "f"},
      "max_tokens": 4096,
      "temperature": 0.5,
      "stream": true
    }

Anthropic response shape (non-stream)::

    {
      "id": "msg_…",
      "type": "message",
      "role": "assistant",
      "model": "...",
      "content": [{"type": "text", "text": "..."},
                  {"type": "tool_use", "id":"x", "name":"f", "input":{...}}],
      "stop_reason": "end_turn" | "tool_use" | "max_tokens" | "stop_sequence",
      "usage": {"input_tokens": ..., "output_tokens": ...}
    }

Streaming uses event-typed SSE: ``message_start``, ``content_block_start``,
``content_block_delta`` (with ``delta: {type:"text_delta", text:"..."}``),
``content_block_stop``, ``message_delta`` (carries stop_reason+usage),
``message_stop``.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Tuple

from ..core.context import RequestContext
from ..core.errors import BadRequestError
from ..utils.ids import new_message_id
from .base import RequestAdapter, ResponseAdapter, StreamAdapter


# --------------------------------------------------------------------- request

class AnthropicToChatAdapter(RequestAdapter):
    name = "anthropic_to_chat"

    async def transform(self, req: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        if "model" not in req:
            raise BadRequestError("missing 'model' in request")

        msgs: List[Dict[str, Any]] = []

        sys_field = req.get("system")
        if isinstance(sys_field, str) and sys_field:
            msgs.append({"role": "system", "content": sys_field})
        elif isinstance(sys_field, list):
            text = "".join(
                p.get("text", "") for p in sys_field
                if isinstance(p, dict) and p.get("type") == "text"
            )
            if text:
                msgs.append({"role": "system", "content": text})

        for m in req.get("messages") or []:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                continue

            text_buf: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            tool_results: List[Tuple[str, str]] = []

            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_buf.append(part.get("text", ""))
                elif ptype == "tool_use" and role == "assistant":
                    tool_calls.append({
                        "id": part.get("id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": _json_dumps(part.get("input", {})),
                        },
                    })
                elif ptype == "tool_result":
                    tool_results.append((
                        part.get("tool_use_id", "call_0"),
                        _stringify(part.get("content", "")),
                    ))

            if role == "assistant":
                m_out: Dict[str, Any] = {"role": "assistant"}
                if text_buf:
                    m_out["content"] = "".join(text_buf)
                if tool_calls:
                    m_out["tool_calls"] = tool_calls
                if "content" in m_out or "tool_calls" in m_out:
                    msgs.append(m_out)
            else:
                if text_buf:
                    msgs.append({"role": role, "content": "".join(text_buf)})
                for tcid, out_text in tool_results:
                    msgs.append({"role": "tool", "tool_call_id": tcid, "content": out_text})

        out: Dict[str, Any] = {
            "model": req["model"],
            "messages": msgs,
            "stream": bool(req.get("stream")),
        }
        if "max_tokens" in req:
            out["max_tokens"] = req["max_tokens"]
        if "temperature" in req:
            out["temperature"] = req["temperature"]
        if "top_p" in req:
            out["top_p"] = req["top_p"]
        if "stop_sequences" in req:
            out["stop"] = req["stop_sequences"]

        if req.get("tools"):
            tools = []
            for t in req["tools"]:
                if not isinstance(t, dict) or "name" not in t:
                    continue
                tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    },
                })
            if tools:
                out["tools"] = tools
        tc = req.get("tool_choice")
        if tc:
            if isinstance(tc, dict) and tc.get("type") == "tool" and "name" in tc:
                out["tool_choice"] = {"type": "function",
                                      "function": {"name": tc["name"]}}
            elif isinstance(tc, dict) and tc.get("type") in ("auto", "any"):
                out["tool_choice"] = "auto"

        ctx.extras["protocol"] = "anthropic"
        return out


def _json_dumps(obj: Any) -> str:
    import orjson
    return orjson.dumps(obj).decode()


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


# --------------------------------------------------------------------- response (full)

_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


class ChatToAnthropicAdapter(ResponseAdapter):
    name = "chat_to_anthropic"

    async def transform(self, chat: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        msg = (chat.get("choices") or [{}])[0].get("message") or {}
        finish = (chat.get("choices") or [{}])[0].get("finish_reason") or "stop"
        content_blocks: List[Dict[str, Any]] = []

        text = msg.get("content") or ""
        if text:
            content_blocks.append({"type": "text", "text": text})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                import orjson
                args = orjson.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", "call_0"),
                "name": fn.get("name", ""),
                "input": args,
            })

        usage = chat.get("usage") or {}
        return {
            "id": new_message_id(),
            "type": "message",
            "role": "assistant",
            "model": chat.get("model"),
            "content": content_blocks or [{"type": "text", "text": ""}],
            "stop_reason": _STOP_MAP.get(finish, "end_turn"),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }


# --------------------------------------------------------------------- response (stream)

class ChatToAnthropicStreamAdapter(StreamAdapter):
    """Translate a stream of OpenAI chat.completion.chunk dicts into the
    Anthropic message-streaming event sequence.

    State machine produces, in order:
      message_start
      content_block_start (index 0, type=text)
      content_block_delta * N
      content_block_stop  (index 0)
      message_delta       (carries final stop_reason + usage)
      message_stop
    """
    name = "chat_to_anthropic_stream"

    async def transform(
        self, chunks: AsyncIterator[Dict[str, Any]], ctx: RequestContext
    ) -> AsyncIterator[Dict[str, Any]]:
        msg_id = new_message_id()
        started = False
        text_block_open = False
        accumulated_finish = None
        usage: Dict[str, Any] = {}

        async for chunk in chunks:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            finish = choices[0].get("finish_reason")

            if not started:
                started = True
                yield {
                    "type": "message_start",
                    "message": {
                        "id": msg_id, "type": "message", "role": "assistant",
                        "model": chunk.get("model"),
                        "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }

            content_delta = delta.get("content")
            if content_delta:
                if not text_block_open:
                    text_block_open = True
                    yield {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                yield {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content_delta},
                }

            if finish:
                accumulated_finish = finish
                if chunk.get("usage"):
                    usage = chunk["usage"]

        if text_block_open:
            yield {"type": "content_block_stop", "index": 0}
        yield {
            "type": "message_delta",
            "delta": {
                "stop_reason": _STOP_MAP.get(accumulated_finish, "end_turn"),
                "stop_sequence": None,
            },
            "usage": {
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }
        yield {"type": "message_stop"}


# ===========================================================================
# Reverse direction: OpenAI Chat Completions  ->  Anthropic Messages
# Used by AnthropicBackend / VertexClaudeBackend so the proxy can speak chat
# externally while the upstream wants Anthropic shape.
# ===========================================================================

# Anthropic stop_reason -> OpenAI finish_reason
_REVERSE_STOP_MAP = {v: k for k, v in _STOP_MAP.items()}


def chat_to_anthropic_request(oai: Dict[str, Any], *, stream: bool,
                              default_max_tokens: int = 4096) -> Dict[str, Any]:
    """Translate an OpenAI Chat Completions request into an Anthropic Messages
    request body.

    * Collapses any leading ``role: system`` messages into the top-level
      ``system`` string (Anthropic doesn't have a system role on the messages
      list).
    * Translates ``role: tool`` messages into ``user`` messages carrying a
      ``tool_result`` content block.
    * Translates ``assistant.tool_calls`` into ``tool_use`` blocks.
    * Translates ``tools[].function`` (OpenAI) into ``tools[]`` (Anthropic).
    * Anthropic *requires* ``max_tokens``; we fall back to
      ``default_max_tokens`` if the caller omits it.
    """
    sys_parts: List[str] = []
    msgs: List[Dict[str, Any]] = []

    for m in oai.get("messages") or []:
        role = m.get("role")
        if role == "system":
            content = m.get("content")
            if isinstance(content, str) and content:
                sys_parts.append(content)
            elif isinstance(content, list):
                txt = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
                if txt:
                    sys_parts.append(txt)
            continue

        if role == "tool":
            msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", "call_0"),
                    "content": _stringify(m.get("content", "")),
                }],
            })
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            content = m.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                        blocks.append({"type": "text", "text": p.get("text", "")})

            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        import orjson
                        args = orjson.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "call_0"),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if blocks:
                msgs.append({"role": "assistant", "content": blocks})
            continue

        # role: user (or anything else) — keep string content as-is, flatten
        # OpenAI vision-style typed parts to Anthropic text blocks.
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                    blocks.append({"type": "text", "text": p.get("text", "")})
            if blocks:
                msgs.append({"role": "user", "content": blocks})

    out: Dict[str, Any] = {
        "model": oai["model"],
        "messages": msgs,
        "stream": stream,
        "max_tokens": int(oai.get("max_tokens") or default_max_tokens),
    }
    if sys_parts:
        out["system"] = "\n\n".join(sys_parts)
    if "temperature" in oai:
        out["temperature"] = oai["temperature"]
    if "top_p" in oai:
        out["top_p"] = oai["top_p"]
    if "stop" in oai and oai["stop"]:
        s = oai["stop"]
        out["stop_sequences"] = s if isinstance(s, list) else [s]

    if oai.get("tools"):
        anth_tools = []
        for t in oai["tools"]:
            if not isinstance(t, dict) or t.get("type") != "function":
                continue
            fn = t.get("function") or {}
            anth_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {},
            })
        if anth_tools:
            out["tools"] = anth_tools
    if oai.get("tool_choice"):
        tc = oai["tool_choice"]
        if isinstance(tc, str) and tc in ("auto", "none"):
            out["tool_choice"] = {"type": "auto" if tc == "auto" else "none"}
        elif isinstance(tc, dict):
            fn = tc.get("function") or {}
            if fn.get("name"):
                out["tool_choice"] = {"type": "tool", "name": fn["name"]}

    return out


def anthropic_to_chat_response(
    anth: Dict[str, Any], *, model: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate an Anthropic Messages (non-streaming) response into an OpenAI
    Chat Completions response."""
    import time as _time

    from ..utils.ids import new_chatcmpl_id

    text_pieces: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for i, block in enumerate(anth.get("content") or []):
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            text_pieces.append(block.get("text", ""))
        elif bt == "tool_use":
            import orjson
            tool_calls.append({
                "id": block.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": orjson.dumps(block.get("input") or {}).decode(),
                },
            })

    msg: Dict[str, Any] = {"role": "assistant", "content": "".join(text_pieces)}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    finish = _REVERSE_STOP_MAP.get(anth.get("stop_reason"), "stop")
    if tool_calls and finish == "stop":
        finish = "tool_calls"

    usage_in = anth.get("usage") or {}
    return {
        "id": new_chatcmpl_id(),
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": anth.get("model") or model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {
            "prompt_tokens":     usage_in.get("input_tokens", 0),
            "completion_tokens": usage_in.get("output_tokens", 0),
            "total_tokens":      (usage_in.get("input_tokens", 0)
                                  + usage_in.get("output_tokens", 0)),
        },
    }


def anthropic_event_to_chat_chunks(
    event: Dict[str, Any], state: Dict[str, Any], *,
    chatcmpl_id: str, created: int, model: Optional[str],
) -> List[Dict[str, Any]]:
    """Translate one Anthropic streaming event into 0..N OpenAI
    chat.completion.chunk dicts. ``state`` is mutated; callers should pass the
    same dict for every event in a session."""
    base = {"id": chatcmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": model}
    out: List[Dict[str, Any]] = []
    et = event.get("type")

    if et == "message_start":
        # Emit role chunk first.
        m = event.get("message") or {}
        if not state.get("role_emitted"):
            state["role_emitted"] = True
            out.append({**base, "model": m.get("model") or model,
                         "choices": [{"index": 0, "delta": {"role": "assistant"},
                                      "finish_reason": None}]})
            state["input_tokens"] = (m.get("usage") or {}).get("input_tokens", 0)

    elif et == "content_block_start":
        block = event.get("content_block") or {}
        if block.get("type") == "tool_use":
            idx = event.get("index", 0)
            state.setdefault("tool_calls", {})[idx] = {
                "id": block.get("id", f"call_{idx}"),
                "name": block.get("name", ""),
                "args": "",
                "emitted_open": False,
            }

    elif et == "content_block_delta":
        delta = event.get("delta") or {}
        idx = event.get("index", 0)
        dt = delta.get("type")
        if dt == "text_delta":
            text = delta.get("text", "")
            if text:
                out.append({**base,
                             "choices": [{"index": 0,
                                          "delta": {"content": text},
                                          "finish_reason": None}]})
        elif dt == "input_json_delta":
            tc = state.get("tool_calls", {}).get(idx)
            if tc is None:
                return out
            partial = delta.get("partial_json") or ""
            tc_chunk: Dict[str, Any] = {
                "index": idx,
                "type": "function",
                "function": {"arguments": partial},
            }
            if not tc["emitted_open"]:
                tc["emitted_open"] = True
                tc_chunk["id"] = tc["id"]
                tc_chunk["function"]["name"] = tc["name"]
            out.append({**base,
                         "choices": [{"index": 0,
                                      "delta": {"tool_calls": [tc_chunk]},
                                      "finish_reason": None}]})

    elif et == "message_delta":
        md = event.get("delta") or {}
        if md.get("stop_reason"):
            state["stop_reason"] = md["stop_reason"]
        u = event.get("usage") or {}
        if "output_tokens" in u:
            state["output_tokens"] = u["output_tokens"]

    elif et == "message_stop":
        finish = _REVERSE_STOP_MAP.get(state.get("stop_reason"), "stop")
        if state.get("tool_calls") and finish == "stop":
            finish = "tool_calls"
        out.append({
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {
                "prompt_tokens":     state.get("input_tokens", 0),
                "completion_tokens": state.get("output_tokens", 0),
                "total_tokens": (state.get("input_tokens", 0)
                                 + state.get("output_tokens", 0)),
            },
        })

    return out
