"""Adapters bridging the Google Gemini API to OpenAI Chat Completions.

Gemini request shape we care about:

    POST .../models/{model}:generateContent
    {
      "contents": [
        {"role": "user",  "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "..."},
                                     {"functionCall": {"name": "...",
                                                       "args": {...}}}]}
      ],
      "systemInstruction": {"parts": [{"text": "..."}]},
      "tools": [{"functionDeclarations": [...]}],
      "toolConfig": {"functionCallingConfig": {"mode": "AUTO"|"ANY"|"NONE",
                                                 "allowedFunctionNames": [...]}},
      "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1024,
                            "topP": 0.9, "stopSequences": [...]}
    }

Gemini non-streaming response:

    {
      "candidates": [{
        "content": {"role": "model", "parts": [{"text": "..."},
                                                {"functionCall": {...}}]},
        "finishReason": "STOP" | "MAX_TOKENS" | "SAFETY" | ...,
        "index": 0
      }],
      "usageMetadata": {"promptTokenCount": ..., "candidatesTokenCount": ...,
                         "totalTokenCount": ...}
    }

Streaming returns the same shape per SSE line; concatenated parts form the
full message.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import orjson

from ..utils.ids import new_chatcmpl_id


# Gemini finishReason -> OpenAI finish_reason
_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}


# ---- request rewrite -------------------------------------------------

def chat_to_gemini_request(oai: Dict[str, Any]) -> Dict[str, Any]:
    sys_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for m in oai.get("messages") or []:
        role = m.get("role")
        content = m.get("content")

        if role == "system":
            if isinstance(content, str) and content:
                sys_parts.append(content)
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                        sys_parts.append(p.get("text", ""))
            continue

        if role == "tool":
            # Gemini represents tool results as functionResponse parts attached
            # to a user-role content block.
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": m.get("name") or m.get("tool_call_id", "tool"),
                        "response": _maybe_json(m.get("content", "")),
                    }
                }],
            })
            continue

        if role == "assistant":
            parts: List[Dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"text": content})
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") in ("text", "output_text"):
                        parts.append({"text": p.get("text", "")})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = orjson.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                parts.append({"functionCall": {
                    "name": fn.get("name", ""), "args": args,
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        # role: user (or anything else)
        if isinstance(content, str):
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts2: List[Dict[str, Any]] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                    parts2.append({"text": p.get("text", "")})
            if parts2:
                contents.append({"role": "user", "parts": parts2})

    out: Dict[str, Any] = {"contents": contents}
    if sys_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}

    gc: Dict[str, Any] = {}
    if "max_tokens" in oai:
        gc["maxOutputTokens"] = oai["max_tokens"]
    if "temperature" in oai:
        gc["temperature"] = oai["temperature"]
    if "top_p" in oai:
        gc["topP"] = oai["top_p"]
    if "stop" in oai and oai["stop"]:
        s = oai["stop"]
        gc["stopSequences"] = s if isinstance(s, list) else [s]
    if gc:
        out["generationConfig"] = gc

    if oai.get("tools"):
        decls = []
        for t in oai["tools"]:
            if not isinstance(t, dict) or t.get("type") != "function":
                continue
            fn = t.get("function") or {}
            decls.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {},
            })
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]

    if oai.get("tool_choice"):
        tc = oai["tool_choice"]
        cfg: Dict[str, Any] = {}
        if isinstance(tc, str) and tc in ("auto", "none"):
            cfg["mode"] = "AUTO" if tc == "auto" else "NONE"
        elif isinstance(tc, dict):
            fn = tc.get("function") or {}
            if fn.get("name"):
                cfg = {"mode": "ANY", "allowedFunctionNames": [fn["name"]]}
        if cfg:
            out["toolConfig"] = {"functionCallingConfig": cfg}

    return out


def _maybe_json(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, str):
        try:
            return orjson.loads(s)
        except Exception:  # noqa: BLE001
            return {"text": s}
    return {"value": s}


# ---- response wrap (full) -------------------------------------------

def gemini_to_chat_response(
    gem: Dict[str, Any], *, model: Optional[str] = None,
) -> Dict[str, Any]:
    candidates = gem.get("candidates") or []
    cand = candidates[0] if candidates else {}
    content = cand.get("content") or {}
    parts = content.get("parts") or []

    text_pieces: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            continue
        if "text" in p:
            text_pieces.append(p.get("text", ""))
        elif "functionCall" in p:
            fc = p["functionCall"] or {}
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": orjson.dumps(fc.get("args") or {}).decode(),
                },
            })

    msg: Dict[str, Any] = {"role": "assistant", "content": "".join(text_pieces)}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    finish = _FINISH_MAP.get(cand.get("finishReason"), "stop")
    if tool_calls and finish == "stop":
        finish = "tool_calls"

    usage = gem.get("usageMetadata") or {}
    return {
        "id": new_chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {
            "prompt_tokens":     usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens":      usage.get("totalTokenCount", 0),
        },
    }


# ---- streaming chunk translation -----------------------------------

def gemini_chunk_to_chat_chunks(
    gem: Dict[str, Any], state: Dict[str, Any], *,
    chatcmpl_id: str, created: int, model: Optional[str],
) -> List[Dict[str, Any]]:
    base = {"id": chatcmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": model}
    out: List[Dict[str, Any]] = []

    if not state.get("role_emitted"):
        state["role_emitted"] = True
        out.append({**base, "choices": [{"index": 0,
                                            "delta": {"role": "assistant"},
                                            "finish_reason": None}]})

    cand = (gem.get("candidates") or [{}])[0]
    content = cand.get("content") or {}
    parts = content.get("parts") or []
    finish_reason = cand.get("finishReason")

    text_buf: List[str] = []
    tool_chunks: List[Dict[str, Any]] = []
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            continue
        if "text" in p:
            text_buf.append(p.get("text", ""))
        elif "functionCall" in p:
            fc = p["functionCall"] or {}
            tc_idx = state.setdefault("_tc_index", 0)
            state["_tc_index"] = tc_idx + 1
            tool_chunks.append({
                "index": tc_idx,
                "id": f"call_{tc_idx}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": orjson.dumps(fc.get("args") or {}).decode(),
                },
            })

    delta: Dict[str, Any] = {}
    if text_buf:
        delta["content"] = "".join(text_buf)
    if tool_chunks:
        delta["tool_calls"] = tool_chunks
        # Mark so the finish chunk uses tool_calls instead of stop.
        state["had_tool_call"] = True

    if delta:
        out.append({**base, "choices": [{"index": 0, "delta": delta,
                                            "finish_reason": None}]})

    usage = gem.get("usageMetadata")
    if usage is not None:
        # Track running totals (Gemini repeats the cumulative usage every chunk).
        state["usage"] = usage

    if finish_reason:
        finish = _FINISH_MAP.get(finish_reason, "stop")
        if state.get("had_tool_call") and finish == "stop":
            finish = "tool_calls"
        chunk = {**base, "choices": [{"index": 0, "delta": {},
                                          "finish_reason": finish}]}
        u = state.get("usage") or {}
        chunk["usage"] = {
            "prompt_tokens":     u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0),
            "total_tokens":      u.get("totalTokenCount", 0),
        }
        out.append(chunk)

    return out
