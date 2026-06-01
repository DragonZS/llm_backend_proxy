"""Native Ollama backend: speaks /api/chat + /api/tags, exposes the same
``Backend`` interface as OpenAI-compat backends.

The proxy's API layer always sends OpenAI-shape requests/streams. This class:

  * Translates POST /v1/chat/completions → POST /api/chat (NDJSON streaming).
  * Translates GET /v1/models → GET /api/tags.
  * Re-emits SSE chunks shaped like ``chat.completion.chunk``.
  * Re-emits a non-streaming response shaped like ``chat.completion``.

That way the rest of the proxy (adapters, agent profiles, /v1/responses,
/v1/messages) just works: the Ollama backend looks identical to a vLLM
backend from the outside.

Tools (function calling) follow the OpenAI shape Ollama 0.3+ already accepts
on /api/chat. Older Ollama versions reject ``tools`` with 4xx, surfaced as an
``UpstreamError`` to the client.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import orjson

from ..config.schema import BackendConfig
from ..core.errors import UpstreamError
from ..utils.ids import new_chatcmpl_id
from .base import PassthroughResponse
from .factory import register_backend
from .openai_compat import OpenAICompatBackend


class OllamaNativeBackend(OpenAICompatBackend):
    """Ollama backend translating to/from /api/chat and /api/tags."""

    DEFAULT_HEALTH_PATH = "/"

    # ---- discovery / health -----------------------------------------
    async def list_models(self) -> List[Dict[str, Any]]:
        r = await self._client.get("/api/tags")
        if r.status_code != 200:
            raise UpstreamError(
                f"backend {self.name} /api/tags returned {r.status_code}",
                extras={"backend": self.name},
            )
        body = r.json()
        out: List[Dict[str, Any]] = []
        for m in body.get("models") or []:
            if not isinstance(m, dict):
                continue
            mid = m.get("name") or m.get("model")
            if not mid:
                continue
            out.append({
                "id": mid,
                "object": "model",
                "owned_by": "ollama",
                "_backend": self.name,
                # Surface useful Ollama metadata as extras
                "_size":     m.get("size"),
                "_digest":   m.get("digest"),
                "_modified": m.get("modified_at"),
            })
        return out

    # ---- routing override -------------------------------------------
    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return await self._chat_non_streaming(json, headers)
        # Fallback: anything else (rare) goes through unchanged.
        return await super().request_json(method, path, json=json, headers=headers)

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return self._chat_streaming(json, headers)
        return await super().stream(method, path, json=json, headers=headers)

    async def passthrough(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> PassthroughResponse:
        # The catch-all router uses passthrough for /v1/chat/completions when
        # there's no explicit handler. Rewire that to our translator so a plain
        # POST to /v1/chat/completions still reaches an Ollama backend.
        if method == "POST" and path == "/v1/chat/completions" and body is not None:
            try:
                payload = orjson.loads(body)
            except orjson.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                streaming = bool(payload.get("stream"))
                if streaming:
                    # Build an SSE byte stream from the translated NDJSON.
                    chunks: list[bytes] = []
                    async for piece in self._chat_streaming(payload, headers):
                        chunks.append(piece)
                    return PassthroughResponse(
                        200,
                        {"content-type": "text/event-stream"},
                        b"".join(chunks),
                    )
                full = await self._chat_non_streaming(payload, headers)
                return PassthroughResponse(
                    200,
                    {"content-type": "application/json"},
                    orjson.dumps(full),
                )
        return await super().passthrough(method, path, body=body, headers=headers)

    # ---- translation core --------------------------------------------
    async def _chat_non_streaming(
        self, oai_req: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        ollama_req = chat_to_ollama_request(oai_req, stream=False)
        h = self._merge_headers(headers)
        h.setdefault("content-type", "application/json")
        r = await self._client.post(
            "/api/chat", content=orjson.dumps(ollama_req), headers=h,
        )
        if r.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.name} /api/chat returned {r.status_code}: {r.text[:300]}",
                status_code=r.status_code,
                extras={"backend": self.name},
            )
        return ollama_to_chat_response(r.json(), model=oai_req.get("model"))

    def _chat_streaming(
        self, oai_req: Dict[str, Any], headers: Optional[Dict[str, str]]
    ) -> AsyncIterator[bytes]:
        """Return an async iterator of SSE bytes (matching what the API layer
        expects from any backend.stream(...))."""
        ollama_req = chat_to_ollama_request(oai_req, stream=True)
        model = oai_req.get("model") or ollama_req.get("model")
        h = self._merge_headers(headers)
        h.setdefault("content-type", "application/json")

        client = self._client

        async def _gen() -> AsyncIterator[bytes]:
            chatcmpl_id = new_chatcmpl_id()
            created = int(time.time())
            async with client.stream(
                "POST", "/api/chat",
                content=orjson.dumps(ollama_req), headers=h,
            ) as r:
                if r.status_code >= 400:
                    err = (await r.aread()).decode("utf-8", "replace")
                    raise UpstreamError(
                        f"upstream {self.name} /api/chat returned {r.status_code}: {err[:300]}",
                        status_code=r.status_code,
                        extras={"backend": self.name},
                    )
                # Emit role chunk first so downstream chat_to_responses
                # adapters can detect the assistant turn opening.
                yield _sse({
                    "id": chatcmpl_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0,
                                 "delta": {"role": "assistant"},
                                 "finish_reason": None}],
                })
                buf = b""
                async for raw in r.aiter_raw():
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = orjson.loads(line)
                        except orjson.JSONDecodeError:
                            continue
                        for chunk in ollama_chunk_to_chat_chunks(
                            obj, chatcmpl_id=chatcmpl_id,
                            created=created, model=model,
                        ):
                            yield _sse(chunk)
                # tail: emit DONE so the API layer's SSE serialiser is satisfied
                yield b"data: [DONE]\n\n"

        return _gen()


# ---------------------------------------------------------------------------
# Translation helpers — pure functions, easy to unit-test
# ---------------------------------------------------------------------------

def chat_to_ollama_request(oai: Dict[str, Any], *, stream: bool) -> Dict[str, Any]:
    """OpenAI Chat Completions request → Ollama /api/chat request body."""
    out: Dict[str, Any] = {
        "model": oai["model"],
        "messages": list(oai.get("messages") or []),
        "stream": stream,
    }
    options: Dict[str, Any] = {}
    # OpenAI: max_tokens   -> Ollama: options.num_predict
    if oai.get("max_tokens") is not None:
        options["num_predict"] = oai["max_tokens"]
    if oai.get("temperature") is not None:
        options["temperature"] = oai["temperature"]
    if oai.get("top_p") is not None:
        options["top_p"] = oai["top_p"]
    if oai.get("stop") is not None:
        options["stop"] = oai["stop"]
    if options:
        out["options"] = options
    # Ollama 0.3+ accepts OpenAI-shape tools on /api/chat
    if oai.get("tools"):
        out["tools"] = oai["tools"]
    if oai.get("tool_choice"):
        out["tool_choice"] = oai["tool_choice"]
    if oai.get("response_format"):
        # Ollama maps response_format.type=json_object -> format: "json"
        rf = oai["response_format"]
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            out["format"] = "json"
    return out


def ollama_to_chat_response(
    ollama: Dict[str, Any], *, model: Optional[str]
) -> Dict[str, Any]:
    """Non-streaming /api/chat response → OpenAI Chat Completions response."""
    msg_in = ollama.get("message") or {}
    msg_out: Dict[str, Any] = {
        "role": msg_in.get("role", "assistant"),
        "content": msg_in.get("content", ""),
    }
    if msg_in.get("tool_calls"):
        # Ollama returns tool_calls as [{function:{name, arguments:dict|str}}, ...]
        tc_out: List[Dict[str, Any]] = []
        for i, tc in enumerate(msg_in["tool_calls"]):
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, (dict, list)):
                args = orjson.dumps(args).decode()
            tc_out.append({
                "id": tc.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args or ""},
            })
        msg_out["tool_calls"] = tc_out
        msg_out["content"] = msg_out.get("content") or ""

    finish = "stop"
    if msg_out.get("tool_calls"):
        finish = "tool_calls"
    elif ollama.get("done_reason") == "length":
        finish = "length"

    usage = {
        "prompt_tokens":     ollama.get("prompt_eval_count", 0),
        "completion_tokens": ollama.get("eval_count", 0),
        "total_tokens": (
            (ollama.get("prompt_eval_count", 0) or 0)
            + (ollama.get("eval_count", 0) or 0)
        ),
    }
    return {
        "id": new_chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ollama.get("model") or model,
        "choices": [{"index": 0, "message": msg_out, "finish_reason": finish}],
        "usage": usage,
    }


def ollama_chunk_to_chat_chunks(
    ollama: Dict[str, Any], *, chatcmpl_id: str, created: int, model: Optional[str],
) -> List[Dict[str, Any]]:
    """One Ollama NDJSON line → 0..2 chat.completion.chunk dicts."""
    base = {"id": chatcmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": ollama.get("model") or model}
    out: List[Dict[str, Any]] = []
    msg = ollama.get("message") or {}
    delta: Dict[str, Any] = {}
    content = msg.get("content")
    if content:
        delta["content"] = content
    # tool_calls only arrive on the final ``done: true`` chunk in Ollama;
    # forward them directly (the OpenAI adapter side already understands them)
    if msg.get("tool_calls"):
        delta["tool_calls"] = []
        for i, tc in enumerate(msg["tool_calls"]):
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, (dict, list)):
                args = orjson.dumps(args).decode()
            delta["tool_calls"].append({
                "index": i,
                "id": tc.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args or ""},
            })

    finish = None
    if ollama.get("done"):
        finish = "stop"
        if msg.get("tool_calls"):
            finish = "tool_calls"
        elif ollama.get("done_reason") == "length":
            finish = "length"

    if delta or finish:
        c = {**base, "choices": [{"index": 0, "delta": delta or {},
                                   "finish_reason": finish}]}
        if finish:
            c["usage"] = {
                "prompt_tokens":     ollama.get("prompt_eval_count", 0),
                "completion_tokens": ollama.get("eval_count", 0),
                "total_tokens": (
                    (ollama.get("prompt_eval_count", 0) or 0)
                    + (ollama.get("eval_count", 0) or 0)
                ),
            }
        out.append(c)
    return out


def _sse(obj: Dict[str, Any]) -> bytes:
    return b"data: " + orjson.dumps(obj) + b"\n\n"


# Register with the factory
register_backend("ollama", OllamaNativeBackend)
