"""A tiny fake vLLM upstream — used by integration tests to replace a real
vLLM server. Only implements the endpoints the proxy talks to in the milestone
under test.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_fake_vllm(
    *,
    name: str = "fake",
    models: List[str] | None = None,
    chat_handler: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
    stream_chunks: List[Dict[str, Any]] | None = None,
    require_bearer: str | None = None,
    require_header: tuple[str, str] | None = None,
) -> FastAPI:
    """Build a FastAPI app that mimics the parts of vLLM the proxy relies on.

    ``require_bearer``: if set, every request must carry
        ``Authorization: Bearer <require_bearer>``; otherwise 401.
    ``require_header``: ``(name, value)`` pair the request must carry; otherwise 401.
    """
    app = FastAPI(title=f"fake-vllm-{name}")
    models = models or [f"{name}-model"]

    def _check_auth(req: Request) -> JSONResponse | None:
        if require_bearer is not None:
            auth = req.headers.get("authorization") or ""
            if auth != f"Bearer {require_bearer}":
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "bad bearer", "got": auth}},
                )
        if require_header is not None:
            name, val = require_header
            if req.headers.get(name) != val:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": f"bad {name}",
                                       "got": req.headers.get(name)}},
                )
        return None

    @app.get("/health")
    async def _health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def _models(req: Request):
        if (err := _check_auth(req)) is not None:
            return err
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": name} for m in models],
        }

    @app.post("/v1/chat/completions")
    async def _chat(req: Request) -> Any:
        if (err := _check_auth(req)) is not None:
            return err
        body = await req.json()
        if body.get("stream"):
            chunks = stream_chunks or _default_stream_chunks(body.get("model") or models[0])

            async def gen():
                for c in chunks:
                    yield f"data: {json.dumps(c)}\n\n".encode()
                    await asyncio.sleep(0)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        if chat_handler:
            return JSONResponse(chat_handler(body))
        return JSONResponse(_default_chat_response(body.get("model") or models[0]))

    return app


def _default_chat_response(model: str) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": model,
        "created": 0,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello from fake"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
    }


def _default_stream_chunks(model: str) -> List[Dict[str, Any]]:
    base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
            "created": 0, "model": model}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "hi"},      "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "!"},       "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {},                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    ]
