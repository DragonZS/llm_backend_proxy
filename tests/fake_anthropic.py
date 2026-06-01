"""Fake Anthropic Messages API server for tests."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_fake_anthropic(
    *,
    name: str = "anthropic",
    models: Optional[List[str]] = None,
    require_api_key: Optional[str] = None,
    messages_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> FastAPI:
    app = FastAPI(title=f"fake-anthropic-{name}")
    models = models or ["claude-3-5-sonnet"]

    def _check(req: Request) -> Optional[JSONResponse]:
        if require_api_key is not None:
            if req.headers.get("x-api-key") != require_api_key:
                return JSONResponse(401, content={"error": {"message": "no key"}})
        return None

    @app.get("/v1/models")
    async def _models(req: Request):
        if (e := _check(req)) is not None:
            return e
        return {"data": [{"id": m, "object": "model",
                          "owned_by": "anthropic"} for m in models]}

    @app.post("/v1/messages")
    async def _messages(req: Request):
        if (e := _check(req)) is not None:
            return e
        body = await req.json()
        if body.get("stream"):
            chunks = _default_anth_stream(body.get("model") or models[0])

            async def gen():
                for ev_type, payload in chunks:
                    yield f"event: {ev_type}\ndata: {json.dumps(payload)}\n\n".encode()
                    await asyncio.sleep(0)

            return StreamingResponse(gen(), media_type="text/event-stream")

        if messages_handler is not None:
            return JSONResponse(messages_handler(body))
        return JSONResponse(_default_anth_response(body.get("model") or models[0]))

    return app


def _default_anth_response(model: str) -> Dict[str, Any]:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello from anthropic"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _default_anth_stream(model: str) -> List[tuple[str, Dict[str, Any]]]:
    return [
        ("message_start", {
            "type": "message_start",
            "message": {"id": "msg_fake", "type": "message", "role": "assistant",
                         "model": model, "content": [],
                         "stop_reason": None, "stop_sequence": None,
                         "usage": {"input_tokens": 5, "output_tokens": 0}},
        }),
        ("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }),
        ("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "!"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
