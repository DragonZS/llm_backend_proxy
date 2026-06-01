"""Fake Vertex AI Anthropic upstream — implements the rawPredict /
streamRawPredict URLs for testing."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_fake_vertex_anthropic(
    *,
    name: str = "vertex-claude",
    project: str = "test-project",
    location: str = "us-central1",
    require_bearer: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title=f"fake-vertex-{name}")
    base = f"/v1/projects/{project}/locations/{location}/publishers/anthropic/models"

    def _check(req: Request):
        if require_bearer is not None:
            if req.headers.get("authorization") != f"Bearer {require_bearer}":
                return JSONResponse(401, content={"error": "bad bearer"})
        return None

    @app.post(base + "/{model}:rawPredict")
    async def _raw(model: str, req: Request):
        if (e := _check(req)) is not None:
            return e
        body = await req.json()
        # Vertex strips 'model' from the body (it's in the URL); echo back.
        return JSONResponse({
            "id": "msg_vertex",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello from vertex-claude"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 4},
        })

    @app.post(base + "/{model}:streamRawPredict")
    async def _stream(model: str, req: Request):
        if (e := _check(req)) is not None:
            return e
        body = await req.json()
        chunks = _default_anth_stream(model)

        async def gen():
            for ev_type, payload in chunks:
                yield f"event: {ev_type}\ndata: {json.dumps(payload)}\n\n".encode()
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _default_anth_stream(model: str) -> List[tuple[str, Dict[str, Any]]]:
    return [
        ("message_start", {"type": "message_start",
                            "message": {"id": "x", "type": "message",
                                        "role": "assistant", "model": model,
                                        "content": [],
                                        "usage": {"input_tokens": 4}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": "!"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
