"""Fake Vertex Gemini upstream."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_fake_vertex_gemini(
    *, project: str = "test-project", location: str = "us-central1",
    require_bearer: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="fake-vertex-gemini")
    base = f"/v1/projects/{project}/locations/{location}/publishers/google/models"

    def _check(req: Request):
        if require_bearer is not None:
            if req.headers.get("authorization") != f"Bearer {require_bearer}":
                return JSONResponse(401, content={"error": "bad bearer"})
        return None

    @app.post(base + "/{model}:generateContent")
    async def _generate(model: str, req: Request):
        if (e := _check(req)) is not None:
            return e
        body = await req.json()
        return JSONResponse(_default_response(model, body))

    @app.post(base + "/{model}:streamGenerateContent")
    async def _stream(model: str, req: Request):
        if (e := _check(req)) is not None:
            return e
        body = await req.json()
        chunks = _default_stream(model)

        async def gen():
            for c in chunks:
                yield ("data: " + json.dumps(c) + "\n\n").encode()
                await asyncio.sleep(0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _default_response(model: str, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidates": [{
            "content": {"role": "model",
                         "parts": [{"text": "hello from gemini"}]},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 4,
                            "candidatesTokenCount": 3,
                            "totalTokenCount": 7},
        "modelVersion": model,
    }


def _default_stream(model: str) -> List[Dict[str, Any]]:
    return [
        {"candidates": [{"content": {"role": "model",
                                       "parts": [{"text": "hi"}]}}]},
        {"candidates": [{"content": {"role": "model",
                                       "parts": [{"text": "!"}]}}]},
        {"candidates": [{"content": {"role": "model", "parts": []},
                          "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2,
                            "totalTokenCount": 3}},
    ]
