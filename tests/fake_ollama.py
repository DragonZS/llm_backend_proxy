"""A tiny fake Ollama server used by R3 tests. Implements just /api/tags and
/api/chat (NDJSON streaming + non-streaming)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def make_fake_ollama(
    *,
    name: str = "ollama",
    models: Optional[List[Dict[str, Any]]] = None,
    chat_response: Optional[Dict[str, Any]] = None,
    stream_chunks: Optional[List[Dict[str, Any]]] = None,
    tool_call_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> FastAPI:
    app = FastAPI(title=f"fake-ollama-{name}")
    models = models or [{"name": "llama3.1:8b", "size": 1, "digest": "x",
                          "modified_at": "2024-01-01T00:00:00Z"}]

    @app.get("/")
    async def _root() -> str:
        return "Ollama is running"

    @app.get("/api/tags")
    async def _tags() -> dict:
        return {"models": models}

    @app.post("/api/chat")
    async def _chat(req: Request):
        body = await req.json()
        if body.get("stream"):
            chunks = stream_chunks or _default_stream_chunks(body.get("model") or "x")

            async def gen():
                for c in chunks:
                    yield (json.dumps(c) + "\n").encode()
                    await asyncio.sleep(0)

            return StreamingResponse(gen(), media_type="application/x-ndjson")

        if tool_call_handler is not None and body.get("tools"):
            return JSONResponse(tool_call_handler(body))
        return JSONResponse(chat_response or _default_chat_response(body.get("model") or "x"))

    return app


def _default_chat_response(model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "hello from ollama"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 4,
    }


def _default_stream_chunks(model: str) -> List[Dict[str, Any]]:
    return [
        {"model": model, "created_at": "t",
         "message": {"role": "assistant", "content": "hi"},
         "done": False},
        {"model": model, "created_at": "t",
         "message": {"role": "assistant", "content": "!"},
         "done": False},
        {"model": model, "created_at": "t",
         "message": {"role": "assistant", "content": ""},
         "done": True, "done_reason": "stop",
         "prompt_eval_count": 1, "eval_count": 2},
    ]
