"""Anthropic backend.

Speaks Anthropic Messages API natively (``/v1/messages`` with
``x-api-key`` + ``anthropic-version`` headers). Exposes the standard
``Backend`` interface so the rest of the proxy stays uniform:

* ``POST /v1/messages``         → forwarded as-is (passthrough).
* ``POST /v1/chat/completions`` → translated to ``/v1/messages`` on the
  way out, response translated back to chat shape on the way in.
* ``GET /v1/models``           → returns the entries declared in
  ``BackendConfig.models``  (Anthropic's ``/v1/models`` endpoint exists in
  newer API versions but is not required for our flow).
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import orjson

from ..adapters.anthropic import (
    anthropic_event_to_chat_chunks,
    anthropic_to_chat_response,
    chat_to_anthropic_request,
)
from ..config.schema import BackendConfig
from ..core.errors import UpstreamError
from ..streaming import parse_sse_lines
from ..utils.ids import new_chatcmpl_id
from .base import PassthroughResponse
from .factory import register_backend
from .openai_compat import OpenAICompatBackend

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(OpenAICompatBackend):
    """Backend speaking the Anthropic Messages API."""

    DEFAULT_HEALTH_PATH = "/v1/models"  # works on api.anthropic.com / Vertex
    models_path = "/v1/models"

    # Subclasses (VertexClaudeBackend) override this to relocate to a
    # cloud-specific URL.
    messages_path: str = "/v1/messages"

    def __init__(self, cfg: BackendConfig) -> None:
        super().__init__(cfg)
        # Ensure outbound headers carry an anthropic-version. Allow override
        # via cfg.headers if the user wants a specific date.
        self._anth_version = (
            (cfg.headers or {}).get("anthropic-version")
            or _DEFAULT_ANTHROPIC_VERSION
        )

    # ---- helpers ----------------------------------------------------
    def _anthropic_headers(self, base: Dict[str, str]) -> Dict[str, str]:
        h = dict(base)
        h.setdefault("anthropic-version", self._anth_version)
        return h

    # ---- discovery / health -----------------------------------------
    async def list_models(self) -> List[Dict[str, Any]]:
        # Try the real /v1/models first; fall back to the configured list.
        try:
            r = await self._client.get(self.models_path)
            if r.status_code == 200:
                body = r.json()
                if isinstance(body, dict) and isinstance(body.get("data"), list):
                    return [
                        {**m, "_backend": self.name}
                        for m in body["data"] if isinstance(m, dict)
                    ]
        except httpx.HTTPError:
            pass
        return [
            {"id": m, "object": "model", "owned_by": "anthropic",
             "_backend": self.name}
            for m in (self.cfg.models or [])
        ]

    async def health(self) -> bool:
        try:
            r = await self._client.get(self.health_path)
            # 200 = OK; 401 still proves the endpoint is alive (auth issue).
            return r.status_code in (200, 401)
        except httpx.HTTPError:
            return False

    # ---- routing overrides ------------------------------------------
    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if method == "POST" and path == "/v1/messages" and json is not None:
            return await self._messages_passthrough(json, headers, stream=False)
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return await self._chat_via_messages(json, headers, stream=False)
        return await super().request_json(method, path, json=json, headers=headers)

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        if method == "POST" and path == "/v1/messages" and json is not None:
            return self._messages_stream(json, headers)
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return self._chat_via_messages_stream(json, headers)
        return await super().stream(method, path, json=json, headers=headers)

    async def passthrough(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> PassthroughResponse:
        if method == "POST" and body is not None and path in (
            "/v1/messages", "/v1/chat/completions",
        ):
            try:
                payload = orjson.loads(body)
            except orjson.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                streaming = bool(payload.get("stream"))
                if path == "/v1/messages":
                    if streaming:
                        chunks: list[bytes] = []
                        async for c in self._messages_stream(payload, headers):
                            chunks.append(c)
                        return PassthroughResponse(
                            200, {"content-type": "text/event-stream"},
                            b"".join(chunks),
                        )
                    full = await self._messages_passthrough(payload, headers, stream=False)
                    return PassthroughResponse(
                        200, {"content-type": "application/json"},
                        orjson.dumps(full),
                    )
                # path == /v1/chat/completions
                if streaming:
                    chunks2: list[bytes] = []
                    async for c in self._chat_via_messages_stream(payload, headers):
                        chunks2.append(c)
                    return PassthroughResponse(
                        200, {"content-type": "text/event-stream"},
                        b"".join(chunks2),
                    )
                full = await self._chat_via_messages(payload, headers, stream=False)
                return PassthroughResponse(
                    200, {"content-type": "application/json"},
                    orjson.dumps(full),
                )
        return await super().passthrough(method, path, body=body, headers=headers)

    # ---- core: Anthropic /v1/messages call ---------------------------
    async def _messages_passthrough(
        self, anth_req: Dict[str, Any],
        headers: Optional[Dict[str, str]], *, stream: bool,
    ) -> Dict[str, Any]:
        """Send an Anthropic-shaped body to ``messages_path``. Returns parsed
        JSON for non-streaming; for streaming use ``_messages_stream``."""
        anth_req = dict(anth_req)
        anth_req["stream"] = stream
        h = self._anthropic_headers(self._merge_headers(headers))
        h.setdefault("content-type", "application/json")
        r = await self._client.post(
            self.messages_path,
            content=orjson.dumps(anth_req), headers=h,
        )
        if r.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.name} {self.messages_path} returned {r.status_code}: {r.text[:300]}",
                status_code=r.status_code,
                extras={"backend": self.name},
            )
        return r.json()

    def _messages_stream(
        self, anth_req: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> AsyncIterator[bytes]:
        """Pure passthrough for /v1/messages streaming — returns the upstream
        SSE bytes verbatim."""
        anth_req = dict(anth_req)
        anth_req["stream"] = True
        h = self._anthropic_headers(self._merge_headers(headers))
        h.setdefault("content-type", "application/json")
        h.setdefault("accept", "text/event-stream")
        client = self._client
        path = self.messages_path

        async def _gen() -> AsyncIterator[bytes]:
            async with client.stream(
                "POST", path, content=orjson.dumps(anth_req), headers=h,
            ) as r:
                if r.status_code >= 400:
                    err = (await r.aread()).decode("utf-8", "replace")
                    raise UpstreamError(
                        f"upstream {self.name} stream returned {r.status_code}: {err[:300]}",
                        status_code=r.status_code,
                        extras={"backend": self.name},
                    )
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

        return _gen()

    # ---- core: chat-completions adapter --------------------------------
    async def _chat_via_messages(
        self, oai_req: Dict[str, Any],
        headers: Optional[Dict[str, str]], *, stream: bool,
    ) -> Dict[str, Any]:
        anth_req = chat_to_anthropic_request(oai_req, stream=False)
        anth_resp = await self._messages_passthrough(anth_req, headers, stream=False)
        return anthropic_to_chat_response(anth_resp, model=oai_req.get("model"))

    def _chat_via_messages_stream(
        self, oai_req: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> AsyncIterator[bytes]:
        """Translate chat-stream → Anthropic-stream → chat-stream. Returns SSE
        bytes ready for the client."""
        anth_req = chat_to_anthropic_request(oai_req, stream=True)
        upstream = self._messages_stream(anth_req, headers)
        chatcmpl_id = new_chatcmpl_id()
        created = int(time.time())
        model = oai_req.get("model")

        async def _gen() -> AsyncIterator[bytes]:
            state: Dict[str, Any] = {}
            async for ev in parse_sse_lines(upstream):
                for chunk in anthropic_event_to_chat_chunks(
                    ev, state, chatcmpl_id=chatcmpl_id,
                    created=created, model=model,
                ):
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _gen()


register_backend("anthropic", AnthropicBackend)
