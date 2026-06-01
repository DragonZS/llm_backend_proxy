"""Vertex AI backends.

Two flavours:

* ``type: vertex-claude``  — Anthropic's Claude served via Vertex AI. Wire
  format is the same as ``api.anthropic.com`` (Messages API), but the URL is
  ``/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict``
  for non-streaming and ``:streamRawPredict`` for streaming. Auth is a Google
  OAuth2 access token in the ``Authorization`` header.

* ``type: vertex-gemini``  — Google's Gemini, native API.

Both share the same auth/access-token machinery.

Required ``base_url``: usually ``https://{region}-aiplatform.googleapis.com``.
Required ``options`` (under ``auth.options`` in YAML or directly under the
backend block):

  project:           "my-gcp-project"
  location:          "us-central1"
  credentials_file:  "/etc/sa.json"          # OR
  credentials_inline: { ... }                # OR (omit both for ADC)
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import orjson

from ..adapters.gemini import (
    chat_to_gemini_request,
    gemini_chunk_to_chat_chunks,
    gemini_to_chat_response,
)
from ..config.schema import BackendConfig
from ..core.errors import BadRequestError, UpstreamError
from ..streaming import parse_sse_lines
from ..utils.ids import new_chatcmpl_id
from .anthropic import AnthropicBackend
from .base import PassthroughResponse
from .factory import register_backend
from .openai_compat import OpenAICompatBackend
from ._gcp_oauth import GcpAccessTokenProvider


# ---------------------------------------------------------------------------
# Shared mixin: Google OAuth + Vertex URL builder
# ---------------------------------------------------------------------------

class _VertexMixin:
    """Adds Google OAuth + project/location config. Subclasses use:

      self._project   – str
      self._location  – str
      self._token_provider – GcpAccessTokenProvider
    """

    cfg: BackendConfig

    def _init_vertex(self, cfg: BackendConfig) -> None:
        opts = (cfg.auth.options if cfg.auth else {}) or {}
        # Backwards-compat: also honour cfg.headers.{project,location}
        self._project: Optional[str] = (
            opts.get("project") or (cfg.headers or {}).get("x-vertex-project")
        )
        self._location: Optional[str] = (
            opts.get("location") or (cfg.headers or {}).get("x-vertex-location") or "us-central1"
        )
        if not self._project:
            raise BadRequestError(
                f"backend {cfg.name}: vertex backends require auth.options.project"
            )
        self._token_provider = GcpAccessTokenProvider({
            "credentials_file":   opts.get("credentials_file"),
            "credentials_inline": opts.get("credentials_inline"),
            "scope":              opts.get("scope"),
        })

    async def _vertex_auth_header(self) -> Dict[str, str]:
        token = await self._token_provider.get()
        return {"authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Vertex × Anthropic Claude
# ---------------------------------------------------------------------------

class VertexClaudeBackend(_VertexMixin, AnthropicBackend):
    """Claude on Vertex AI. Reuses AnthropicBackend's chat↔messages translation
    plumbing; only the URL and auth differ."""

    DEFAULT_HEALTH_PATH = ""  # we override health() below

    def __init__(self, cfg: BackendConfig) -> None:
        AnthropicBackend.__init__(self, cfg)
        self._init_vertex(cfg)

    # ---- URL builders -------------------------------------------------
    def _vertex_messages_path(self, model: str, *, stream: bool) -> str:
        verb = ":streamRawPredict" if stream else ":rawPredict"
        return (
            f"/v1/projects/{self._project}"
            f"/locations/{self._location}"
            f"/publishers/anthropic/models/{model}{verb}"
        )

    # ---- discovery / health ------------------------------------------
    async def list_models(self) -> List[Dict[str, Any]]:
        # Vertex doesn't expose /v1/models for publishers; use cfg.models.
        return [
            {"id": m, "object": "model", "owned_by": "google-vertex",
             "_backend": self.name}
            for m in (self.cfg.models or [])
        ]

    async def health(self) -> bool:
        # Just attempt to mint an access token; success ≈ creds are valid.
        try:
            await self._token_provider.get()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- override the network call for /v1/messages ------------------
    async def _messages_passthrough(
        self, anth_req: Dict[str, Any],
        headers: Optional[Dict[str, str]], *, stream: bool,
    ) -> Dict[str, Any]:
        model = anth_req.get("model")
        if not model:
            raise BadRequestError("missing 'model' in anthropic request")
        # Vertex requires anthropic_version but does NOT take 'model' in the body.
        body = dict(anth_req)
        body.pop("model", None)
        body.setdefault("anthropic_version", "vertex-2023-10-16")
        body["stream"] = stream

        h = self._merge_headers(headers)
        h.update(await self._vertex_auth_header())
        h.setdefault("content-type", "application/json")
        url = self._vertex_messages_path(model, stream=stream)
        r = await self._client.post(url, content=orjson.dumps(body), headers=h)
        if r.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.name} {url} returned {r.status_code}: {r.text[:300]}",
                status_code=r.status_code,
                extras={"backend": self.name},
            )
        out = r.json()
        # Vertex's response doesn't echo the model field; restore it so the
        # downstream translator sees a well-formed Anthropic envelope.
        if isinstance(out, dict):
            out.setdefault("model", model)
        return out

    def _messages_stream(
        self, anth_req: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> AsyncIterator[bytes]:
        model = anth_req.get("model")
        body = dict(anth_req)
        body.pop("model", None)
        body.setdefault("anthropic_version", "vertex-2023-10-16")
        body["stream"] = True

        client = self._client
        get_auth = self._vertex_auth_header
        url = self._vertex_messages_path(model or "claude", stream=True)

        async def _gen() -> AsyncIterator[bytes]:
            h = self._merge_headers(headers)
            h.update(await get_auth())
            h.setdefault("content-type", "application/json")
            h.setdefault("accept", "text/event-stream")
            async with client.stream(
                "POST", url, content=orjson.dumps(body), headers=h,
            ) as r:
                if r.status_code >= 400:
                    err = (await r.aread()).decode("utf-8", "replace")
                    raise UpstreamError(
                        f"upstream {self.name} {url} returned {r.status_code}: {err[:300]}",
                        status_code=r.status_code,
                        extras={"backend": self.name},
                    )
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

        return _gen()


register_backend("vertex-claude", VertexClaudeBackend)


# ---------------------------------------------------------------------------
# Vertex × Gemini
# ---------------------------------------------------------------------------

class VertexGeminiBackend(_VertexMixin, OpenAICompatBackend):
    """Gemini on Vertex AI. Speaks Gemini's native generateContent API,
    translates to/from OpenAI Chat Completions on the proxy side."""

    DEFAULT_HEALTH_PATH = ""  # we override health() below

    def __init__(self, cfg: BackendConfig) -> None:
        OpenAICompatBackend.__init__(self, cfg)
        self._init_vertex(cfg)

    # ---- URL builder -------------------------------------------------
    def _gemini_path(self, model: str, *, stream: bool) -> str:
        verb = ":streamGenerateContent" if stream else ":generateContent"
        sse = "?alt=sse" if stream else ""
        return (
            f"/v1/projects/{self._project}"
            f"/locations/{self._location}"
            f"/publishers/google/models/{model}{verb}{sse}"
        )

    # ---- discovery / health -----------------------------------------
    async def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": m, "object": "model", "owned_by": "google-vertex",
             "_backend": self.name}
            for m in (self.cfg.models or [])
        ]

    async def health(self) -> bool:
        try:
            await self._token_provider.get()
            return True
        except Exception:  # noqa: BLE001
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
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return await self._chat_via_gemini(json, headers, stream=False)
        return await OpenAICompatBackend.request_json(
            self, method, path, json=json, headers=headers,
        )

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        if method == "POST" and path == "/v1/chat/completions" and json is not None:
            return self._chat_via_gemini_stream(json, headers)
        return await OpenAICompatBackend.stream(
            self, method, path, json=json, headers=headers,
        )

    async def passthrough(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> PassthroughResponse:
        if method == "POST" and path == "/v1/chat/completions" and body is not None:
            try:
                payload = orjson.loads(body)
            except orjson.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                if payload.get("stream"):
                    chunks: list[bytes] = []
                    async for c in self._chat_via_gemini_stream(payload, headers):
                        chunks.append(c)
                    return PassthroughResponse(
                        200, {"content-type": "text/event-stream"},
                        b"".join(chunks),
                    )
                full = await self._chat_via_gemini(payload, headers, stream=False)
                return PassthroughResponse(
                    200, {"content-type": "application/json"},
                    orjson.dumps(full),
                )
        return await OpenAICompatBackend.passthrough(
            self, method, path, body=body, headers=headers,
        )

    # ---- core --------------------------------------------------------
    async def _chat_via_gemini(
        self, oai_req: Dict[str, Any],
        headers: Optional[Dict[str, str]], *, stream: bool,
    ) -> Dict[str, Any]:
        model = oai_req.get("model")
        if not model:
            raise BadRequestError("missing 'model' in request")
        gem_req = chat_to_gemini_request(oai_req)
        h = self._merge_headers(headers)
        h.update(await self._vertex_auth_header())
        h.setdefault("content-type", "application/json")
        url = self._gemini_path(model, stream=False)
        r = await self._client.post(url, content=orjson.dumps(gem_req), headers=h)
        if r.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.name} {url} returned {r.status_code}: {r.text[:300]}",
                status_code=r.status_code,
                extras={"backend": self.name},
            )
        return gemini_to_chat_response(r.json(), model=model)

    def _chat_via_gemini_stream(
        self, oai_req: Dict[str, Any],
        headers: Optional[Dict[str, str]],
    ) -> AsyncIterator[bytes]:
        model = oai_req.get("model")
        gem_req = chat_to_gemini_request(oai_req)
        client = self._client
        get_auth = self._vertex_auth_header
        url = self._gemini_path(model or "gemini", stream=True)
        chatcmpl_id = new_chatcmpl_id()
        created = int(time.time())

        async def _gen() -> AsyncIterator[bytes]:
            h = self._merge_headers(headers)
            h.update(await get_auth())
            h.setdefault("content-type", "application/json")
            h.setdefault("accept", "text/event-stream")
            async with client.stream(
                "POST", url, content=orjson.dumps(gem_req), headers=h,
            ) as r:
                if r.status_code >= 400:
                    err = (await r.aread()).decode("utf-8", "replace")
                    raise UpstreamError(
                        f"upstream {self.name} {url} returned {r.status_code}: {err[:300]}",
                        status_code=r.status_code,
                        extras={"backend": self.name},
                    )
                state: Dict[str, Any] = {}
                async for ev in parse_sse_lines(r.aiter_raw()):
                    for chunk in gemini_chunk_to_chat_chunks(
                        ev, state, chatcmpl_id=chatcmpl_id,
                        created=created, model=model,
                    ):
                        yield b"data: " + orjson.dumps(chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _gen()


register_backend("vertex-gemini", VertexGeminiBackend)
