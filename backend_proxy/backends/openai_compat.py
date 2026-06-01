"""Backend talking the OpenAI Chat-Completions wire format.

This is the shared implementation behind every "OpenAI-compatible" backend
type — vLLM, TGI, llama.cpp, SGLang, Ollama-in-compat-mode, and the official
``api.openai.com``. Per-type knobs live in tiny ``preset_*`` subclasses or in
``factory.make_backend()``.

Per-backend auth: each backend renders its own ``Authorization`` (or other)
header from ``BackendConfig.auth``. Client-supplied ``Authorization`` is NOT
forwarded by default — that's exactly how independent per-backend keys are
isolated. The ``passthrough`` auth scheme opts back into the legacy behaviour.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

import httpx
import orjson

from ..config.schema import BackendConfig, AuthConfig
from ..core.errors import UpstreamError
from .base import Backend, PassthroughResponse


# Hop-by-hop headers we never forward — keeps httpx in control of framing.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
    "content-encoding", "content-length", "host",
}


class OpenAICompatBackend(Backend):
    """Backend speaking the OpenAI Chat-Completions wire format.

    Subclasses can override:
      * ``DEFAULT_HEALTH_PATH`` — used when ``cfg.health_path is None``.
      * ``models_path``         — for backends that don't expose ``/v1/models``.
    """

    DEFAULT_HEALTH_PATH: str = "/health"
    models_path: str = "/v1/models"

    def __init__(self, cfg: BackendConfig) -> None:
        self.name = cfg.name
        self.cfg = cfg
        connect = cfg.connect
        pool_max = connect.pool_max_connections or max(cfg.max_concurrency * 2, 64)
        pool_keepalive = connect.pool_max_keepalive or max(cfg.max_concurrency, 32)
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=httpx.Timeout(cfg.timeout, connect=connect.connect_timeout),
            limits=httpx.Limits(
                max_connections=pool_max,
                max_keepalive_connections=pool_keepalive,
                keepalive_expiry=connect.keepalive_expiry,
            ),
            headers=self._auth_headers(),
        )
        # Independent client for liveness probes:
        #   * short fixed timeout (probes must NOT inherit the 600s inference
        #     timeout — a stuck probe would block the watcher loop)
        #   * no keepalive (avoid sharing sockets that vLLM may close mid-stream
        #     during heavy generation, which would surface as RemoteProtocolError
        #     and falsely mark the backend unhealthy)
        # Built lazily so a config without HealthCheckConfig still works; the
        # pool overrides the timeout via ``set_health_probe(timeout_s=...)``.
        self._probe_client: Optional[httpx.AsyncClient] = None
        self._probe_timeout_s: float = 2.0

    def set_health_probe(self, *, timeout_s: float) -> None:
        """Configure the probe client. Called by the pool after construction
        so the routing-level HealthCheckConfig wins over the backend default."""
        self._probe_timeout_s = max(0.1, float(timeout_s))
        # Recreate if already built so the new timeout takes effect.
        if self._probe_client is not None:
            try:
                # Schedule async close; safe to fire-and-forget here.
                import asyncio
                asyncio.get_event_loop().create_task(self._probe_client.aclose())
            except Exception:  # noqa: BLE001
                pass
            self._probe_client = None

    def _get_probe_client(self) -> httpx.AsyncClient:
        if self._probe_client is None:
            self._probe_client = httpx.AsyncClient(
                base_url=self.cfg.base_url,
                timeout=httpx.Timeout(self._probe_timeout_s),
                limits=httpx.Limits(
                    max_connections=2,
                    max_keepalive_connections=0,    # always fresh TCP for probes
                    keepalive_expiry=0.0,
                ),
                headers=self._auth_headers(),
            )
        return self._probe_client
        # Independent client for liveness probes:
        #   * short fixed timeout (probes must NOT inherit the 600s inference
        #     timeout — a stuck probe would block the watcher loop)
        #   * no keepalive (avoid sharing sockets that vLLM may close mid-stream
        #     during heavy generation, which would surface as RemoteProtocolError
        #     and falsely mark the backend unhealthy)
        # Built lazily so a config without HealthCheckConfig still works; the
        # pool overrides the timeout via ``set_health_probe(timeout_s=...)``.
        self._probe_client: Optional[httpx.AsyncClient] = None
        self._probe_timeout_s: float = 2.0

    def set_health_probe(self, *, timeout_s: float) -> None:
        """Configure the probe client. Called by the pool after construction
        so the routing-level HealthCheckConfig wins over the backend default."""
        self._probe_timeout_s = max(0.1, float(timeout_s))
        # Recreate if already built so the new timeout takes effect.
        if self._probe_client is not None:
            try:
                # Schedule async close; safe to fire-and-forget here.
                import asyncio
                asyncio.get_event_loop().create_task(self._probe_client.aclose())
            except Exception:  # noqa: BLE001
                pass
            self._probe_client = None

    def _get_probe_client(self) -> httpx.AsyncClient:
        if self._probe_client is None:
            self._probe_client = httpx.AsyncClient(
                base_url=self.cfg.base_url,
                timeout=httpx.Timeout(self._probe_timeout_s),
                limits=httpx.Limits(
                    max_connections=2,
                    max_keepalive_connections=0,    # always fresh TCP for probes
                    keepalive_expiry=0.0,
                ),
                headers=self._auth_headers(),
            )
        return self._probe_client

    @property
    def health_path(self) -> str:
        return self.cfg.health_path or self.DEFAULT_HEALTH_PATH

    # ---- auth --------------------------------------------------------
    def _auth_headers(self, *, client_auth: Optional[str] = None) -> Dict[str, str]:
        """Render per-backend auth headers. ``client_auth`` is the caller's own
        ``Authorization`` value, only used when scheme is ``passthrough``."""
        h: Dict[str, str] = {"accept": "application/json"}
        h.update(self.cfg.headers or {})

        auth = self.cfg.auth or AuthConfig(scheme="none")
        scheme = auth.scheme

        if scheme == "bearer" and auth.api_key:
            h["authorization"] = f"Bearer {auth.api_key}"
        elif scheme == "x_api_key" and auth.api_key:
            h["x-api-key"] = auth.api_key
        elif scheme == "api_key_header" and auth.api_key and auth.header_name:
            h[auth.header_name.lower()] = auth.api_key
        elif scheme == "passthrough":
            if client_auth:
                h["authorization"] = client_auth
        # scheme == "none": no auth headers
        return h

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._probe_client is not None:
            try:
                await self._probe_client.aclose()
            finally:
                self._probe_client = None
        if self._probe_client is not None:
            try:
                await self._probe_client.aclose()
            finally:
                self._probe_client = None

    # ---- discovery / health -----------------------------------------
    async def list_models(self) -> list[Dict[str, Any]]:
        r = await self._client.get(self.models_path)
        if r.status_code != 200:
            raise UpstreamError(
                f"backend {self.name} {self.models_path} returned {r.status_code}",
                extras={"backend": self.name},
            )
        body = r.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return []
        return [{**m, "_backend": self.name} for m in data if isinstance(m, dict)]

    async def health(self) -> bool:
        try:
            r = await self._get_probe_client().get(self.health_path)
            return 200 <= r.status_code < 500
        except httpx.HTTPError:
            return False

    # ---- relays ------------------------------------------------------
    def _merge_headers(self, extra: Optional[Dict[str, str]]) -> Dict[str, str]:
        client_auth = None
        if extra:
            client_auth = extra.get("authorization") or extra.get("Authorization")
        out = dict(self._auth_headers(client_auth=client_auth))

        scheme = (self.cfg.auth or AuthConfig(scheme="none")).scheme
        if extra:
            for k, v in extra.items():
                kl = k.lower()
                if kl in _HOP_BY_HOP:
                    continue
                # When auth is NOT passthrough, drop the caller's credential
                # headers so each backend uses only its own configured key.
                if scheme != "passthrough" and kl in ("authorization", "x-api-key"):
                    continue
                # Don't let arbitrary callers pin a specific API key header.
                if (scheme == "api_key_header"
                        and self.cfg.auth and self.cfg.auth.header_name
                        and kl == self.cfg.auth.header_name.lower()):
                    continue
                out[kl] = v
        return out

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        body = orjson.dumps(json) if json is not None else None
        h = self._merge_headers(headers)
        if body is not None:
            h.setdefault("content-type", "application/json")
        # Single-shot retry on PURE CONNECTION-LEVEL errors (closed keepalive
        # socket, mid-flight reset, DNS hiccup). 4xx/5xx responses from the
        # upstream are NOT retried here — those are real outcomes the caller
        # must see. Idempotency is the user's responsibility for non-GET; we
        # only retry once and only on connection failures, which means the
        # upstream cannot have started processing the request. This is what
        # turns a one-off keepalive RST into a transparent recovery instead of
        # a user-visible 502.
        attempt = 0
        while True:
            try:
                r = await self._client.request(method, path, content=body, headers=h)
                break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                    httpx.WriteError, httpx.PoolTimeout) as e:
                if attempt >= 1:
                    raise UpstreamError(
                        f"upstream {self.name} connection error after retry: {type(e).__name__}: {e}",
                        status_code=502, code="connection_error",
                        extras={"backend": self.name},
                    ) from e
                attempt += 1
        if r.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.name} returned {r.status_code}: {r.text[:300]}",
                status_code=r.status_code,
                extras={"backend": self.name},
            )
        return r.json()

    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        body = orjson.dumps(json) if json is not None else None
        h = self._merge_headers(headers)
        if body is not None:
            h.setdefault("content-type", "application/json")
        h.setdefault("accept", "text/event-stream")

        async def _gen() -> AsyncIterator[bytes]:
            async with self._client.stream(method, path, content=body, headers=h) as r:
                if r.status_code >= 400:
                    err_body = (await r.aread()).decode("utf-8", "replace")
                    raise UpstreamError(
                        f"upstream {self.name} stream returned {r.status_code}: {err_body[:300]}",
                        status_code=r.status_code,
                        extras={"backend": self.name},
                    )
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk
        return _gen()

    async def passthrough(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> PassthroughResponse:
        h = self._merge_headers(headers)
        r = await self._client.request(method, path, content=body, headers=h)
        out_headers = {
            k: v for k, v in r.headers.items() if k.lower() not in _HOP_BY_HOP
        }
        return PassthroughResponse(r.status_code, out_headers, r.content)


# ---------------------------------------------------------------------------
# Per-type presets
# ---------------------------------------------------------------------------

class VLLMBackend(OpenAICompatBackend):
    """vLLM exposes ``/health`` as a 200/OK endpoint."""
    DEFAULT_HEALTH_PATH = "/health"


class TGIBackend(OpenAICompatBackend):
    """HuggingFace text-generation-inference: ``/health``."""
    DEFAULT_HEALTH_PATH = "/health"


class LlamaCppBackend(OpenAICompatBackend):
    """llama.cpp ``server`` (--api): ``/health``."""
    DEFAULT_HEALTH_PATH = "/health"


class SGLangBackend(OpenAICompatBackend):
    """SGLang server: ``/health``."""
    DEFAULT_HEALTH_PATH = "/health"


class OllamaOpenAIBackend(OpenAICompatBackend):
    """Ollama 0.1.31+ exposes /v1/chat/completions and /v1/models. Health
    endpoint is the root path (returns 'Ollama is running'). list_models still
    works through /v1/models."""
    DEFAULT_HEALTH_PATH = "/"


class OpenAICloudBackend(OpenAICompatBackend):
    """api.openai.com (or any OpenAI-shaped cloud). No /health endpoint —
    treat /v1/models as a liveness probe."""
    DEFAULT_HEALTH_PATH = "/v1/models"
