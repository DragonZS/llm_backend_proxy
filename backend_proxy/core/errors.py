"""Unified error handling — domain exceptions and a helper to render them as
OpenAI-compatible JSON error envelopes."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


class ProxyError(Exception):
    """Base exception. ``status_code`` becomes the HTTP status; ``code`` is a
    short machine-readable string surfaced inside the JSON envelope."""

    status_code: int = 500
    code: str = "proxy_error"

    def __init__(self, message: str, *, code: Optional[str] = None,
                 status_code: Optional[int] = None,
                 extras: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.extras = extras or {}

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"error": {"message": self.message, "type": self.code}}
        if self.extras:
            body["error"].update(self.extras)
        return body


class BadRequestError(ProxyError):
    status_code = 400
    code = "bad_request"


class UpstreamError(ProxyError):
    status_code = 502
    code = "upstream_error"


class NoBackendAvailable(ProxyError):
    status_code = 503
    code = "no_backend_available"


def render_error(err: ProxyError) -> JSONResponse:
    return JSONResponse(status_code=err.status_code, content=err.to_dict())
