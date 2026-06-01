"""Backend abstraction.

M0 ships a single concrete subclass (``VLLMBackend``) and uses it directly.
M2 will introduce ``BackendPool`` and routing strategies on top of this same
interface — the API layer should not need to be aware of pool logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional


class Backend(ABC):
    """Abstract backend. Concrete subclasses know how to talk to a specific
    upstream API (vLLM, llama.cpp, vendor cloud, mock, ...)."""

    name: str

    # ---- lifecycle ---------------------------------------------------
    @abstractmethod
    async def aclose(self) -> None: ...

    # ---- discovery ---------------------------------------------------
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def health(self) -> bool: ...

    # ---- relays ------------------------------------------------------
    @abstractmethod
    async def stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        """Yield raw bytes from the upstream response. The caller is responsible
        for any framing (line splitting, SSE event parsing, etc.)."""
        ...  # pragma: no cover

    @abstractmethod
    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Issue a non-streaming request and return parsed JSON."""
        ...

    @abstractmethod
    async def passthrough(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> "PassthroughResponse":
        """Issue a request, returning enough info for the API layer to mirror
        the response 1:1 to the client (status, content-type, body)."""
        ...


class PassthroughResponse:
    __slots__ = ("status_code", "headers", "content")

    def __init__(self, status_code: int, headers: Dict[str, str], content: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content
