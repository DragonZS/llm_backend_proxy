"""Adapter base classes.

Three kinds of adapters operate on three different stages:

* ``RequestAdapter``  — runs **before** the upstream call, mutating the JSON
  payload that will be sent. Example: ``responses_to_chat`` (rewrites a
  Responses-API body into a Chat-Completions body).
* ``StreamAdapter``   — async-iterator transformer that consumes upstream SSE
  events (already parsed into dicts) and yields the events the client should
  see. Example: ``chat_to_responses_stream``.
* ``ResponseAdapter`` — for the non-streaming path: takes the upstream's
  parsed JSON response and rewrites it for the client. Example:
  ``chat_to_responses``.

All adapters are stateless w.r.t. process-global state: any per-request state
lives in ``RequestContext.extras``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional

from ..core.context import RequestContext


class RequestAdapter(ABC):
    name: str

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}

    @abstractmethod
    async def transform(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]: ...


class StreamAdapter(ABC):
    name: str

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}

    @abstractmethod
    def transform(
        self, chunks: AsyncIterator[Dict[str, Any]], ctx: RequestContext
    ) -> AsyncIterator[Dict[str, Any]]: ...


class ResponseAdapter(ABC):
    name: str

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}

    @abstractmethod
    async def transform(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]: ...
