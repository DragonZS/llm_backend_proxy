"""Adapter pipelines: pre-built ordered lists of adapters, run per-request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List

from ..core.context import RequestContext
from .base import RequestAdapter, ResponseAdapter, StreamAdapter


@dataclass
class AdapterPipeline:
    request:  List[RequestAdapter]  = field(default_factory=list)
    stream:   List[StreamAdapter]   = field(default_factory=list)
    response: List[ResponseAdapter] = field(default_factory=list)

    async def apply_request(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        for a in self.request:
            payload = await a.transform(payload, ctx)
        return payload

    async def apply_response(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        for a in self.response:
            payload = await a.transform(payload, ctx)
        return payload

    def apply_stream(
        self, chunks: AsyncIterator[Dict[str, Any]], ctx: RequestContext
    ) -> AsyncIterator[Dict[str, Any]]:
        out = chunks
        for a in self.stream:
            out = a.transform(out, ctx)
        return out
