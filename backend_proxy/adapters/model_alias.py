"""Map client-facing model names to backend-facing ones, e.g.
``gpt-5-codex -> qwen3-coder-30b``."""

from __future__ import annotations

from typing import Any, Dict

from ..core.context import RequestContext
from .base import RequestAdapter


class ModelAliasAdapter(RequestAdapter):
    name = "model_alias"

    async def transform(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        aliases: Dict[str, str] = self.options.get("aliases", {})
        if not aliases:
            return payload
        m = payload.get("model")
        if isinstance(m, str) and m in aliases:
            payload["model"] = aliases[m]
        return payload
