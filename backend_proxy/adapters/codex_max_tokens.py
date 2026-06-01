"""Optional adapter that fills in a default ``max_tokens`` when the client
omits it.

Codex's Responses API requests typically don't carry ``max_output_tokens``.
By default this adapter does **nothing** in that case — it leaves
``max_tokens`` unset and lets the upstream choose its own ceiling.

Why "unset" is the right default for vLLM:

  vLLM enforces ``prompt_tokens + max_tokens <= max_model_len``. When
  ``max_tokens`` is omitted, vLLM uses ``max_model_len - prompt_tokens``,
  which adapts to every request. Hard-coding a large fallback (the legacy
  ``1_010_000``) bricks long prompts: a 12k-token prompt + 1M cap exceeds
  a 1.01M context window and the upstream returns 400.

When you DO want a hard cap (e.g. to bound cost or latency), set it
explicitly — values <= 0 are treated as "unset":

    adapters:
      codex_max_tokens:
        options:
          fallback_max_tokens: 8192     # cap output at 8k tokens

Splitting this out as a separate adapter (instead of baking it into
``responses_to_chat``) lets non-Codex agents disable it cleanly.
"""

from __future__ import annotations

from typing import Any, Dict

from ..core.context import RequestContext
from .base import RequestAdapter

# 0 / negative / missing -> do not inject a fallback. Let the upstream pick.
DEFAULT_FALLBACK = 0


class CodexMaxTokensAdapter(RequestAdapter):
    name = "codex_max_tokens"

    async def transform(self, payload: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        # Already set by the caller (or by a prior adapter)? Don't override.
        if payload.get("max_tokens"):
            return payload
        try:
            fallback = int(self.options.get("fallback_max_tokens", DEFAULT_FALLBACK))
        except (TypeError, ValueError):
            fallback = DEFAULT_FALLBACK
        # <= 0 means "leave max_tokens unset and let the upstream decide";
        # vLLM will use ``max_model_len - prompt_tokens`` automatically.
        if fallback > 0:
            payload["max_tokens"] = fallback
        return payload
