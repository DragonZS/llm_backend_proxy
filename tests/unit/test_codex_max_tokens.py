"""Unit tests for the codex_max_tokens RequestAdapter.

Pinned behaviour:

* Default (no options, or ``fallback_max_tokens=0``): adapter is a no-op
  when the caller didn't specify ``max_tokens`` — leave the field unset
  so the upstream picks ``max_model_len - prompt_tokens``.
* Positive ``fallback_max_tokens``: inject it ONLY when ``max_tokens`` is
  missing/falsy.
* If the caller (or a prior adapter) already set ``max_tokens``, never
  overwrite — even when ``fallback_max_tokens`` is configured.

Regression: a previously baked-in fallback of 1_010_000 caused vLLM to
reject prompts longer than ``max_model_len - 1_010_000`` tokens with HTTP
400, because vLLM enforces ``prompt_tokens + max_tokens <= max_model_len``.
"""

from __future__ import annotations

import pytest

from backend_proxy.adapters.codex_max_tokens import CodexMaxTokensAdapter
from backend_proxy.core.context import RequestContext


def _ctx() -> RequestContext:
    return RequestContext.new(method="POST", path="/v1/responses", headers={})


@pytest.mark.asyncio
async def test_default_options_does_not_inject_max_tokens():
    """No options at all → leave max_tokens unset (let upstream decide)."""
    adapter = CodexMaxTokensAdapter()
    out = await adapter.transform({"model": "m", "messages": []}, _ctx())
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_zero_fallback_does_not_inject():
    """fallback=0 is the explicit "let upstream decide" sentinel."""
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": 0})
    out = await adapter.transform({"model": "m"}, _ctx())
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_negative_fallback_does_not_inject():
    """Negative values are treated as "unset" — no max_tokens emitted."""
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": -1})
    out = await adapter.transform({"model": "m"}, _ctx())
    assert "max_tokens" not in out


@pytest.mark.asyncio
async def test_positive_fallback_injects_when_missing():
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": 8192})
    out = await adapter.transform({"model": "m"}, _ctx())
    assert out["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_existing_max_tokens_is_not_overwritten():
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": 8192})
    out = await adapter.transform({"model": "m", "max_tokens": 50}, _ctx())
    assert out["max_tokens"] == 50  # preserved, not stomped


@pytest.mark.asyncio
async def test_string_options_value_is_coerced():
    """YAML can hand us a string; coerce defensively without crashing."""
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": "1024"})
    out = await adapter.transform({"model": "m"}, _ctx())
    assert out["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_garbage_options_value_falls_back_to_default():
    """A non-coercible value must NOT crash the pipeline."""
    adapter = CodexMaxTokensAdapter({"fallback_max_tokens": "not-a-number"})
    out = await adapter.transform({"model": "m"}, _ctx())
    # default is 0 → unset
    assert "max_tokens" not in out
