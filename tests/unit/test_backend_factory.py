"""Unit tests for the backend type factory + per-type presets."""

from __future__ import annotations

import pytest

from backend_proxy.backends import (
    LlamaCppBackend,
    OllamaOpenAIBackend,
    OpenAICloudBackend,
    SGLangBackend,
    TGIBackend,
    VLLMBackend,
    make_backend,
)
from backend_proxy.config.schema import BackendConfig


@pytest.mark.parametrize("type_name,cls,health", [
    ("vllm",          VLLMBackend,         "/health"),
    ("tgi",           TGIBackend,          "/health"),
    ("llamacpp",      LlamaCppBackend,     "/health"),
    ("sglang",        SGLangBackend,       "/health"),
    ("ollama-openai", OllamaOpenAIBackend, "/"),
    ("openai",        OpenAICloudBackend,  "/v1/models"),
])
def test_factory_picks_right_class_and_health_path(type_name, cls, health):
    cfg = BackendConfig(name="x", type=type_name, base_url="http://example")
    b = make_backend(cfg)
    try:
        assert isinstance(b, cls)
        assert b.health_path == health
    finally:
        # backends create httpx clients eagerly; close to avoid leaked sockets
        import asyncio
        asyncio.run(b.aclose())


def test_explicit_health_path_overrides_preset():
    cfg = BackendConfig(name="x", type="vllm", base_url="http://example",
                        health_path="/custom/health")
    b = make_backend(cfg)
    try:
        assert b.health_path == "/custom/health"
    finally:
        import asyncio
        asyncio.run(b.aclose())


def test_unknown_type_raises():
    # Bypass the Literal validator with model_construct so we hit the factory's
    # explicit ValueError path rather than pydantic's.
    cfg = BackendConfig.model_construct(
        name="x", type="nope", base_url="http://e",  # type: ignore[arg-type]
        api_key=None, models=[], weight=1, max_concurrency=64, timeout=600.0,
        health_path=None, headers={}, auth=None,
    )
    with pytest.raises(ValueError):
        make_backend(cfg)


def test_invalid_type_rejected_by_schema():
    with pytest.raises(Exception):
        BackendConfig(name="x", type="nope", base_url="http://e")  # type: ignore[arg-type]


def test_default_type_is_vllm():
    cfg = BackendConfig(name="x", base_url="http://example")
    assert cfg.type == "vllm"
    b = make_backend(cfg)
    try:
        assert isinstance(b, VLLMBackend)
    finally:
        import asyncio
        asyncio.run(b.aclose())
