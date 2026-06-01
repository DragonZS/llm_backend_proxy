"""Unit tests for per-backend auth header rendering.

Goal: prove that each ``auth.scheme`` produces the right outgoing headers AND
that client-supplied auth headers are dropped when the backend has its own key
configured (so two backends with different keys don't bleed credentials).
"""

from __future__ import annotations

import pytest

from backend_proxy.backends.openai_compat import OpenAICompatBackend
from backend_proxy.config.schema import AuthConfig, BackendConfig


def _make(auth: AuthConfig) -> OpenAICompatBackend:
    return OpenAICompatBackend(BackendConfig(
        name="x", base_url="http://e", auth=auth,
    ))


@pytest.mark.parametrize("auth,expected", [
    # Bearer token
    (AuthConfig(scheme="bearer", api_key="sk-A"),
     {"authorization": "Bearer sk-A"}),
    # x-api-key (Anthropic-style)
    (AuthConfig(scheme="x_api_key", api_key="sk-B"),
     {"x-api-key": "sk-B"}),
    # custom header
    (AuthConfig(scheme="api_key_header", api_key="sk-C", header_name="x-custom"),
     {"x-custom": "sk-C"}),
    # none
    (AuthConfig(scheme="none"), {}),
])
async def test_auth_header_rendering(auth, expected):
    b = _make(auth)
    try:
        h = b._auth_headers()
        for k, v in expected.items():
            assert h.get(k) == v, h
    finally:
        await b.aclose()


async def test_legacy_api_key_promotes_to_bearer():
    cfg = BackendConfig(name="x", base_url="http://e", api_key="sk-legacy")
    assert cfg.auth is not None
    assert cfg.auth.scheme == "bearer"
    assert cfg.auth.api_key == "sk-legacy"


async def test_no_auth_when_neither_field_set():
    cfg = BackendConfig(name="x", base_url="http://e")
    assert cfg.auth is not None and cfg.auth.scheme == "none"


async def test_explicit_auth_block_wins_over_legacy_api_key():
    cfg = BackendConfig(
        name="x", base_url="http://e",
        api_key="sk-IGNORED",
        auth=AuthConfig(scheme="x_api_key", api_key="sk-WIN"),
    )
    assert cfg.auth.scheme == "x_api_key"
    assert cfg.auth.api_key == "sk-WIN"


async def test_client_authorization_dropped_when_backend_has_own_key():
    """Default behaviour: each backend uses its own configured key, the
    caller's Authorization header is NOT forwarded. This is the property that
    keeps multiple backends with different keys isolated."""
    b = _make(AuthConfig(scheme="bearer", api_key="sk-OWN"))
    try:
        merged = b._merge_headers({"Authorization": "Bearer sk-CLIENT",
                                   "x-api-key": "sk-CLIENT-2"})
        assert merged["authorization"] == "Bearer sk-OWN"
        assert "x-api-key" not in merged or merged["x-api-key"] != "sk-CLIENT-2"
    finally:
        await b.aclose()


async def test_passthrough_scheme_forwards_client_auth():
    b = _make(AuthConfig(scheme="passthrough"))
    try:
        merged = b._merge_headers({"Authorization": "Bearer sk-CLIENT"})
        assert merged["authorization"] == "Bearer sk-CLIENT"
    finally:
        await b.aclose()


async def test_passthrough_with_no_client_auth_yields_no_auth_header():
    b = _make(AuthConfig(scheme="passthrough"))
    try:
        merged = b._merge_headers({})
        assert "authorization" not in merged
    finally:
        await b.aclose()


async def test_api_key_header_validation():
    # Schema rejects api_key_header without header_name
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        BackendConfig(name="x", base_url="http://e",
                      auth=AuthConfig(scheme="api_key_header", api_key="k"))
