"""Unit tests for retry logic with exponential backoff."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend_proxy.backends.base import Backend, PassthroughResponse
from backend_proxy.backends.pool import BackendPool
from backend_proxy.backends.retry import (
    _compute_backoff,
    _effective_retry_config,
    _is_retryable,
    retry_passthrough,
    retry_request_json,
    retry_stream_start,
)
from backend_proxy.config.schema import (
    BackendConfig,
    CircuitBreakerConfig,
    RetryConfig,
    RoutingConfig,
)
from backend_proxy.core.errors import NoBackendAvailable, UpstreamError


# ---- helpers --------------------------------------------------------

def _make_config(**kw) -> BackendConfig:
    defaults = dict(name="test-be", base_url="http://localhost:8080")
    defaults.update(kw)
    return BackendConfig(**defaults)


def _make_pool(n: int = 1, routing: Optional[RoutingConfig] = None) -> BackendPool:
    from backend_proxy.backends.factory import make_backend
    cfgs = [_make_config(name=f"be-{i}") for i in range(n)]
    routing = routing or RoutingConfig(
        circuit_breaker=CircuitBreakerConfig(failures=50, cooldown_s=1),
    )
    return BackendPool([make_backend(c) for c in cfgs], routing)


# ---- _is_retryable --------------------------------------------------

def test_network_connect_error_is_retryable():
    assert _is_retryable(httpx.ConnectError("refused"), [500]) is True


def test_network_timeout_is_retryable():
    assert _is_retryable(httpx.TimeoutException("timed out"), [500]) is True


def test_upstream_500_is_retryable():
    exc = UpstreamError("oops", status_code=500)
    assert _is_retryable(exc, [429, 500, 502, 503, 504]) is True


def test_upstream_429_is_retryable():
    exc = UpstreamError("rate limited", status_code=429)
    assert _is_retryable(exc, [429, 500, 502, 503, 504]) is True


def test_upstream_400_is_not_retryable():
    exc = UpstreamError("bad request", status_code=400)
    assert _is_retryable(exc, [429, 500, 502, 503, 504]) is False


def test_local_rate_limited_429_is_not_retryable():
    exc = UpstreamError("rate_limited", status_code=429, code="rate_limited")
    assert _is_retryable(exc, [429, 500, 502, 503, 504]) is False


def test_generic_exception_is_not_retryable():
    assert _is_retryable(ValueError("unexpected"), [500]) is False


# ---- _compute_backoff -----------------------------------------------

def test_backoff_increases_with_attempts():
    cfg = RetryConfig(initial_backoff=1.0, max_backoff=60.0, backoff_multiplier=2.0)
    b0 = _compute_backoff(0, cfg)
    b1 = _compute_backoff(1, cfg)
    b2 = _compute_backoff(2, cfg)
    # b0 ≈ 1.0, b1 ≈ 2.0, b2 ≈ 4.0 (all + jitter)
    assert b0 < b1 < b2


def test_backoff_respects_max():
    cfg = RetryConfig(initial_backoff=1.0, max_backoff=3.0, backoff_multiplier=10.0)
    # Even at attempt=10, base would be 1.0 * 10^10 but capped at 3.0
    b = _compute_backoff(10, cfg)
    # b = min(3.0, huge) + jitter; jitter is at most 0.25 * 3.0 = 0.75
    assert b <= 3.0 + 0.75


def test_backoff_includes_jitter():
    cfg = RetryConfig(initial_backoff=1.0, max_backoff=60.0, backoff_multiplier=2.0)
    # Run several times to verify jitter introduces variation
    values = [_compute_backoff(1, cfg) for _ in range(20)]
    assert len(set(values)) > 1  # not all identical


# ---- _effective_retry_config ----------------------------------------

def test_backend_config_wins_over_pool():
    pool_cfg = RetryConfig(max_retries=1)
    be_cfg = RetryConfig(max_retries=5)
    result = _effective_retry_config(pool_cfg, be_cfg)
    assert result.max_retries == 5


def test_pool_config_used_when_backend_is_none():
    pool_cfg = RetryConfig(max_retries=3)
    result = _effective_retry_config(pool_cfg, None)
    assert result.max_retries == 3


def test_defaults_when_both_none():
    result = _effective_retry_config(None, None)
    assert result.max_retries == 2  # default


# ---- retry_request_json ---------------------------------------------

async def test_request_json_succeeds_first_try():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.request_json = AsyncMock(return_value={"choices": []})  # type: ignore[assignment]

    result, be = await retry_request_json(
        pool, model=None, method="POST", path="/v1/chat/completions",
        json={"model": "test"}, headers={},
        retry_cfg=RetryConfig(max_retries=2),
    )
    assert result == {"choices": []}
    assert be is backend
    assert backend.request_json.call_count == 1


async def test_request_json_retries_on_retryable_error():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.request_json = AsyncMock(  # type: ignore[assignment]
        side_effect=[UpstreamError("502", status_code=502), {"choices": []}]
    )

    result, be = await retry_request_json(
        pool, model=None, method="POST", path="/v1/chat/completions",
        json={"model": "test"}, headers={},
        retry_cfg=RetryConfig(max_retries=2, initial_backoff=0.01),
    )
    assert result == {"choices": []}
    assert backend.request_json.call_count == 2


async def test_request_json_raises_non_retryable():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.request_json = AsyncMock(  # type: ignore[assignment]
        side_effect=UpstreamError("bad request", status_code=400)
    )

    with pytest.raises(UpstreamError, match="bad request"):
        await retry_request_json(
            pool, model=None, method="POST", path="/v1/chat/completions",
            json={}, headers={},
            retry_cfg=RetryConfig(max_retries=3, initial_backoff=0.01),
        )
    assert backend.request_json.call_count == 1


async def test_request_json_exhausts_retries():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.request_json = AsyncMock(  # type: ignore[assignment]
        side_effect=UpstreamError("503", status_code=503)
    )

    with pytest.raises(UpstreamError, match="503"):
        await retry_request_json(
            pool, model=None, method="POST", path="/v1/chat/completions",
            json={}, headers={},
            retry_cfg=RetryConfig(max_retries=1, initial_backoff=0.01),
        )
    # max_retries=1 means 2 total attempts (initial + 1 retry)
    assert backend.request_json.call_count == 2


async def test_request_json_zero_retries_no_retry():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.request_json = AsyncMock(  # type: ignore[assignment]
        side_effect=UpstreamError("500", status_code=500)
    )

    with pytest.raises(UpstreamError):
        await retry_request_json(
            pool, model=None, method="POST", path="/v1/chat/completions",
            json={}, headers={},
            retry_cfg=RetryConfig(max_retries=0),
        )
    assert backend.request_json.call_count == 1


# ---- retry_stream_start ---------------------------------------------

async def test_stream_start_succeeds_immediately():
    pool = _make_pool()
    backend = pool.backends[0]

    async def _fake_stream(method, path, *, json=None, headers=None):
        async def _gen():
            yield b'data: {"choices":[]}\n\n'
        return _gen()

    backend.stream = AsyncMock(side_effect=_fake_stream)  # type: ignore[assignment]

    upstream_bytes, be, slot = await retry_stream_start(
        pool, model=None, method="POST", path="/v1/chat/completions",
        json={"model": "test"}, headers={},
        retry_cfg=RetryConfig(max_retries=2),
    )
    # Should be able to read the chunk
    chunks = []
    async for c in upstream_bytes:
        chunks.append(c)
    assert len(chunks) == 1
    assert b"choices" in chunks[0]
    await slot.__aexit__(None, None, None)


async def test_stream_start_retries_on_connection_error():
    pool = _make_pool()
    backend = pool.backends[0]

    call_count = 0

    async def _fake_stream(method, path, *, json=None, headers=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("refused")
        async def _gen():
            yield b'data: {"ok":true}\n\n'
        return _gen()

    backend.stream = AsyncMock(side_effect=_fake_stream)  # type: ignore[assignment]

    upstream_bytes, be, slot = await retry_stream_start(
        pool, model=None, method="POST", path="/v1/chat/completions",
        json={"model": "test"}, headers={},
        retry_cfg=RetryConfig(max_retries=2, initial_backoff=0.01),
    )
    chunks = []
    async for c in upstream_bytes:
        chunks.append(c)
    assert len(chunks) == 1
    await slot.__aexit__(None, None, None)


async def test_stream_start_raises_non_retryable():
    pool = _make_pool()
    backend = pool.backends[0]

    async def _fail_stream(method, path, *, json=None, headers=None):
        raise UpstreamError("bad request", status_code=400)

    backend.stream = AsyncMock(side_effect=_fail_stream)  # type: ignore[assignment]

    with pytest.raises(UpstreamError, match="bad request"):
        await retry_stream_start(
            pool, model=None, method="POST", path="/v1/chat/completions",
            json={}, headers={},
            retry_cfg=RetryConfig(max_retries=3, initial_backoff=0.01),
        )


# ---- retry_passthrough ----------------------------------------------

async def test_passthrough_succeeds_first_try():
    pool = _make_pool()
    backend = pool.backends[0]
    resp = PassthroughResponse(200, {}, b'{"ok":true}')
    backend.passthrough = AsyncMock(return_value=resp)  # type: ignore[assignment]

    result, be = await retry_passthrough(
        pool, model=None, method="GET", path="/v1/models",
        body=None, headers={},
        retry_cfg=RetryConfig(max_retries=2),
    )
    assert result.status_code == 200
    assert be is backend


async def test_passthrough_retries_on_502():
    pool = _make_pool()
    backend = pool.backends[0]
    resp = PassthroughResponse(200, {}, b'{"ok":true}')
    backend.passthrough = AsyncMock(  # type: ignore[assignment]
        side_effect=[UpstreamError("502", status_code=502), resp]
    )

    result, be = await retry_passthrough(
        pool, model=None, method="GET", path="/v1/models",
        body=None, headers={},
        retry_cfg=RetryConfig(max_retries=2, initial_backoff=0.01),
    )
    assert result.status_code == 200


async def test_passthrough_no_retry_on_401():
    pool = _make_pool()
    backend = pool.backends[0]
    backend.passthrough = AsyncMock(  # type: ignore[assignment]
        side_effect=UpstreamError("unauthorized", status_code=401)
    )

    with pytest.raises(UpstreamError, match="unauthorized"):
        await retry_passthrough(
            pool, model=None, method="GET", path="/v1/models",
            body=None, headers={},
            retry_cfg=RetryConfig(max_retries=3, initial_backoff=0.01),
        )
    assert backend.passthrough.call_count == 1
