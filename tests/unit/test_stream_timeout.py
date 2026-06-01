"""Unit tests for streaming timeout (with_timeout)."""

from __future__ import annotations

import asyncio

import pytest

from backend_proxy.streaming.sse import StreamTimeoutError, with_timeout


async def _chunks(*items: bytes, delay: float = 0.0) -> asyncio.AsyncIterator[bytes]:
    """Helper that yields chunks with an optional inter-chunk delay."""
    for item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item


async def test_no_timeout_passes_all_chunks():
    async def upstream():
        for c in (b"a", b"b", b"c"):
            yield c

    result = []
    async for chunk in with_timeout(upstream(), idle_timeout_s=0, absolute_timeout_s=0):
        result.append(chunk)
    assert result == [b"a", b"b", b"c"]


async def test_idle_timeout_raises_when_upstream_stalls():
    async def upstream():
        yield b"first"
        await asyncio.sleep(1.0)  # stall longer than idle timeout
        yield b"never"

    with pytest.raises(StreamTimeoutError, match="idle timeout"):
        async for _ in with_timeout(upstream(), idle_timeout_s=0.1, absolute_timeout_s=0):
            pass


async def test_absolute_timeout_raises_when_duration_exceeded():
    async def upstream():
        for i in range(100):
            await asyncio.sleep(0.05)
            yield f"chunk-{i}".encode()

    with pytest.raises(StreamTimeoutError, match="absolute timeout"):
        async for _ in with_timeout(upstream(), idle_timeout_s=0, absolute_timeout_s=0.1):
            pass


async def test_stream_completes_within_timeout():
    async def upstream():
        yield b"a"
        yield b"b"

    result = []
    async for chunk in with_timeout(upstream(), idle_timeout_s=5.0, absolute_timeout_s=5.0):
        result.append(chunk)
    assert result == [b"a", b"b"]


async def test_idle_timeout_not_triggered_by_fast_chunks():
    async def upstream():
        for i in range(5):
            await asyncio.sleep(0.02)
            yield f"x{i}".encode()

    result = []
    async for chunk in with_timeout(upstream(), idle_timeout_s=0.5, absolute_timeout_s=0):
        result.append(chunk)
    assert len(result) == 5


async def test_stream_timeout_error_has_reason():
    err = StreamTimeoutError("test reason")
    assert err.reason == "test reason"
    assert "test reason" in str(err)


async def test_idle_timeout_allows_some_chunks_before_timeout():
    """First chunk arrives quickly, second arrives after stall."""
    chunks_received = []

    async def upstream():
        yield b"fast"
        await asyncio.sleep(0.5)  # exceeds idle timeout of 0.2s
        yield b"slow"

    with pytest.raises(StreamTimeoutError):
        async for chunk in with_timeout(upstream(), idle_timeout_s=0.2, absolute_timeout_s=0):
            chunks_received.append(chunk)

    # Should have received the fast chunk before timing out
    assert b"fast" in chunks_received
