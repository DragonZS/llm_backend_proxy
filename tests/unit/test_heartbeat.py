"""Unit tests for the SSE heartbeat injector."""

from __future__ import annotations

import asyncio

from backend_proxy.streaming import with_heartbeat


async def test_heartbeat_passes_through_when_upstream_is_busy():
    async def upstream():
        for c in (b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"):
            yield c

    chunks = []
    async for c in with_heartbeat(upstream(), interval=10.0, payload="hb"):
        chunks.append(c)
    assert b"".join(chunks) == b"data: a\n\ndata: b\n\ndata: c\n\n"


async def test_heartbeat_emitted_when_upstream_silent():
    async def upstream():
        # Stay silent for ~150ms then emit, total run < 250ms
        await asyncio.sleep(0.15)
        yield b"data: x\n\n"

    chunks = []
    async for c in with_heartbeat(upstream(), interval=0.05, payload="hb"):
        chunks.append(c)
    text = b"".join(chunks).decode()
    # We expect at least one comment frame (": hb\n\n") before the data
    assert ": hb\n\n" in text
    assert "data: x" in text
    # Heartbeat must come strictly before the actual data
    assert text.index(": hb") < text.index("data: x")


async def test_heartbeat_disabled_when_interval_zero():
    async def upstream():
        await asyncio.sleep(0.05)
        yield b"data: x\n\n"

    chunks = []
    async for c in with_heartbeat(upstream(), interval=0.0):
        chunks.append(c)
    assert b"".join(chunks) == b"data: x\n\n"


async def test_heartbeat_continues_after_each_emit():
    """Heartbeat clock resets after each upstream chunk."""
    async def upstream():
        yield b"data: a\n\n"
        await asyncio.sleep(0.12)
        yield b"data: b\n\n"

    chunks = []
    async for c in with_heartbeat(upstream(), interval=0.04, payload="hb"):
        chunks.append(c)
    text = b"".join(chunks).decode()
    assert text.index("data: a") < text.index(": hb") < text.index("data: b")
