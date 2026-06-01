"""Unit tests for the async TokenBucket."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend_proxy.backends.rate_limit import RateLimited, TokenBucket


def test_disabled_when_rps_zero():
    tb = TokenBucket(rps=0, burst=0)
    assert not tb.enabled
    # acquire returns immediately even without enabled state
    asyncio.run(tb.acquire())


async def test_initial_burst_capacity_consumed_immediately():
    tb = TokenBucket(rps=10, burst=3)
    t0 = time.monotonic()
    for _ in range(3):
        await tb.acquire()
    elapsed = time.monotonic() - t0
    # All three should fit in the initial burst → near-instant.
    assert elapsed < 0.05


async def test_reject_mode_raises_when_empty():
    tb = TokenBucket(rps=1, burst=1)
    await tb.acquire()
    with pytest.raises(RateLimited):
        await tb.acquire(mode="reject")


async def test_wait_mode_blocks_until_token_available():
    tb = TokenBucket(rps=10, burst=1)
    await tb.acquire()  # drain
    t0 = time.monotonic()
    await tb.acquire(mode="wait")  # should wait ~0.1s
    elapsed = time.monotonic() - t0
    assert 0.05 <= elapsed <= 0.5


async def test_refill_replenishes_over_time():
    tb = TokenBucket(rps=20, burst=2)
    await tb.acquire()
    await tb.acquire()
    # bucket now empty; sleep 100ms → ~2 tokens refilled
    await asyncio.sleep(0.1)
    # both should fit without further wait
    t0 = time.monotonic()
    await tb.acquire()
    await tb.acquire()
    assert time.monotonic() - t0 < 0.05
