"""Async token-bucket rate limiter, used per-backend."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


class RateLimited(Exception):
    """Raised by ``TokenBucket.acquire(mode='reject')`` when no token is
    available. The pool catches this and turns it into a 429."""


@dataclass
class TokenBucket:
    rps: float                     # refill rate, tokens per second
    burst: float                   # max bucket size
    _tokens: float = 0.0
    _last: float = 0.0

    def __post_init__(self) -> None:
        # Start full so the first burst worth of requests goes through immediately.
        self._tokens = self.burst
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.rps > 0 and self.burst > 0

    def _refill(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rps)
            self._last = now

    async def acquire(self, *, mode: str = "wait") -> None:
        """Take one token. With ``mode='wait'`` block until available; with
        ``mode='reject'`` raise :class:`RateLimited` immediately on miss."""
        if not self.enabled:
            return
        async with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            if mode == "reject":
                raise RateLimited(
                    f"rate limit exceeded (rps={self.rps}, burst={self.burst})"
                )
            # wait mode: figure out how long until we can have one token
            need = 1.0 - self._tokens
            wait_s = need / self.rps
        # Sleep outside the lock so other waiters can also schedule themselves.
        await asyncio.sleep(wait_s)
        async with self._lock:
            self._refill(time.monotonic())
            # Best-effort consume; if another waiter beat us, fall back to
            # a short re-wait. Bounded recursion: rps>0 ensures progress.
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
        await self.acquire(mode=mode)
