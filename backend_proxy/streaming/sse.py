"""Async iterators for SSE handling.

* ``parse_sse_lines(byte_stream)`` — yields parsed JSON dicts from a raw byte
  stream that follows the SSE protocol. Skips ``[DONE]`` / heartbeats /
  malformed payloads silently (matches the legacy ``proxy_v1`` lenience).

* ``serialise_events(events)`` — turns dicts back into ``data: …\\n\\n``
  bytes and tacks on ``data: [DONE]\\n\\n`` at the end.

* ``with_heartbeat(byte_stream, *, interval, payload)`` — wraps any outbound
  byte iterator and injects ``: <payload>\\n\\n`` whenever the upstream goes
  silent for longer than ``interval`` seconds. SSE clients ignore comment
  lines so this is safe even for downstream parsers.

* ``with_timeout(byte_stream, *, idle_timeout_s, absolute_timeout_s)`` — wraps
  a byte iterator and raises ``StreamTimeoutError`` when the upstream is idle
  too long or the total stream duration exceeds a limit. The error flows
  through the existing stream adapter error handling to produce a clean
  ``response.completed`` with ``status=incomplete``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional

import orjson


class StreamTimeoutError(Exception):
    """Raised when a streaming request exceeds its idle or absolute timeout."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"stream timeout: {reason}")


async def parse_sse_lines(byte_stream: AsyncIterator[bytes]) -> AsyncIterator[Dict[str, Any]]:
    buffer = b""
    async for chunk in byte_stream:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                yield orjson.loads(payload)
            except orjson.JSONDecodeError:
                continue
    # Flush any tail line lacking a trailing newline.
    tail = buffer.strip()
    if tail.startswith(b"data:"):
        payload = tail[5:].strip()
        if payload and payload != b"[DONE]":
            try:
                yield orjson.loads(payload)
            except orjson.JSONDecodeError:
                pass


async def serialise_events(events: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[bytes]:
    async for obj in events:
        yield b"data: " + orjson.dumps(obj) + b"\n\n"
    yield b"data: [DONE]\n\n"


async def with_heartbeat(
    upstream: AsyncIterator[bytes], *,
    interval: float = 15.0,
    payload: str = "hb",
) -> AsyncIterator[bytes]:
    """Yield from ``upstream``; whenever it stays silent for ``interval``
    seconds, emit a SSE comment frame so idle proxies / clients don't drop
    the connection. Stops cleanly when ``upstream`` is exhausted."""
    if interval <= 0:
        async for chunk in upstream:
            yield chunk
        return

    heartbeat_bytes = f": {payload}\n\n".encode()
    iterator = upstream.__aiter__()

    async def _next() -> bytes | None:
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            return None

    pending: asyncio.Task[bytes | None] | None = asyncio.create_task(_next())
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
            except asyncio.TimeoutError:
                yield heartbeat_bytes
                continue
            if chunk is None:
                pending = None
                return
            yield chunk
            pending = asyncio.create_task(_next())
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, BaseException):
                pass


async def with_timeout(
    upstream: AsyncIterator[bytes],
    *,
    idle_timeout_s: float = 0.0,
    absolute_timeout_s: float = 0.0,
) -> AsyncIterator[bytes]:
    """Wrap a byte-stream iterator with idle and absolute timeouts.

    * ``idle_timeout_s`` — if no chunk arrives for this many seconds, abort.
    * ``absolute_timeout_s`` — if the total stream duration exceeds this, abort.

    Either value of 0 disables that timeout. When a timeout fires, raises
    ``StreamTimeoutError`` which flows through the existing error handling in
    ChatToResponsesStreamAdapter (it catches BaseException and synthesizes
    an incomplete completion event).
    """
    if idle_timeout_s <= 0 and absolute_timeout_s <= 0:
        async for chunk in upstream:
            yield chunk
        return

    start = time.monotonic()
    iterator = upstream.__aiter__()

    async def _next() -> bytes | None:
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            return None

    pending: asyncio.Task[bytes | None] = asyncio.create_task(_next())
    try:
        while True:
            # Compute the wait deadline: min of idle timeout and remaining
            # absolute timeout.
            wait_s: Optional[float] = None

            if idle_timeout_s > 0:
                wait_s = idle_timeout_s

            if absolute_timeout_s > 0:
                remaining = absolute_timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise StreamTimeoutError(
                        f"absolute timeout ({absolute_timeout_s}s) exceeded"
                    )
                if wait_s is None or remaining < wait_s:
                    wait_s = remaining

            try:
                chunk = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=wait_s
                )
            except asyncio.TimeoutError:
                # Determine which timeout actually expired
                if absolute_timeout_s > 0:
                    elapsed = time.monotonic() - start
                    if elapsed >= absolute_timeout_s - 0.1:
                        raise StreamTimeoutError(
                            f"absolute timeout ({absolute_timeout_s}s) exceeded"
                        )
                if idle_timeout_s > 0:
                    raise StreamTimeoutError(
                        f"idle timeout ({idle_timeout_s}s) exceeded"
                    )
                # Shouldn't reach here
                raise StreamTimeoutError("unknown timeout exceeded")

            if chunk is None:
                pending = None
                return
            yield chunk
            pending = asyncio.create_task(_next())
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, BaseException):
                pass
