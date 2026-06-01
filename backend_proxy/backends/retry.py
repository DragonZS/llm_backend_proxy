"""Retry logic for transient upstream failures.

Works with the pool's circuit breaker: each retry iteration re-picks a backend
and re-acquires the pool slot (semaphore + rate limit + CB recording).

Two main entry points:

* ``retry_request_json()`` — non-streaming requests with retry.
* ``retry_stream_start()`` — streaming requests with retry (only before the
  first byte is sent to the client; after that, mid-stream failures are
  handled by the stream adapter's BaseException catch which synthesises
  an incomplete completion event).
* ``retry_passthrough()`` — passthrough requests with retry.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from ..config.schema import RetryConfig
from ..core.errors import UpstreamError
from .base import Backend, PassthroughResponse
from .pool import BackendPool

log = logging.getLogger("backend_proxy.retry")

# Network-level exceptions that are always retryable.
_RETRYABLE_NETWORK_ERRORS = (httpx.ConnectError, httpx.TimeoutException)


def _is_retryable(
    exc: Exception,
    retryable_status_codes: list[int],
) -> bool:
    """Determine if an exception warrants a retry.

    Locally-generated errors (e.g. rate-limiter 429s) are NOT retryable —
    retrying them would just hit the same rate limit again. We identify these
    by checking ``UpstreamError.code == "rate_limited"``.
    """
    if isinstance(exc, _RETRYABLE_NETWORK_ERRORS):
        return True
    if isinstance(exc, UpstreamError):
        # Locally-generated rate-limit 429s are not retryable.
        if exc.code == "rate_limited":
            return False
        return exc.status_code in retryable_status_codes
    return False


def _compute_backoff(attempt: int, cfg: RetryConfig) -> float:
    """Exponential backoff with jitter: base * multiplier^attempt + jitter."""
    backoff = min(
        cfg.initial_backoff * (cfg.backoff_multiplier ** attempt),
        cfg.max_backoff,
    )
    # Add jitter: random 0-25% of backoff
    jitter = backoff * 0.25 * random.random()
    return backoff + jitter


def _effective_retry_config(
    pool_cfg: Optional[RetryConfig],
    backend_cfg: Optional[RetryConfig],
) -> RetryConfig:
    """Resolve per-backend override vs global default. Per-backend wins."""
    # If backend has its own retry config (non-None), prefer it.
    # BackendConfig.retry defaults to None, so routing-level config
    # is used unless the user explicitly sets retry on a backend.
    if backend_cfg is not None:
        return backend_cfg
    if pool_cfg is not None:
        return pool_cfg
    return RetryConfig()  # safe default


async def retry_request_json(
    pool: BackendPool,
    *,
    model: Optional[str],
    method: str,
    path: str,
    json: Dict[str, Any],
    headers: Dict[str, str],
    retry_cfg: RetryConfig,
) -> Tuple[Dict[str, Any], Backend]:
    """Non-streaming request with retry.

    Each attempt: pick() -> use() -> request_json(). On retryable failure,
    release the slot and try again (possibly a different backend).

    Returns (result_json, backend_that_succeeded).
    """
    last_exc: Optional[Exception] = None

    for attempt in range(retry_cfg.max_retries + 1):
        backend = pool.pick(model=model)
        effective_cfg = _effective_retry_config(
            pool_cfg=retry_cfg,
            backend_cfg=getattr(backend.cfg, "retry", None),
        )
        try:
            async with pool.use(backend):
                result = await backend.request_json(method, path, json=json, headers=headers)
            return result, backend
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, effective_cfg.retryable_status_codes):
                raise
            if attempt >= effective_cfg.max_retries:
                log.warning(
                    "[%s] retry exhausted after %d attempts: %s",
                    backend.name, attempt + 1, exc,
                )
                raise
            backoff = _compute_backoff(attempt, effective_cfg)
            log.info(
                "[%s] retryable error on attempt %d/%d, backoff %.1fs: %s",
                backend.name, attempt + 1, effective_cfg.max_retries + 1, backoff, exc,
            )
            await asyncio.sleep(backoff)

    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


async def retry_stream_start(
    pool: BackendPool,
    *,
    model: Optional[str],
    method: str,
    path: str,
    json: Dict[str, Any],
    headers: Dict[str, str],
    retry_cfg: RetryConfig,
) -> Tuple[AsyncIterator[bytes], Backend, Any]:
    """Streaming request with retry (only before first byte).

    Retry is only possible before any data has been sent to the client.
    Once the first chunk is received, we are committed and no further
    retries are attempted.

    Returns (upstream_bytes_iterator, backend, slot) — caller is responsible
    for releasing the slot via ``await slot.__aexit__(...)``.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(retry_cfg.max_retries + 1):
        backend = pool.pick(model=model)
        effective_cfg = _effective_retry_config(
            pool_cfg=retry_cfg,
            backend_cfg=getattr(backend.cfg, "retry", None),
        )
        slot = pool.use(backend)
        await slot.__aenter__()

        try:
            upstream_bytes = await backend.stream(method, path, json=json, headers=headers)

            # Probe: try to get the first chunk. If this fails with a
            # retryable error, we can still retry because nothing has been
            # sent to the client yet.
            first_chunk: Optional[bytes] = None
            try:
                first_chunk = await upstream_bytes.__anext__()
            except StopAsyncIteration:
                # Empty stream — rare, but not an error per se.
                pass

            # We got past the connection phase. Wrap the iterator to
            # yield the buffered first chunk, then continue.
            async def _replay_first(
                buf: Optional[bytes], it: AsyncIterator[bytes]
            ) -> AsyncIterator[bytes]:
                if buf is not None:
                    yield buf
                async for chunk in it:
                    yield chunk

            return _replay_first(first_chunk, upstream_bytes), backend, slot

        except Exception as exc:
            # Release the slot on failure
            await slot.__aexit__(type(exc), exc, exc.__traceback__)
            last_exc = exc
            if not _is_retryable(exc, effective_cfg.retryable_status_codes):
                raise
            if attempt >= effective_cfg.max_retries:
                log.warning(
                    "[%s] stream retry exhausted after %d attempts: %s",
                    backend.name, attempt + 1, exc,
                )
                raise
            backoff = _compute_backoff(attempt, effective_cfg)
            log.info(
                "[%s] stream retryable error on attempt %d/%d, backoff %.1fs: %s",
                backend.name, attempt + 1, effective_cfg.max_retries + 1, backoff, exc,
            )
            await asyncio.sleep(backoff)

    raise last_exc  # type: ignore[misc]


async def retry_passthrough(
    pool: BackendPool,
    *,
    model: Optional[str],
    method: str,
    path: str,
    body: Optional[bytes],
    headers: Dict[str, str],
    retry_cfg: RetryConfig,
) -> Tuple[PassthroughResponse, Backend]:
    """Passthrough request with retry. Same logic as retry_request_json()
    but calls backend.passthrough() instead of backend.request_json()."""
    last_exc: Optional[Exception] = None
    for attempt in range(retry_cfg.max_retries + 1):
        backend = pool.pick(model=model)
        effective_cfg = _effective_retry_config(
            pool_cfg=retry_cfg,
            backend_cfg=getattr(backend.cfg, "retry", None),
        )
        try:
            async with pool.use(backend):
                result = await backend.passthrough(method, path, body=body, headers=headers)
            return result, backend
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc, effective_cfg.retryable_status_codes):
                raise
            if attempt >= effective_cfg.max_retries:
                raise
            backoff = _compute_backoff(attempt, effective_cfg)
            await asyncio.sleep(backoff)
    raise last_exc  # type: ignore[misc]
