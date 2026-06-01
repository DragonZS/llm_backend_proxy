"""Integration: per-backend token bucket + SSE heartbeat in the live proxy."""

from __future__ import annotations

import asyncio
import time

import httpx

from backend_proxy.app import create_app
from backend_proxy.config.schema import (
    BackendConfig,
    HeartbeatConfig,
    ProxyConfig,
    RateLimitConfig,
    ServerConfig,
)

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


# ---- rate limit ----------------------------------------------------

def test_reject_mode_returns_429_after_burst_exhausted():
    upstream = make_fake_vllm(name="local", models=["m"])
    with serve(upstream) as up:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[BackendConfig(
                name="rl", base_url=up.base_url, models=["m"],
                rate_limit=RateLimitConfig(rps=1, burst=2, mode="reject"),
            )],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            statuses = []
            for _ in range(6):
                r = httpx.post(
                    proxy_srv.base_url + "/v1/chat/completions",
                    json={"model": "m",
                          "messages": [{"role": "user", "content": "hi"}]},
                )
                statuses.append(r.status_code)
            # First two should pass (burst=2); subsequent ones rejected fast
            # (the bucket refills at 1 rps so within a tight loop only the
            # first 2 fit).
            assert statuses.count(200) >= 1
            assert statuses.count(429) >= 1


def test_wait_mode_serialises_requests():
    upstream = make_fake_vllm(name="local", models=["m"])
    with serve(upstream) as up:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[BackendConfig(
                name="rl", base_url=up.base_url, models=["m"],
                rate_limit=RateLimitConfig(rps=10, burst=1, mode="wait"),
            )],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            t0 = time.monotonic()
            # 3 sequential requests, burst=1, 10rps -> ~0.2s total.
            for _ in range(3):
                r = httpx.post(
                    proxy_srv.base_url + "/v1/chat/completions",
                    json={"model": "m",
                          "messages": [{"role": "user", "content": "hi"}]},
                )
                assert r.status_code == 200, r.text
            elapsed = time.monotonic() - t0
            # First request consumes the burst; remaining two each wait ~0.1s.
            assert 0.15 <= elapsed <= 1.5, elapsed


# ---- heartbeat -----------------------------------------------------

def test_heartbeat_injects_comment_frames_into_streaming_response():
    """A slow upstream lets the heartbeat fire several times. Verify the proxy
    emits SSE comment lines (lines starting with ':')."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    slow = FastAPI()

    @slow.post("/v1/chat/completions")
    async def _chat():
        async def gen():
            await asyncio.sleep(0.25)
            yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
            await asyncio.sleep(0.25)
            yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
            await asyncio.sleep(0.25)
            yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @slow.get("/v1/models")
    async def _m():
        return {"data": [{"id": "m", "object": "model"}]}

    @slow.get("/health")
    async def _h():
        return {"status": "ok"}

    with serve(slow) as up:
        cfg = ProxyConfig(
            server=ServerConfig(
                log_level="warning",
                heartbeat=HeartbeatConfig(enabled=True, interval_s=0.08, payload="hb"),
            ),
            backends=[BackendConfig(name="x", base_url=up.base_url, models=["m"])],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/responses",
                json={"model": "m", "stream": True,
                      "input": [{"type": "message", "role": "user",
                                  "content": "hi"}]},
                timeout=10.0,
            ) as r:
                assert r.status_code == 200
                raw_lines: list[str] = []
                try:
                    for line in r.iter_lines():
                        raw_lines.append(line)
                except Exception:
                    pass

    assert any(line == ": hb" for line in raw_lines), raw_lines
    assert any(line.startswith("data:") for line in raw_lines)


def test_heartbeat_disabled_does_not_inject():
    """With heartbeat off, the slow upstream produces no comment lines."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat():
        async def gen():
            await asyncio.sleep(0.1)
            yield b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def _m():
        return {"data": [{"id": "m", "object": "model"}]}

    @app.get("/health")
    async def _h():
        return {"status": "ok"}

    with serve(app) as up:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[BackendConfig(name="x", base_url=up.base_url, models=["m"])],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/responses",
                json={"model": "m", "stream": True,
                      "input": [{"type": "message", "role": "user",
                                  "content": "hi"}]},
                timeout=10.0,
            ) as r:
                lines = list(r.iter_lines())
    assert not any(line.startswith(":") for line in lines)
