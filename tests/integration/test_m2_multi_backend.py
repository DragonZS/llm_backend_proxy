"""Integration: two fake vLLM upstreams + model aggregation + failover."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from backend_proxy.app import create_app
from backend_proxy.config.schema import (
    BackendConfig,
    ProxyConfig,
    RoutingConfig,
    ServerConfig,
)

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


def test_models_aggregation_across_two_backends():
    a = make_fake_vllm(name="a", models=["a-1", "common"])
    b = make_fake_vllm(name="b", models=["b-1", "common"])
    with serve(a) as sa, serve(b) as sb:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url, models=["a-1", "common"]),
                BackendConfig(name="B", base_url=sb.base_url, models=["b-1", "common"]),
            ],
            routing=RoutingConfig(strategy="model_affinity", fallback="round_robin"),
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/v1/models")
            ids = sorted(m["id"] for m in r.json()["data"])
            # "common" deduped across backends
            assert ids == ["a-1", "b-1", "common"]


def test_model_affinity_routes_to_correct_backend():
    """Backend A claims model a-1; the fake responds with content "from-a";
    Backend B responds with "from-b". A request asking for a-1 must hit A."""
    a = make_fake_vllm(
        name="a", models=["a-1"],
        chat_handler=lambda req: {
            "id": "x", "object": "chat.completion", "model": req["model"],
            "created": 0, "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "from-a"},
                "finish_reason": "stop",
            }],
        },
    )
    b = make_fake_vllm(
        name="b", models=["b-1"],
        chat_handler=lambda req: {
            "id": "x", "object": "chat.completion", "model": req["model"],
            "created": 0, "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "from-b"},
                "finish_reason": "stop",
            }],
        },
    )
    with serve(a) as sa, serve(b) as sb:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url, models=["a-1"]),
                BackendConfig(name="B", base_url=sb.base_url, models=["b-1"]),
            ],
            routing=RoutingConfig(strategy="model_affinity"),
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            ra = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "a-1", "messages": [{"role": "user", "content": "hi"}]})
            rb = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "b-1", "messages": [{"role": "user", "content": "hi"}]})
            assert ra.json()["choices"][0]["message"]["content"] == "from-a"
            assert rb.json()["choices"][0]["message"]["content"] == "from-b"


def test_failover_when_one_backend_is_down():
    """Configure two backends; only one is actually serving. A request that
    doesn't request a specific model must reach the live one (fallback RR
    over healthy candidates only)."""
    live = make_fake_vllm(name="live", models=["m"])
    with serve(live) as live_srv:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="dead", base_url="http://127.0.0.1:1",   # nothing listens
                              models=["m"], timeout=1.0, health_path="/health"),
                BackendConfig(name="live", base_url=live_srv.base_url,
                              models=["m"]),
            ],
            routing=RoutingConfig(strategy="round_robin"),
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            # Wait briefly so the initial health probe marks 'dead' unhealthy.
            import time
            time.sleep(0.5)
            for _ in range(4):
                r = httpx.post(
                    proxy_srv.base_url + "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    timeout=5.0,
                )
                assert r.status_code == 200, r.text
                assert r.json()["choices"][0]["message"]["content"] == "hello from fake"
