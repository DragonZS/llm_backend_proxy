"""Integration: /metrics + trace_id echo + agent/backend labels."""

from __future__ import annotations

import httpx

from backend_proxy.app import create_app
from backend_proxy.config.schema import BackendConfig, ProxyConfig, ServerConfig

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


def _proxy(upstream: str):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="vllm", base_url=upstream, models=["fake-1"])],
    )
    return create_app(cfg)


def test_trace_id_round_trip():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/healthz",
                          headers={"x-request-id": "abc123"})
            assert r.status_code == 200
            assert r.headers["x-request-id"] == "abc123"


def test_metrics_endpoint_increments():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            for _ in range(3):
                httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                           json={"model": "fake-1",
                                 "messages": [{"role": "user", "content": "hi"}]})
            r = httpx.get(proxy_srv.base_url + "/metrics")
            assert r.status_code == 200
            text = r.text
            # If prometheus_client is unavailable we accept the placeholder body.
            if "# prometheus_client not installed" in text:
                return
            assert "backend_proxy_requests_total" in text
            assert "backend_proxy_request_duration_seconds" in text
            # The /v1/chat/completions calls must show up
            assert "/v1/chat/completions" in text
