"""End-to-end M0 smoke: boot a fake vLLM, point the proxy at it, hit endpoints."""

from __future__ import annotations

import httpx
import pytest

from backend_proxy.app import create_app
from backend_proxy.config.schema import BackendConfig, ProxyConfig, ServerConfig

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


@pytest.fixture
def fake_upstream():
    app = make_fake_vllm(name="local", models=["fake-1"])
    with serve(app) as s:
        yield s


def test_healthz_and_readyz(fake_upstream):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="local-vllm", base_url=fake_upstream.base_url)],
    )
    proxy = create_app(cfg)
    with serve(proxy) as proxy_srv:
        r = httpx.get(proxy_srv.base_url + "/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        r = httpx.get(proxy_srv.base_url + "/readyz")
        assert r.status_code == 200
        assert r.json()["backends"] == {"local-vllm": True}


def test_models_aggregation(fake_upstream):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="local-vllm", base_url=fake_upstream.base_url)],
    )
    proxy = create_app(cfg)
    with serve(proxy) as proxy_srv:
        r = httpx.get(proxy_srv.base_url + "/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert ids == ["fake-1"]


def test_passthrough_chat_completions(fake_upstream):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="local-vllm", base_url=fake_upstream.base_url)],
    )
    proxy = create_app(cfg)
    with serve(proxy) as proxy_srv:
        r = httpx.post(
            proxy_srv.base_url + "/v1/chat/completions",
            json={"model": "fake-1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "hello from fake"
