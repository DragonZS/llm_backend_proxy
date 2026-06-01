"""Integration: two backends with different per-backend keys, both reachable
through the proxy without the client knowing either key."""

from __future__ import annotations

import httpx

from backend_proxy.app import create_app
from backend_proxy.config.schema import (
    AuthConfig,
    BackendConfig,
    ProxyConfig,
    RoutingConfig,
    ServerConfig,
)

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


def test_two_backends_isolate_their_own_api_keys():
    a = make_fake_vllm(name="A", models=["a-model"], require_bearer="key-A")
    b = make_fake_vllm(name="B", models=["b-model"], require_bearer="key-B")
    with serve(a) as sa, serve(b) as sb:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url,
                              models=["a-model"],
                              auth=AuthConfig(scheme="bearer", api_key="key-A")),
                BackendConfig(name="B", base_url=sb.base_url,
                              models=["b-model"],
                              auth=AuthConfig(scheme="bearer", api_key="key-B")),
            ],
            routing=RoutingConfig(strategy="model_affinity"),
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            # Client carries no auth header at all — proxy still reaches both
            ra = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "a-model",
                                  "messages": [{"role": "user", "content": "hi"}]})
            assert ra.status_code == 200, ra.text
            rb = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "b-model",
                                  "messages": [{"role": "user", "content": "hi"}]})
            assert rb.status_code == 200, rb.text


def test_client_authorization_does_not_leak_to_other_backend():
    """Client sends a token usable on backend A; the request actually routes
    to backend B (different key). B must accept because the proxy substitutes
    B's own configured token; the proxy never forwards the client's."""
    a_only_key = "key-A"
    b = make_fake_vllm(name="B", models=["b-model"], require_bearer="key-B")
    with serve(b) as sb:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="B", base_url=sb.base_url,
                              models=["b-model"],
                              auth=AuthConfig(scheme="bearer", api_key="key-B")),
            ],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {a_only_key}"},
                json={"model": "b-model",
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text


def test_passthrough_scheme_actually_forwards_client_auth():
    a = make_fake_vllm(name="A", models=["m"], require_bearer="from-client")
    with serve(a) as sa:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url, models=["m"],
                              auth=AuthConfig(scheme="passthrough")),
            ],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            ok = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                headers={"Authorization": "Bearer from-client"},
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert ok.status_code == 200, ok.text
            # Without the client header → backend rejects (no key)
            bad = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert bad.status_code == 401


def test_x_api_key_scheme():
    a = make_fake_vllm(name="A", models=["m"],
                       require_header=("x-api-key", "anth-key"))
    with serve(a) as sa:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url, models=["m"],
                              auth=AuthConfig(scheme="x_api_key", api_key="anth-key")),
            ],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text


def test_api_key_header_custom_name():
    a = make_fake_vllm(name="A", models=["m"],
                       require_header=("x-tgi-key", "tgi-1"))
    with serve(a) as sa:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="A", base_url=sa.base_url, models=["m"],
                              auth=AuthConfig(scheme="api_key_header",
                                              api_key="tgi-1",
                                              header_name="x-tgi-key")),
            ],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text
