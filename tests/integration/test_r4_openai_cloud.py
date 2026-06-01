"""Integration: a 'cloud' backend (api.openai.com lookalike, served by
fake_vllm with bearer auth) coexists with a local vllm in the same pool.
model_affinity routes correctly; per-backend api_key isolation holds."""

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


def test_local_vllm_plus_openai_cloud_routing_and_isolation():
    # local vllm — no auth, claims qwen-coder
    local = make_fake_vllm(name="local", models=["qwen-coder"])
    # "cloud" — pretends to be api.openai.com, requires our OPENAI_API_KEY
    cloud = make_fake_vllm(name="cloud", models=["gpt-4o", "gpt-4o-mini"],
                           require_bearer="sk-CLOUD")

    with serve(local) as l, serve(cloud) as c:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="local", type="vllm",
                              base_url=l.base_url, models=["qwen-coder"]),
                BackendConfig(
                    name="openai", type="openai",
                    base_url=c.base_url,
                    models=["gpt-4o", "gpt-4o-mini"],
                    auth=AuthConfig(scheme="bearer", api_key="sk-CLOUD"),
                ),
            ],
            routing=RoutingConfig(strategy="model_affinity"),
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            # /v1/models aggregates both providers
            ids = sorted(m["id"] for m in
                         httpx.get(proxy_srv.base_url + "/v1/models").json()["data"])
            assert ids == ["gpt-4o", "gpt-4o-mini", "qwen-coder"]

            # Client sends NO Authorization. local backend doesn't need any;
            # cloud backend uses its own configured sk-CLOUD.
            local_resp = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "qwen-coder",
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            assert local_resp.status_code == 200, local_resp.text

            cloud_resp = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            assert cloud_resp.status_code == 200, cloud_resp.text


def test_openai_cloud_health_via_models_endpoint():
    """OpenAICloudBackend's DEFAULT_HEALTH_PATH is /v1/models — verify the
    readyz probe still works against a fake that requires auth."""
    cloud = make_fake_vllm(name="cloud", models=["x"], require_bearer="sk-K")
    with serve(cloud) as c:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[BackendConfig(
                name="openai", type="openai",
                base_url=c.base_url,
                auth=AuthConfig(scheme="bearer", api_key="sk-K"),
            )],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/readyz")
            assert r.status_code == 200, r.text
            assert r.json()["backends"]["openai"] is True
