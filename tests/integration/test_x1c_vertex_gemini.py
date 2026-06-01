"""Integration: VertexGeminiBackend behind the proxy. Translates OpenAI Chat
Completions to Gemini's generateContent / streamGenerateContent."""

from __future__ import annotations

import time

import httpx
import orjson
import pytest

from backend_proxy.app import create_app
from backend_proxy.config.schema import (
    AuthConfig,
    BackendConfig,
    ProxyConfig,
    ServerConfig,
)

from tests.conftest import serve
from tests.fake_vertex_gemini import make_fake_vertex_gemini


@pytest.fixture
def patched_token(monkeypatch):
    def _fake(self):
        return ("fake-vertex-token", time.time() + 3600)
    from backend_proxy.backends._gcp_oauth import GcpAccessTokenProvider
    monkeypatch.setattr(GcpAccessTokenProvider, "_refresh_blocking", _fake)
    return "fake-vertex-token"


def _proxy(upstream: str):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(
            name="gemini", type="vertex-gemini", base_url=upstream,
            models=["gemini-1.5-pro"],
            auth=AuthConfig(scheme="none", options={
                "project": "test-project", "location": "us-central1",
            }),
        )],
    )
    return create_app(cfg)


def test_gemini_chat_completions_non_streaming(patched_token):
    upstream = make_fake_vertex_gemini(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "gemini-1.5-pro",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 100},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["choices"][0]["message"]["content"] == "hello from gemini"
            assert body["choices"][0]["finish_reason"] == "stop"
            assert body["usage"]["total_tokens"] == 7


def test_gemini_chat_completions_streaming(patched_token):
    upstream = make_fake_vertex_gemini(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "gemini-1.5-pro", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            ) as r:
                assert r.status_code == 200
                events = []
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    events.append(orjson.loads(payload))
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert "hi" in text and "!" in text
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["prompt_tokens"] == 1


def test_gemini_models_list(patched_token):
    upstream = make_fake_vertex_gemini(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/v1/models")
            assert r.status_code == 200
            ids = [m["id"] for m in r.json()["data"]]
            assert ids == ["gemini-1.5-pro"]
