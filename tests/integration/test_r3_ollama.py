"""Integration: native Ollama backend reachable via OpenAI-shaped API on the
proxy. Client speaks OpenAI; the backend speaks Ollama; nobody knows."""

from __future__ import annotations

import httpx
import orjson

from backend_proxy.app import create_app
from backend_proxy.config.schema import BackendConfig, ProxyConfig, ServerConfig

from tests.conftest import serve
from tests.fake_ollama import make_fake_ollama


def _proxy(upstream: str):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="ollama", type="ollama",
                                base_url=upstream,
                                models=["llama3.1:8b"])],
    )
    return create_app(cfg)


def test_ollama_models_aggregated_through_openai_endpoint():
    upstream = make_fake_ollama(models=[
        {"name": "llama3.1:8b", "size": 1, "digest": "x", "modified_at": "t"},
        {"name": "qwen:7b", "size": 2, "digest": "y", "modified_at": "t"},
    ])
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/v1/models")
            assert r.status_code == 200, r.text
            ids = sorted(m["id"] for m in r.json()["data"])
            assert ids == ["llama3.1:8b", "qwen:7b"]


def test_ollama_chat_non_streaming():
    upstream = make_fake_ollama()
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "llama3.1:8b",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 10},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"]["content"] == "hello from ollama"
            assert body["choices"][0]["finish_reason"] == "stop"
            assert body["usage"]["prompt_tokens"] == 5
            assert body["usage"]["completion_tokens"] == 4


def test_ollama_chat_streaming_emits_openai_sse():
    upstream = make_fake_ollama()
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "llama3.1:8b", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            ) as r:
                assert r.status_code == 200, r.text
                events = []
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    events.append(orjson.loads(payload))

    # First chunk: role=assistant
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert "hi!" in text
    # Last chunk: finish_reason
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert "usage" in events[-1]


def test_ollama_via_responses_endpoint():
    """Codex-style: client hits /v1/responses on the proxy, proxy translates
    to OpenAI chat, OllamaNativeBackend translates that to /api/chat."""
    upstream = make_fake_ollama()
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/responses",
                json={"model": "llama3.1:8b",
                      "input": [{"type": "message", "role": "user",
                                  "content": "hi"}]},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["object"] == "response"
            assert body["output"][0]["content"][0]["text"] == "hello from ollama"


def test_mixed_pool_vllm_and_ollama_route_by_model():
    """vllm + ollama in the same pool; model_affinity routes correctly."""
    from tests.fake_vllm import make_fake_vllm
    vllm_up = make_fake_vllm(name="vllm", models=["qwen-coder"])
    ollama_up = make_fake_ollama(models=[
        {"name": "llama3.1:8b", "size": 0, "digest": "x", "modified_at": "t"},
    ])
    with serve(vllm_up) as v, serve(ollama_up) as o:
        cfg = ProxyConfig(
            server=ServerConfig(log_level="warning"),
            backends=[
                BackendConfig(name="vllm", type="vllm", base_url=v.base_url,
                              models=["qwen-coder"]),
                BackendConfig(name="ollama", type="ollama", base_url=o.base_url,
                              models=["llama3.1:8b"]),
            ],
        )
        proxy = create_app(cfg)
        with serve(proxy) as proxy_srv:
            r1 = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "qwen-coder",
                                  "messages": [{"role": "user", "content": "hi"}]})
            assert r1.json()["choices"][0]["message"]["content"] == "hello from fake"

            r2 = httpx.post(proxy_srv.base_url + "/v1/chat/completions",
                            json={"model": "llama3.1:8b",
                                  "messages": [{"role": "user", "content": "hi"}]})
            assert r2.json()["choices"][0]["message"]["content"] == "hello from ollama"
