"""End-to-end test for /v1/responses against a fake vLLM, both streaming and
non-streaming paths."""

from __future__ import annotations

import httpx
import pytest

from backend_proxy.app import create_app
from backend_proxy.config.schema import BackendConfig, ProxyConfig, ServerConfig

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


def _proxy_app(upstream_url: str):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(name="local-vllm", base_url=upstream_url,
                                models=["fake-1"])],
    )
    return create_app(cfg)


def test_responses_non_streaming_wraps_chat_response():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy_app(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/responses",
                json={"model": "fake-1",
                      "input": [{"type": "message", "role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["object"] == "response"
            assert body["status"] == "completed"
            assert body["output"][0]["content"][0]["text"] == "hello from fake"


def test_responses_streaming_emits_responses_events():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy_app(up.base_url)
        with serve(proxy) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/responses",
                json={"model": "fake-1", "stream": True,
                      "input": [{"type": "message", "role": "user", "content": "hi"}]},
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
                    import orjson
                    events.append(orjson.loads(payload))

    types = [e.get("type") for e in events]
    # Required prefix sequence (order matters)
    assert types[0] == "response.created"
    assert types[1] == "response.output_item.added"
    assert types[2] == "response.content_part.added"
    # Some output_text.delta events follow
    assert "response.output_text.delta" in types
    # Tail
    assert types[-1] == "response.completed"
    assert types[-2] == "response.output_item.done"
    # Final accumulated text
    completed = [e for e in events if e["type"] == "response.completed"][0]
    text = completed["response"]["output"][0]["content"][0]["text"]
    assert "hi!" in text  # default fake stream chunks emit "hi" then "!"
