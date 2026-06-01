"""Integration test: /v1/messages translated end-to-end through the proxy."""

from __future__ import annotations

import httpx
import orjson

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


def test_messages_non_stream_round_trip():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/messages",
                json={
                    "model": "fake-1",
                    "system": "be brief",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 50,
                },
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["type"] == "message"
            assert body["role"] == "assistant"
            assert body["content"][0]["text"] == "hello from fake"
            assert body["stop_reason"] == "end_turn"


def test_messages_streaming_emits_anthropic_event_envelope():
    upstream = make_fake_vllm(name="local", models=["fake-1"])
    with serve(upstream) as up:
        proxy = _proxy(up.base_url)
        with serve(proxy) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/messages",
                json={"model": "fake-1", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 50},
                timeout=10.0,
            ) as r:
                assert r.status_code == 200, r.text
                events = []
                current_event_type = None
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        current_event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload:
                            events.append((current_event_type, orjson.loads(payload)))

    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert types[-1] == "message_stop"

    text = "".join(
        e["delta"]["text"]
        for t, e in events
        if t == "content_block_delta"
    )
    assert "hi" in text and "!" in text
