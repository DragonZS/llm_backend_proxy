"""Integration: VertexClaudeBackend behind the proxy. We monkey-patch the
GcpAccessTokenProvider so tests don't need real GCP credentials."""

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
from tests.fake_vertex import make_fake_vertex_anthropic


@pytest.fixture
def patched_token(monkeypatch):
    """Replace the OAuth refresh path with a stub that returns a fixed token.
    Tests expect the proxy to send ``Authorization: Bearer fake-vertex-token``
    on every Vertex request."""
    def _fake(self):
        return ("fake-vertex-token", time.time() + 3600)
    from backend_proxy.backends._gcp_oauth import GcpAccessTokenProvider
    monkeypatch.setattr(GcpAccessTokenProvider, "_refresh_blocking", _fake)
    return "fake-vertex-token"


def _proxy(upstream_url: str, *, project="test-project", location="us-central1"):
    cfg = ProxyConfig(
        server=ServerConfig(log_level="warning"),
        backends=[BackendConfig(
            name="vertex", type="vertex-claude", base_url=upstream_url,
            models=["claude-3-5-sonnet@20240620"],
            auth=AuthConfig(scheme="none", options={
                "project": project, "location": location,
            }),
        )],
    )
    return create_app(cfg)


def test_vertex_claude_messages_passthrough(patched_token):
    upstream = make_fake_vertex_anthropic(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/messages",
                json={"model": "claude-3-5-sonnet@20240620",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 50},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["content"][0]["text"] == "hello from vertex-claude"
            assert body["stop_reason"] == "end_turn"


def test_vertex_claude_chat_completions_translated(patched_token):
    upstream = make_fake_vertex_anthropic(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            r = httpx.post(
                proxy_srv.base_url + "/v1/chat/completions",
                json={"model": "claude-3-5-sonnet@20240620",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 50},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["choices"][0]["message"]["content"] == "hello from vertex-claude"
            assert body["choices"][0]["finish_reason"] == "stop"


def test_vertex_claude_streaming(patched_token):
    upstream = make_fake_vertex_anthropic(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            with httpx.stream(
                "POST",
                proxy_srv.base_url + "/v1/messages",
                json={"model": "claude-3-5-sonnet@20240620", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 50},
                timeout=10.0,
            ) as r:
                assert r.status_code == 200
                # Collect events; the upstream is Anthropic-shape so output
                # should preserve message_start..message_stop.
                events = []
                cur = None
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        cur = line[6:].strip()
                    elif line.startswith("data:"):
                        events.append((cur, orjson.loads(line[5:].strip())))
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert "content_block_delta" in types
    assert types[-1] == "message_stop"


def test_vertex_claude_health(patched_token):
    """health() returns True if we can mint an access token."""
    upstream = make_fake_vertex_anthropic(require_bearer=patched_token)
    with serve(upstream) as up:
        with serve(_proxy(up.base_url)) as proxy_srv:
            r = httpx.get(proxy_srv.base_url + "/readyz")
            assert r.json()["backends"]["vertex"] is True


def test_vertex_requires_project(patched_token):
    """Missing project should raise at backend construction time."""
    from backend_proxy.backends import make_backend
    cfg = BackendConfig(
        name="vertex", type="vertex-claude", base_url="http://example",
        # auth.options.project intentionally missing
    )
    with pytest.raises(Exception):
        make_backend(cfg)
