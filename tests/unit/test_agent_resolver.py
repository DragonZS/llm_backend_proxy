"""Unit tests for the agent resolver."""

from __future__ import annotations

from backend_proxy.agents.resolver import AgentResolver
from backend_proxy.config.schema import AgentMatcher, AgentProfileConfig


def _profile(**kw):
    kw.setdefault("name", "x")
    return AgentProfileConfig(**kw)


def test_path_prefix_match():
    profiles = [
        _profile(name="codex", match=AgentMatcher(path_prefix="/v1/responses")),
        _profile(name="claude", match=AgentMatcher(path_prefix="/v1/messages")),
        _profile(name="default", match=AgentMatcher(default=True)),
    ]
    r = AgentResolver(profiles, {})
    assert r.resolve(path="/v1/responses", headers={}, query_params={}).name == "codex"
    assert r.resolve(path="/v1/messages", headers={}, query_params={}).name == "claude"
    assert r.resolve(path="/v1/chat/completions", headers={}, query_params={}).name == "default"


def test_header_contains_match_case_insensitive():
    profiles = [
        _profile(name="codex",
                 match=AgentMatcher(header="user-agent", contains="codex")),
        _profile(name="default", match=AgentMatcher(default=True)),
    ]
    r = AgentResolver(profiles, {})
    hit = r.resolve(path="/v1/chat/completions",
                    headers={"User-Agent": "Codex/0.135"}, query_params={})
    assert hit.name == "codex"


def test_default_only_when_nothing_matches():
    profiles = [
        _profile(name="codex", match=AgentMatcher(path_prefix="/v1/responses")),
    ]
    r = AgentResolver(profiles, {})
    assert r.resolve(path="/v1/chat/completions",
                     headers={}, query_params={}) is None


def test_query_param_match():
    profiles = [
        _profile(name="x", match=AgentMatcher(query_param="agent")),
        _profile(name="default", match=AgentMatcher(default=True)),
    ]
    r = AgentResolver(profiles, {})
    assert r.resolve(path="/", headers={}, query_params={"agent": "x"}).name == "x"
    assert r.resolve(path="/", headers={}, query_params={}).name == "default"
