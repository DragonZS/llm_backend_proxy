"""Built-in AgentProfile presets — used when no `agents:` section appears in
the user's YAML. Replicates the behaviour of legacy ``proxy_v1`` for Codex,
adds Claude Code routing for ``/v1/messages``, and a default OpenCode-style
pass-through.
"""

from __future__ import annotations

from typing import List

from ..config.schema import AdapterConfig, AgentMatcher, AgentProfileConfig


def builtin_agents() -> List[AgentProfileConfig]:
    return [
        AgentProfileConfig(
            name="codex",
            match=AgentMatcher(path_prefix="/v1/responses"),
            request_adapters=["responses_to_chat", "codex_max_tokens", "model_alias"],
            stream_adapters=["chat_to_responses_stream"],
            response_adapters=["chat_to_responses"],
        ),
        AgentProfileConfig(
            name="claude-code",
            match=AgentMatcher(path_prefix="/v1/messages"),
            request_adapters=["anthropic_to_chat", "model_alias"],
            stream_adapters=["chat_to_anthropic_stream"],
            response_adapters=["chat_to_anthropic"],
        ),
        AgentProfileConfig(
            name="opencode",
            match=AgentMatcher(default=True),
            request_adapters=[],
            stream_adapters=[],
            response_adapters=[],
        ),
    ]


def builtin_adapter_config() -> dict[str, AdapterConfig]:
    return {
        # 0 = let the upstream decide max_tokens (vLLM uses
        # max_model_len - prompt_tokens automatically). Override in YAML if
        # you want a hard cap. See adapters/codex_max_tokens.py.
        "codex_max_tokens": AdapterConfig(enabled=True, options={"fallback_max_tokens": 0}),
    }
