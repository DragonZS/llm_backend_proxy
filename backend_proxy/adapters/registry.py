"""Adapter registry — name -> class. Used by the agent profile loader to
build pipelines from YAML."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from ..config.schema import AdapterConfig, AgentProfileConfig
from .anthropic import (
    AnthropicToChatAdapter,
    ChatToAnthropicAdapter,
    ChatToAnthropicStreamAdapter,
)
from .base import RequestAdapter, ResponseAdapter, StreamAdapter
from .chat_to_responses import ChatToResponsesAdapter, ChatToResponsesStreamAdapter
from .codex_max_tokens import CodexMaxTokensAdapter
from .model_alias import ModelAliasAdapter
from .pipeline import AdapterPipeline
from .responses_to_chat import ResponsesToChatAdapter

REQUEST_ADAPTERS: Dict[str, Type[RequestAdapter]] = {
    "responses_to_chat": ResponsesToChatAdapter,
    "anthropic_to_chat": AnthropicToChatAdapter,
    "codex_max_tokens":  CodexMaxTokensAdapter,
    "model_alias":       ModelAliasAdapter,
}

STREAM_ADAPTERS: Dict[str, Type[StreamAdapter]] = {
    "chat_to_responses_stream": ChatToResponsesStreamAdapter,
    "chat_to_anthropic_stream": ChatToAnthropicStreamAdapter,
}

RESPONSE_ADAPTERS: Dict[str, Type[ResponseAdapter]] = {
    "chat_to_responses":  ChatToResponsesAdapter,
    "chat_to_anthropic":  ChatToAnthropicAdapter,
}


def _opts_for(name: str, adapter_cfg: Dict[str, AdapterConfig], extra: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    cfg = adapter_cfg.get(name)
    if cfg and cfg.enabled:
        base.update(cfg.options)
    base.update(extra)
    return base


def build_pipeline(
    profile: AgentProfileConfig,
    adapter_cfg: Dict[str, AdapterConfig],
) -> AdapterPipeline:
    """Compile a pipeline from an agent profile + global adapter options."""

    def _build_one(name: str, table: Dict[str, Type], extra: Dict[str, Any] | None = None):
        cls = table.get(name)
        if cls is None:
            raise ValueError(f"unknown adapter: {name}")
        cfg = adapter_cfg.get(name)
        if cfg is not None and not cfg.enabled:
            return None
        return cls(_opts_for(name, adapter_cfg, extra or {}))

    request: List[RequestAdapter] = []
    for n in profile.request_adapters:
        extra: Dict[str, Any] = {}
        if n == "model_alias":
            extra["aliases"] = profile.model_aliases
        a = _build_one(n, REQUEST_ADAPTERS, extra)
        if a is not None:
            request.append(a)

    stream: List[StreamAdapter] = []
    for n in profile.stream_adapters:
        a = _build_one(n, STREAM_ADAPTERS)
        if a is not None:
            stream.append(a)

    response: List[ResponseAdapter] = []
    for n in profile.response_adapters:
        a = _build_one(n, RESPONSE_ADAPTERS)
        if a is not None:
            response.append(a)

    return AdapterPipeline(request=request, stream=stream, response=response)
