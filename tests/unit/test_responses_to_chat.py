"""Unit tests for Responses API -> Chat Completions request mapping."""

from __future__ import annotations

import pytest

from backend_proxy.adapters.responses_to_chat import ResponsesToChatAdapter
from backend_proxy.core.context import RequestContext
from backend_proxy.core.errors import BadRequestError


def _ctx() -> RequestContext:
    return RequestContext(trace_id="test", started_at=0.0, method="POST", path="/v1/responses")


async def _transform(payload):
    return await ResponsesToChatAdapter().transform(payload, _ctx())


async def test_maps_common_generation_parameters():
    out = await _transform({
        "model": "m",
        "input": "hello",
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["STOP"],
        "seed": 123,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "parallel_tool_calls": False,
        "max_output_tokens": 42,
    })

    assert out["model"] == "m"
    assert out["messages"] == [{"role": "user", "content": "hello"}]
    assert out["stream"] is True
    assert out["temperature"] == 0.2
    assert out["top_p"] == 0.9
    assert out["stop"] == ["STOP"]
    assert out["seed"] == 123
    assert out["presence_penalty"] == 0.1
    assert out["frequency_penalty"] == 0.2
    assert out["parallel_tool_calls"] is False
    assert out["max_tokens"] == 42


async def test_maps_reasoning_effort():
    out = await _transform({
        "model": "m",
        "input": "hello",
        "reasoning": {"effort": "high"},
    })
    assert out["reasoning_effort"] == "high"


async def test_maps_text_format_to_response_format():
    fmt = {
        "type": "json_schema",
        "name": "answer",
        "schema": {"type": "object"},
    }
    out = await _transform({
        "model": "m",
        "input": "hello",
        "text": {"format": fmt},
    })
    assert out["response_format"] == fmt


async def test_response_format_takes_precedence_over_text_format():
    text_fmt = {"type": "json_schema", "name": "from_text", "schema": {"type": "object"}}
    response_fmt = {"type": "json_object"}
    out = await _transform({
        "model": "m",
        "input": "hello",
        "text": {"format": text_fmt},
        "response_format": response_fmt,
    })
    assert out["response_format"] == response_fmt


async def test_text_format_text_is_ignored():
    out = await _transform({
        "model": "m",
        "input": "hello",
        "text": {"format": {"type": "text"}},
    })
    assert "response_format" not in out


async def test_maps_function_tools_with_strict():
    out = await _transform({
        "model": "m",
        "input": "hello",
        "tools": [{
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a patch",
            "parameters": {"type": "object"},
            "strict": True,
        }],
    })
    assert out["tools"] == [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch",
            "parameters": {"type": "object"},
            "strict": True,
        },
    }]


async def test_unsupported_tool_type_is_dropped_silently():
    # Real-world bug: Codex 0.135+ injects internal tool types
    # (`namespace`, `web_search`, `local_shell`, ...) into every request.
    # We must NOT 400 — the OpenAI Responses API silently ignores tools the
    # backend doesn't implement. Raising would brick every Codex turn.
    out = await _transform({
        "model": "m",
        "input": "hello",
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "name": "apply_patch",
             "parameters": {"type": "object"}},
            {"type": "namespace", "name": "internal"},
        ],
    })
    # Only the function tool survives.
    assert out.get("tools") == [{
        "type": "function",
        "function": {"name": "apply_patch", "parameters": {"type": "object"}},
    }]


async def test_only_unsupported_tools_yields_no_tools_field():
    # If every tool is unsupported, the request still goes through (no 400)
    # — the backend just sees no `tools` array at all.
    out = await _transform({
        "model": "m",
        "input": "hello",
        "tools": [{"type": "web_search"}, {"type": "namespace"}],
    })
    assert "tools" not in out


async def test_unsupported_tool_choice_type_raises_bad_request():
    with pytest.raises(BadRequestError, match="unsupported Responses tool_choice"):
        await _transform({
            "model": "m",
            "input": "hello",
            "tool_choice": {"type": "mcp", "server_label": "docs"},
        })


async def test_function_tool_choice_passes_through():
    choice = {"type": "function", "function": {"name": "shell"}}
    out = await _transform({
        "model": "m",
        "input": "hello",
        "tool_choice": choice,
    })
    assert out["tool_choice"] == choice


async def test_message_parts_still_flatten_text_only():
    out = await _transform({
        "model": "m",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"},
                {"type": "input_image", "image_url": "ignored"},
            ],
        }],
    })
    assert out["messages"] == [{"role": "user", "content": "ab"}]


# ---- Codex tool hint injection ----------------------------------------

async def test_codex_hint_injected_when_apply_patch_missing():
    """When the forwarded tool list looks like Codex (has exec_command,
    write_stdin, no apply_patch), the proxy must inject a system-level
    reminder so Qwen-style models don't try to call apply_patch or use
    write_stdin for file writes — both of which silently abort the agent."""
    out = await _transform({
        "model": "Qwen/Qwen3-27B",
        "input": [{"role": "user", "content": "write a hello.py"}],
        "tools": [
            {"type": "function", "name": "exec_command",
             "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": "write_stdin",
             "parameters": {"type": "object", "properties": {}}},
        ],
    })
    sys_msg = out["messages"][0]
    assert sys_msg["role"] == "system"
    body = sys_msg["content"]
    assert "apply_patch" in body
    assert "write_stdin" in body
    assert "heredoc" in body.lower()


async def test_codex_hint_NOT_injected_when_apply_patch_present():
    """If the forwarded tool list already contains apply_patch (e.g. an
    OpenAI-grade Codex setup), the hint must NOT be injected."""
    out = await _transform({
        "model": "Qwen/Qwen3-27B",
        "input": [{"role": "user", "content": "write hello.py"}],
        "tools": [
            {"type": "function", "name": "exec_command",
             "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": "apply_patch",
             "parameters": {"type": "object", "properties": {}}},
        ],
    })
    # No leading system message at all (input had only user content)
    assert out["messages"][0]["role"] == "user"


async def test_codex_hint_NOT_injected_for_unrelated_tool_set():
    """Anthropic-style or random tool sets (no exec_command) should not
    trigger the Codex-specific hint."""
    out = await _transform({
        "model": "Qwen/Qwen3-27B",
        "input": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "name": "get_weather",
                   "parameters": {"type": "object", "properties": {}}}],
    })
    assert out["messages"][0]["role"] == "user"
