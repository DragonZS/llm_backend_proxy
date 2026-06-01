"""Unit tests for the Anthropic Messages adapters."""

from __future__ import annotations

import pytest

from backend_proxy.adapters.anthropic import (
    AnthropicToChatAdapter,
    ChatToAnthropicAdapter,
    ChatToAnthropicStreamAdapter,
)
from backend_proxy.core.context import RequestContext


def _ctx() -> RequestContext:
    return RequestContext.new(method="POST", path="/v1/messages", headers={})


# ---- request rewrite -------------------------------------------------

async def test_string_system_becomes_system_message():
    adapter = AnthropicToChatAdapter()
    out = await adapter.transform({
        "model": "claude-sonnet-4",
        "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }, _ctx())
    assert out["model"] == "claude-sonnet-4"
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 100


async def test_typed_system_concatenated():
    adapter = AnthropicToChatAdapter()
    out = await adapter.transform({
        "model": "m",
        "system": [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}],
        "messages": [{"role": "user", "content": "hi"}],
    }, _ctx())
    assert out["messages"][0]["content"] == "a b"


async def test_typed_user_content_flattened():
    out = await AnthropicToChatAdapter().transform({
        "model": "m",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "x "},
                {"type": "text", "text": "y"},
            ],
        }],
    }, _ctx())
    assert out["messages"] == [{"role": "user", "content": "x y"}]


async def test_tool_use_and_tool_result_round_trip():
    out = await AnthropicToChatAdapter().transform({
        "model": "m",
        "messages": [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling..."},
                {"type": "tool_use", "id": "call_1", "name": "f", "input": {"a": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "42"},
            ]},
        ],
    }, _ctx())
    # User text  /  assistant w/ tool_calls  /  role:tool with tool_call_id
    assert out["messages"][0] == {"role": "user", "content": "do X"}
    assert out["messages"][1]["role"] == "assistant"
    assert out["messages"][1]["content"] == "calling..."
    assert out["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "f"
    assert '"a":1' in out["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert out["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


async def test_tools_translated():
    out = await AnthropicToChatAdapter().transform({
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "search", "description": "do search",
                   "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "search"},
    }, _ctx())
    assert out["tools"][0] == {
        "type": "function",
        "function": {"name": "search", "description": "do search",
                     "parameters": {"type": "object"}},
    }
    assert out["tool_choice"] == {"type": "function", "function": {"name": "search"}}


# ---- response wrap (full) -------------------------------------------

async def test_chat_to_anthropic_text_only():
    chat = {
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "answer"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    out = await ChatToAnthropicAdapter().transform(chat, _ctx())
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "answer"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 2}


async def test_chat_to_anthropic_tool_use():
    chat = {
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    out = await ChatToAnthropicAdapter().transform(chat, _ctx())
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "f", "input": {"a": 1},
    }


# ---- stream ----------------------------------------------------------

async def _gen(items):
    for it in items:
        yield it


async def test_anthropic_stream_event_sequence():
    chunks = [
        {"model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]},
        {"model": "m", "choices": [{"index": 0, "delta": {"content": "hi"},
                                    "finish_reason": None}]},
        {"model": "m", "choices": [{"index": 0, "delta": {"content": "!"},
                                    "finish_reason": None}]},
        {"model": "m", "choices": [{"index": 0, "delta": {},
                                    "finish_reason": "stop"}],
         "usage": {"completion_tokens": 2}},
    ]
    events = []
    async for ev in ChatToAnthropicStreamAdapter().transform(_gen(chunks), _ctx()):
        events.append(ev)

    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    # We only emit content_block_start once we see actual content.
    cbs_idx = types.index("content_block_start")
    assert cbs_idx > 0
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in deltas) == "hi!"
    assert "content_block_stop" in types
    md = [e for e in events if e["type"] == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["output_tokens"] == 2
    assert types[-1] == "message_stop"
