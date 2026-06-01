"""Unit tests for chat ↔ Anthropic adapters (the reverse direction added
in X1a)."""

from __future__ import annotations

import orjson

from backend_proxy.adapters.anthropic import (
    anthropic_event_to_chat_chunks,
    anthropic_to_chat_response,
    chat_to_anthropic_request,
)


# ---- chat_to_anthropic_request --------------------------------------

def test_system_messages_collapsed_to_top_level():
    out = chat_to_anthropic_request({
        "model": "claude",
        "messages": [
            {"role": "system", "content": "rule A"},
            {"role": "system", "content": "rule B"},
            {"role": "user", "content": "hi"},
        ],
    }, stream=False)
    assert out["system"] == "rule A\n\nrule B"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_max_tokens_required_falls_back():
    out = chat_to_anthropic_request({
        "model": "c", "messages": [{"role": "user", "content": "x"}],
    }, stream=False, default_max_tokens=512)
    assert out["max_tokens"] == 512


def test_assistant_tool_calls_become_tool_use_blocks():
    out = chat_to_anthropic_request({
        "model": "c",
        "messages": [
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "calling",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {"name": "f", "arguments": '{"a": 1}'},
             }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ],
    }, stream=False)
    assert out["messages"][1]["role"] == "assistant"
    blocks = out["messages"][1]["content"]
    assert any(b["type"] == "text" for b in blocks)
    assert any(b["type"] == "tool_use" and b["name"] == "f"
               and b["input"] == {"a": 1} for b in blocks)
    assert out["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "42"}],
    }


def test_tools_translated():
    out = chat_to_anthropic_request({
        "model": "c", "messages": [],
        "tools": [{"type": "function",
                    "function": {"name": "search",
                                 "description": "do s",
                                 "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
    }, stream=False)
    assert out["tools"][0] == {
        "name": "search", "description": "do s",
        "input_schema": {"type": "object"},
    }
    assert out["tool_choice"] == {"type": "tool", "name": "search"}


# ---- anthropic_to_chat_response -------------------------------------

def test_text_only_response():
    anth = {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    out = anthropic_to_chat_response(anth)
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 2,
                             "total_tokens": 5}


def test_tool_use_response():
    anth = {
        "id": "x", "type": "message", "role": "assistant", "model": "c",
        "content": [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "call_1", "name": "f",
             "input": {"a": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = anthropic_to_chat_response(anth)
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    assert orjson.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


# ---- anthropic_event_to_chat_chunks ---------------------------------

def test_stream_translation_full_round_trip():
    events = [
        ("message_start", {"type": "message_start",
                            "message": {"id": "x", "type": "message",
                                        "role": "assistant", "model": "c",
                                        "content": [],
                                        "usage": {"input_tokens": 5}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": "!"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    state = {}
    chunks = []
    for _ev, payload in events:
        chunks += anthropic_event_to_chat_chunks(
            payload, state, chatcmpl_id="ccid", created=0, model="c",
        )

    # First chunk is the role
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Concatenated content
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "hi!" in text
    # Final chunk has finish_reason + usage
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens"] == 5
    assert chunks[-1]["usage"]["completion_tokens"] == 2
