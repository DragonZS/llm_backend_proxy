"""Unit tests for the OpenAI ↔ Ollama translation helpers."""

from __future__ import annotations

import orjson
import pytest

from backend_proxy.backends.ollama_native import (
    chat_to_ollama_request,
    ollama_chunk_to_chat_chunks,
    ollama_to_chat_response,
)


# ---- request rewrite -------------------------------------------------

def test_chat_to_ollama_minimal():
    out = chat_to_ollama_request(
        {"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
        stream=False,
    )
    assert out == {
        "model": "llama",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_chat_to_ollama_options_mapping():
    out = chat_to_ollama_request({
        "model": "m", "messages": [],
        "max_tokens": 200, "temperature": 0.5, "top_p": 0.9, "stop": ["\nUser:"],
    }, stream=True)
    assert out["stream"] is True
    assert out["options"] == {
        "num_predict": 200, "temperature": 0.5, "top_p": 0.9,
        "stop": ["\nUser:"],
    }


def test_chat_to_ollama_passes_tools_through():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    out = chat_to_ollama_request(
        {"model": "m", "messages": [], "tools": tools, "tool_choice": "auto"},
        stream=False,
    )
    assert out["tools"] == tools
    assert out["tool_choice"] == "auto"


def test_chat_to_ollama_json_mode():
    out = chat_to_ollama_request(
        {"model": "m", "messages": [], "response_format": {"type": "json_object"}},
        stream=False,
    )
    assert out["format"] == "json"


# ---- non-streaming response wrap ------------------------------------

def test_ollama_to_chat_text_only():
    full = {
        "model": "llama",
        "message": {"role": "assistant", "content": "ok"},
        "done": True, "done_reason": "stop",
        "prompt_eval_count": 3, "eval_count": 1,
    }
    out = ollama_to_chat_response(full, model="llama")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 3
    assert out["usage"]["completion_tokens"] == 1
    assert out["usage"]["total_tokens"] == 4


def test_ollama_to_chat_length_finish():
    full = {
        "model": "m",
        "message": {"role": "assistant", "content": "..."},
        "done": True, "done_reason": "length",
    }
    out = ollama_to_chat_response(full, model="m")
    assert out["choices"][0]["finish_reason"] == "length"


def test_ollama_to_chat_tool_calls():
    full = {
        "model": "m",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "f", "arguments": {"a": 1}},
            }],
        },
        "done": True,
    }
    out = ollama_to_chat_response(full, model="m")
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    args = orjson.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert args == {"a": 1}


# ---- streaming chunk translation -----------------------------------

def test_ollama_chunk_content_only():
    chunks = ollama_chunk_to_chat_chunks(
        {"model": "m", "message": {"role": "assistant", "content": "hi"},
         "done": False},
        chatcmpl_id="x", created=0, model="m",
    )
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"] == {"content": "hi"}
    assert chunks[0]["choices"][0]["finish_reason"] is None


def test_ollama_chunk_done_emits_finish_and_usage():
    chunks = ollama_chunk_to_chat_chunks(
        {"model": "m", "message": {"role": "assistant", "content": ""},
         "done": True, "done_reason": "stop",
         "prompt_eval_count": 2, "eval_count": 3},
        chatcmpl_id="x", created=0, model="m",
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c["choices"][0]["finish_reason"] == "stop"
    assert c["usage"]["prompt_tokens"] == 2
    assert c["usage"]["completion_tokens"] == 3


def test_ollama_chunk_skips_empty_non_done():
    chunks = ollama_chunk_to_chat_chunks(
        {"model": "m", "message": {"role": "assistant", "content": ""},
         "done": False},
        chatcmpl_id="x", created=0, model="m",
    )
    assert chunks == []
