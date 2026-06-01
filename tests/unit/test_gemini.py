"""Unit tests for chat ↔ Gemini adapters."""

from __future__ import annotations

import orjson

from backend_proxy.adapters.gemini import (
    chat_to_gemini_request,
    gemini_chunk_to_chat_chunks,
    gemini_to_chat_response,
)


# ---- chat_to_gemini_request ----------------------------------------

def test_basic_user_message():
    out = chat_to_gemini_request({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert out["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_system_messages_become_systemInstruction():
    out = chat_to_gemini_request({
        "model": "g",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "system", "content": "no emoji"},
            {"role": "user", "content": "hi"},
        ],
    })
    assert out["systemInstruction"]["parts"][0]["text"] == "be brief\n\nno emoji"


def test_assistant_role_is_model():
    out = chat_to_gemini_request({
        "model": "g",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "more"},
        ],
    })
    roles = [c["role"] for c in out["contents"]]
    assert roles == ["user", "model", "user"]


def test_tool_calls_become_functionCall_parts():
    out = chat_to_gemini_request({
        "model": "g",
        "messages": [
            {"role": "user", "content": "find x"},
            {"role": "assistant", "content": "calling",
             "tool_calls": [{
                 "id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "x"}'},
             }]},
            {"role": "tool", "tool_call_id": "call_1", "name": "search",
             "content": '{"result": 42}'},
        ],
    })
    assistant_parts = out["contents"][1]["parts"]
    assert any("text" in p for p in assistant_parts)
    fc = next(p["functionCall"] for p in assistant_parts if "functionCall" in p)
    assert fc == {"name": "search", "args": {"q": "x"}}
    # Tool result lands as a user-role functionResponse part
    fr = out["contents"][2]["parts"][0]["functionResponse"]
    assert fr["name"] == "search"


def test_generation_config_mapping():
    out = chat_to_gemini_request({
        "model": "g", "messages": [],
        "max_tokens": 1024, "temperature": 0.5, "top_p": 0.9,
        "stop": ["\nUser:"],
    })
    assert out["generationConfig"] == {
        "maxOutputTokens": 1024, "temperature": 0.5, "topP": 0.9,
        "stopSequences": ["\nUser:"],
    }


def test_tools_translated():
    out = chat_to_gemini_request({
        "model": "g", "messages": [],
        "tools": [{"type": "function",
                    "function": {"name": "f", "description": "d",
                                 "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    })
    decls = out["tools"][0]["functionDeclarations"]
    assert decls[0]["name"] == "f"
    cfg = out["toolConfig"]["functionCallingConfig"]
    assert cfg == {"mode": "ANY", "allowedFunctionNames": ["f"]}


# ---- gemini_to_chat_response ---------------------------------------

def test_gemini_text_response():
    gem = {
        "candidates": [{
            "content": {"role": "model",
                         "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2,
                            "totalTokenCount": 5},
    }
    out = gemini_to_chat_response(gem, model="gemini-1.5-pro")
    assert out["choices"][0]["message"]["content"] == "ok"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 5


def test_gemini_function_call_response():
    gem = {
        "candidates": [{
            "content": {"role": "model",
                         "parts": [{"functionCall": {"name": "f", "args": {"a": 1}}}]},
            "finishReason": "STOP",
        }],
    }
    out = gemini_to_chat_response(gem, model="g")
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "f"
    assert orjson.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


# ---- gemini_chunk_to_chat_chunks -----------------------------------

def test_gemini_stream_chunk_role_then_content():
    state: dict = {}
    out = gemini_chunk_to_chat_chunks(
        {"candidates": [{"content": {"role": "model",
                                       "parts": [{"text": "hi"}]}}]},
        state, chatcmpl_id="x", created=0, model="g",
    )
    assert out[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert out[1]["choices"][0]["delta"] == {"content": "hi"}


def test_gemini_stream_finish_with_usage():
    state: dict = {"role_emitted": True}
    out = gemini_chunk_to_chat_chunks(
        {"candidates": [{"content": {"parts": [{"text": "!"}]},
                          "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2,
                            "totalTokenCount": 3}},
        state, chatcmpl_id="x", created=0, model="g",
    )
    assert out[0]["choices"][0]["delta"] == {"content": "!"}
    last = out[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["prompt_tokens"] == 1
    assert last["usage"]["completion_tokens"] == 2
