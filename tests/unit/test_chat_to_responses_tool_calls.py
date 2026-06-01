"""Unit tests for the chat→responses adapter, focusing on tool_calls.

Regression: streaming tool_calls were silently dropped. The model would
say "I'll write code" then call apply_patch — the proxy translated none
of that and the client (codex / cursor) saw response.completed with no
function_call item, rendering the turn as an empty answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend_proxy.adapters.chat_to_responses import (
    chat_chunk_to_responses_events,
    chat_full_to_responses,
)


# ----------- non-streaming -----------------------------------------------

def test_full_with_tool_calls_emits_function_call_item():
    chat = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I'll create that file.",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": '{"patch":"X"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    out = chat_full_to_responses(chat)
    types = [item["type"] for item in out["output"]]
    assert types == ["message", "function_call"]
    fc = out["output"][1]
    assert fc["call_id"] == "call_abc"
    assert fc["name"] == "apply_patch"
    assert fc["arguments"] == '{"patch":"X"}'


def test_full_with_only_tool_calls_no_text_emits_only_function_call():
    """Tool-only turns must not emit a phantom empty assistant message."""
    chat = {
        "choices": [{
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    out = chat_full_to_responses(chat)
    types = [item["type"] for item in out["output"]]
    assert types == ["function_call"]


def test_full_text_only_unchanged():
    """The historical text-only path must keep its single message item."""
    chat = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    out = chat_full_to_responses(chat)
    assert [i["type"] for i in out["output"]] == ["message"]
    assert out["output"][0]["content"][0]["text"] == "hello"


def test_full_usage_maps_chat_names_to_responses_names():
    chat = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    out = chat_full_to_responses(chat)
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


# ----------- streaming ----------------------------------------------------

def _drive(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the chunk-to-events pipeline and collect emitted event dicts."""
    state: Dict[str, Any] = {
        "started": False, "text": "", "phase": None,
        "resp_id": "resp_test", "item_id": "msg_test",
        "msg_opened": False, "msg_closed": False,
        "tool_calls": {}, "next_output_index": 0,
    }
    events: List[Dict[str, Any]] = []
    for c in chunks:
        for _t, obj in chat_chunk_to_responses_events(c, state):
            events.append(obj)
    return events


def test_stream_tool_call_lifecycle():
    """Realistic 4-fragment vLLM tool_call stream — id+name in chunk 1,
    arguments split across chunks 2 and 3, finish_reason in chunk 4."""
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{
            "id": "call_xyz", "type": "function", "index": 0,
            "function": {"name": "apply_patch", "arguments": ""}
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": '{"pat'}
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": 'ch":"X"}'}
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = _drive(chunks)
    types = [e["type"] for e in events]

    assert types[0] == "response.created"
    # Function-call shell appears as soon as we know the name (chunk 1).
    assert "response.output_item.added" in types
    added = next(e for e in events if e["type"] == "response.output_item.added")
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "apply_patch"
    assert added["item"]["call_id"] == "call_xyz"

    deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert "".join(e["delta"] for e in deltas) == '{"patch":"X"}'

    done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
    assert done["arguments"] == '{"patch":"X"}'

    completed = next(e for e in events if e["type"] == "response.completed")
    out = completed["response"]["output"]
    assert len(out) == 1 and out[0]["type"] == "function_call"
    assert out[0]["arguments"] == '{"patch":"X"}'


def test_stream_text_then_tool_call():
    """The model emits a sentence of plain text, THEN decides to call a tool.
    The output must contain BOTH a message item and a function_call item,
    in that order."""
    chunks = [
        {"choices": [{"delta": {"content": "I'll "}}]},
        {"choices": [{"delta": {"content": "do it."}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "id": "c1", "index": 0, "type": "function",
            "function": {"name": "apply_patch", "arguments": "{}"}
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = _drive(chunks)
    completed = next(e for e in events if e["type"] == "response.completed")
    items = completed["response"]["output"]
    assert [i["type"] for i in items] == ["message", "function_call"]
    assert items[0]["content"][0]["text"] == "I'll do it."
    assert items[1]["arguments"] == "{}"


def test_stream_text_only_still_works():
    """Existing text-only behaviour preserved end-to-end."""
    chunks = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    events = _drive(chunks)
    types = [e["type"] for e in events]
    # No function_call events should appear in a pure-text turn.
    assert not any("function_call" in t for t in types)
    completed = next(e for e in events if e["type"] == "response.completed")
    assert completed["response"]["output"][0]["type"] == "message"


def test_stream_thinking_then_finish_closes_think_tag():
    """Reasoning that doesn't transition to content must still close </think>."""
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "let me think"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    events = _drive(chunks)
    completed = next(e for e in events if e["type"] == "response.completed")
    text = completed["response"]["output"][0]["content"][0]["text"]
    assert text.startswith("<think>") and text.rstrip().endswith("</think>")
