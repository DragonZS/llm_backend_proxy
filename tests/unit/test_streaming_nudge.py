"""Unit tests for the streaming inline nudge logic."""
from __future__ import annotations
import asyncio
import json
import re
from typing import List

import pytest

from backend_proxy.adapters.chat_to_responses import (
    _new_state,
    chat_chunk_to_responses_events,
)
from backend_proxy.api.openai_responses import (
    _consume_one_stream,
    _reset_state_for_rescue_phase,
    _state_to_output,
)


def _enc_chunk(d: dict) -> bytes:
    return b"data: " + json.dumps(d).encode() + b"\n\n"


async def _fake_stream(chunks: List[dict]):
    for c in chunks:
        yield _enc_chunk(c)


def _content_chunk(text: str, finish: str | None = None) -> dict:
    return {
        "model": "test", "choices": [{
            "index": 0,
            "delta": {"content": text} if text else {},
            "finish_reason": finish,
        }],
    }


def _tool_call_chunk(idx: int, name: str | None, args: str, finish: str | None = None) -> dict:
    tc = {"index": idx, "function": {"arguments": args}}
    if name:
        tc["id"] = f"call_{idx}"
        tc["function"]["name"] = name
    return {
        "model": "test", "choices": [{
            "index": 0,
            "delta": {"tool_calls": [tc]},
            "finish_reason": finish,
        }],
    }


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---- _consume_one_stream: suppress_completed semantics --------------

def test_consume_suppresses_completed_when_asked():
    state = _new_state()
    chunks = [
        _content_chunk("Now I'll fix it."),
        _content_chunk("", finish="stop"),
    ]
    async def collect():
        out = []
        async for piece in _consume_one_stream(_fake_stream(chunks), state, suppress_completed=True):
            out.append(piece)
        return out

    pieces = _run(collect())
    body = b"".join(pieces).decode()
    # response.completed must NOT appear when suppressed
    assert "response.completed" not in body
    # but content deltas should be there
    assert "Now I'll fix it." in body
    # state should have been rolled back so caller can decide
    assert state["completed"] is False
    # finish_reason still recorded so stuck detection can run
    assert state.get("telemetry_finish_reason") == "stop"


def test_consume_emits_completed_when_not_suppressed():
    state = _new_state()
    chunks = [
        _content_chunk("Hello."),
        _content_chunk("", finish="stop"),
    ]
    async def collect():
        out = []
        async for piece in _consume_one_stream(_fake_stream(chunks), state, suppress_completed=False):
            out.append(piece)
        return out

    pieces = _run(collect())
    body = b"".join(pieces).decode()
    assert "response.completed" in body
    assert state["completed"] is True


# ---- _reset_state_for_rescue_phase ------------------------------------

def test_reset_state_keeps_resp_id_and_started():
    state = _new_state()
    state["started"] = True
    state["resp_id"] = "resp_keep"
    state["msg_opened"] = True
    state["msg_closed"] = True
    state["text"] = "promised"
    state["next_output_index"] = 1
    old_item_id = state["item_id"]

    _reset_state_for_rescue_phase(state)

    # resp_id and started preserved (response.created was already sent)
    assert state["resp_id"] == "resp_keep"
    assert state["started"] is True
    # message-item lifecycle reset so a NEW item can open
    assert state["msg_opened"] is False
    assert state["msg_closed"] is False
    assert state["text"] == ""
    assert state["item_id"] != old_item_id
    # output_index counter must NOT reset — preserves ordering
    assert state["next_output_index"] == 1
    # completed flag reset so the rescue phase can finalise
    assert state["completed"] is False


# ---- _state_to_output ---------------------------------------------------

def test_state_to_output_includes_message_and_tool_calls_in_order():
    state = _new_state()
    state["msg_opened"] = True
    state["msg_output_index"] = 0
    state["next_output_index"] = 1
    state["text"] = "hi"
    state["tool_calls"][0] = {
        "output_index": 1, "item_id": "msg_tc",
        "call_id": "call_0", "name": "do_thing", "arguments": '{"x":1}',
        "opened": True, "closed": True,
    }
    out = _state_to_output(state)
    assert len(out) == 2
    assert out[0]["type"] == "message"
    assert out[0]["content"][0]["text"] == "hi"
    assert out[1]["type"] == "function_call"
    assert out[1]["name"] == "do_thing"


def test_state_to_output_tool_only():
    state = _new_state()
    state["next_output_index"] = 1
    state["tool_calls"][0] = {
        "output_index": 0, "item_id": "msg_tc",
        "call_id": "call_0", "name": "do_thing", "arguments": "{}",
        "opened": True, "closed": True,
    }
    out = _state_to_output(state)
    assert len(out) == 1
    assert out[0]["type"] == "function_call"


# ---- shared state across two streams ---------------------------------

def test_two_phase_stream_shares_resp_id_and_advances_output_index():
    """End-to-end-ish: phase 1 produces a message that announces an action;
    after rescue reset, phase 2 emits a tool call. Both should share resp_id
    and the tool call should have output_index >= the message's."""
    state = _new_state()
    phase1 = [_content_chunk("Now I'll fix it."), _content_chunk("", finish="stop")]
    phase2 = [
        _tool_call_chunk(0, "do_thing", '{"x":'),
        _tool_call_chunk(0, None, '1}'),
        _tool_call_chunk(0, None, '', finish="tool_calls"),
    ]

    async def go():
        bytes1 = []
        async for p in _consume_one_stream(_fake_stream(phase1), state, suppress_completed=True):
            bytes1.append(p)

        # After phase 1 we should have detected text but no tool calls.
        assert state.get("telemetry_finish_reason") == "stop"
        assert len(state["tool_calls"]) == 0
        original_resp_id = state["resp_id"]
        original_msg_idx = state.get("msg_output_index")

        _reset_state_for_rescue_phase(state)
        bytes2 = []
        async for p in _consume_one_stream(_fake_stream(phase2), state, suppress_completed=False):
            bytes2.append(p)
        return bytes1, bytes2, original_resp_id, original_msg_idx

    b1, b2, orig_resp, orig_msg_idx = _run(go())
    body1 = b"".join(b1).decode()
    body2 = b"".join(b2).decode()

    # Both phases reference the SAME response id
    assert orig_resp in body1
    # Phase 2 contains the tool_call shell event
    assert "function_call" in body2
    # Phase 2 emits response.completed (we did NOT suppress)
    assert "response.completed" in body2
    # The tool call's output_index must be >= the message's so ordering
    # is well-defined when the client builds output[]
    m = re.search(r'"output_index":(\d+).*?"function_call"', body2, re.DOTALL)
    assert m is not None, "expected function_call event in phase 2 body"
    tool_idx = int(m.group(1))
    assert tool_idx >= (orig_msg_idx or 0)
