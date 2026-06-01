"""Tests for the stuck-detection auto-nudge logic."""
from backend_proxy.adapters.stuck_rescue import (
    looks_stuck,
    build_nudge_messages,
)


# ---- looks_stuck -----------------------------------------------------

def test_stuck_detected_now_ill():
    assert looks_stuck("stop", "Got it. Now I'll fix the bug.", 0)


def test_stuck_detected_let_me():
    assert looks_stuck("stop", "Let me write the test file.", 0)


def test_stuck_detected_fixing_now():
    assert looks_stuck("stop", "Fixing buggy.py now.", 0)


def test_stuck_detected_running_tests():
    assert looks_stuck("stop", "Now running the tests:", 0)


def test_stuck_detected_proceeding_to():
    assert looks_stuck("stop", "Proceeding to step 3.", 0)


def test_NOT_stuck_when_finish_is_tool_calls():
    """Real tool call already happened — not stuck."""
    assert not looks_stuck("tool_calls", "Now I'll fix it", 1)


def test_NOT_stuck_when_finish_is_length():
    """max_tokens cut — different category, not stuck."""
    assert not looks_stuck("length", "Now I'll fix it", 0)


def test_NOT_stuck_when_text_is_summary():
    """Past-tense closing — task is genuinely done, no rescue."""
    assert not looks_stuck("stop", "Done. Created qsort.py and test_qsort.py.", 0)


def test_NOT_stuck_when_text_is_empty():
    """Pure tool call with no narration — fine."""
    assert not looks_stuck("stop", "", 0)


def test_NOT_stuck_when_short_text():
    """A 4-char text isn't a meaningful announcement."""
    assert not looks_stuck("stop", "ok", 0)


# ---- build_nudge_messages --------------------------------------------

def test_nudge_messages_appends_assistant_and_user():
    orig = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "do something"},
    ]
    out = build_nudge_messages(orig, "I'll do it now.")
    assert len(out) == 4
    assert out[:2] == orig
    assert out[2] == {"role": "assistant", "content": "I'll do it now."}
    # Critical: the nudge must be role=user, not role=system.
    # Many chat templates (Qwen3) reject non-leading system messages.
    assert out[3]["role"] == "user"
    assert "tool_call" in out[3]["content"].lower() or "promise" in out[3]["content"].lower()


def test_nudge_messages_does_not_use_system_role():
    """Regression test: an earlier draft used role=system, which made vLLM
    reject every rescue with 'System message must be at the beginning'.
    Lock the fix in."""
    out = build_nudge_messages([{"role": "user", "content": "x"}], "ok")
    nudge = out[-1]
    assert nudge["role"] != "system", \
        "nudge must not be role=system — chat templates reject non-leading system messages"
