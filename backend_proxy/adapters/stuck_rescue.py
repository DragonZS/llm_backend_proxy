"""Stuck-detection auto-nudge for the Codex/Qwen problem.

Problem: Qwen-style models in Codex sometimes finish a streaming turn with
``finish_reason=stop`` and ``tool_calls=[]`` while the visible text just
ANNOUNCES an action it didn't take ("now I'll fix the bug", "let me write
the test"). The OpenAI Responses API treats that as a completed turn,
Codex stops the agent loop, and the user sees the task half-done.

Auto-nudge: when the proxy detects this exact shape, it opens a SECOND
upstream call with the original messages plus an injected "you said you'd
do X but didn't — do it now" reminder, and substitutes the second
response's content into the streaming output. Off by default; opt in
with ``BACKEND_PROXY_AUTO_NUDGE=1``.

Design notes:
  * We re-issue against ``/v1/chat/completions`` non-streamed. That is the
    simplest correct way to get a fresh result without juggling two SSE
    streams concurrently. The latency cost is one full re-generation.
  * Detection runs on the buffered Chat-Completions chunks, NOT on the
    Responses-API events the client sees. We need finish_reason/tool_calls
    counts which are native Chat fields.
  * Trigger conditions (all must hold):
      - finish_reason == "stop"
      - 0 tool_calls
      - non-trivial text (≥ 1 char)
      - text matches one of the announce-future-action patterns
  * If any condition fails we pass through unchanged.
"""
from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import orjson

log = logging.getLogger("backend_proxy.stuck_rescue")


# Patterns that strongly suggest the model promised an action it didn't
# actually take. Tuned against real Codex/Qwen3 logs. Case-insensitive.
_ANNOUNCE_PATTERNS = [
    r"\bnow (?:i|let me|i'll|i will)\b",
    r"\blet me\s+(?:write|create|run|fix|check|verify|inspect|read|edit|update|add|remove)\b",
    r"\bi(?:'ll| will| am going to| am about to)\s+(?:write|create|run|fix|check|verify|inspect|read|edit|update|add|remove)\b",
    r"\bfixing\s+\S+\s+now\b",
    r"\bwriting\s+(?:the|a|an|now|next)\b",
    r"\brunning\s+(?:the|a|an|now|next|pytest|tests?)\b",
    r"\bnext,?\s+i(?:'ll| will)\b",
    r"\bproceeding to\b",
    r"\bmoving (?:on )?to\s+step\b",
]
_ANNOUNCE_RE = re.compile("|".join(_ANNOUNCE_PATTERNS), re.IGNORECASE)


def looks_stuck(finish: Optional[str], text: str, n_tool_calls: int) -> bool:
    if finish != "stop":
        return False
    if n_tool_calls > 0:
        return False
    if not text or len(text) < 5:
        return False
    return bool(_ANNOUNCE_RE.search(text))


_NUDGE_TEXT = (
    "Your previous response promised an action ('now I'll …', 'let me …', "
    "'fixing X now', etc.) but did not emit a tool_call. The user is "
    "waiting and the task is incomplete. Continue from where you left off "
    "by IMMEDIATELY emitting the tool_call(s) needed to fulfill what you "
    "promised. Do not narrate; just call the tool. If the task was actually "
    "complete, respond with one short past-tense summary sentence and no "
    "future-tense verbs."
)


def build_nudge_messages(
    original_messages: List[Dict[str, Any]],
    assistant_text: str,
) -> List[Dict[str, Any]]:
    """Return a new messages array: original + assistant's stuck reply +
    a USER nudge. Keeping the assistant turn lets the model see what
    it just said so it can finish the announced action.

    Note: the nudge is sent as ``role=user`` not ``role=system``. Many
    chat templates (Qwen3 included) reject non-leading system messages
    with "System message must be at the beginning" — so a system message
    here would break the rescue call entirely. A user-role nudge works
    universally."""
    return [
        *original_messages,
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": _NUDGE_TEXT},
    ]
