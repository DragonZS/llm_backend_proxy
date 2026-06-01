"""ID generators — match the prefixes used by Codex / OpenAI."""

from __future__ import annotations

import uuid


def new_response_id() -> str:
    return "resp_" + uuid.uuid4().hex[:24]


def new_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def new_chatcmpl_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]
