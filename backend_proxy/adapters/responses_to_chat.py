"""Translate a Responses-API request body into a Chat-Completions one.

This is a 1:1 OOP port of ``responses_to_chat`` from
``proxy_v1/responses_to_chat_proxy.py`` — the legacy behaviour is preserved
verbatim and gated by golden tests in ``tests/unit/test_responses_to_chat.py``.

Key behaviours preserved:

* Top-level ``instructions`` becomes a leading ``system`` message.
* ``role: "developer"`` is folded into ``system`` (Qwen chat templates only
  understand system/user/assistant/tool).
* Each input item's ``content`` may be a string or a list of typed parts; we
  flatten the latter by concatenating the ``text`` of any ``input_text`` /
  ``output_text`` / ``text`` parts.
* ``function_call``        -> assistant message with a ``tool_calls`` entry.
* ``function_call_output`` -> ``role: tool`` message with ``tool_call_id``.
* ``tools`` (Responses style ``{type: function, name, description, parameters}``)
  is rewritten to Chat style ``{type: function, function: {...}}``.
* Common generation parameters are mapped / passed through where Chat API and
  vLLM support them.
* ``max_tokens``: derived from ``max_output_tokens`` when present, else left
  unset here so the dedicated ``codex_max_tokens`` adapter (next in the
  Codex pipeline) can apply its configurable fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..core.context import RequestContext
from ..core.errors import BadRequestError
from .base import RequestAdapter

log = logging.getLogger("backend_proxy.adapters.responses_to_chat")


_DIRECT_PARAM_MAP = {
    "frequency_penalty": "frequency_penalty",
    "parallel_tool_calls": "parallel_tool_calls",
    "presence_penalty": "presence_penalty",
    "seed": "seed",
    "stop": "stop",
    "temperature": "temperature",
    "top_p": "top_p",
}


# When Codex is paired with a non-OpenAI model (Qwen, Llama, …), the system
# prompt mentions tools that don't actually exist in the forwarded tool list
# — most notably `apply_patch`. The model then either invents a call to
# `apply_patch` (rejected by Codex), tries `write_stdin` for file-writing
# (rejected: "Unknown process id"), or just gives up and emits a chatty
# response with no tool_call (the user-visible "agent stopped" symptom).
#
# We patch that gap by appending a short, low-noise reminder to whatever
# system / instructions block already exists, listing the tools that
# ARE available and pinning down how to write files. The reminder is only
# emitted when (a) the actual tool set lacks `apply_patch` AND
# (b) it includes `exec_command` — i.e., the Codex-CLI shape exactly.
def _maybe_codex_tool_hint(forwarded_tool_names: set[str]) -> str | None:
    if not forwarded_tool_names:
        return None
    if "apply_patch" in forwarded_tool_names:
        return None
    if "exec_command" not in forwarded_tool_names:
        return None
    # Kept short. Long instructions dilute attention; the most important
    # rules need to fit in the model's working memory alongside the user's
    # actual task. Stricter wording for the action-then-call discipline
    # is enforced by the streaming-layer auto-nudge in api/openai_responses.py.
    return (
        "\n\n# Codex-on-non-OpenAI environment — proxy notes\n"
        "Tools available here do NOT include `apply_patch`. To create or "
        "edit files, use `exec_command` with `cmd=[\"bash\",\"-lc\",\"cat > "
        "<path> <<'PROXY_EOF'\\n<content>\\nPROXY_EOF\"]`. Inside the "
        "heredoc body keep quotes literal — do not escape them. Never use "
        "`write_stdin` for file writes; it only feeds keystrokes to a "
        "running process and will fail with 'Unknown process id'.\n"
        "If your turn promises to do something next ('now I'll …', "
        "'let me …', 'fixing X now'), immediately emit the corresponding "
        "tool_call in the same turn — do not stop on a promise."
    )


def _extract_text_content(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict)
        and p.get("type") in ("input_text", "output_text", "text")
    )


def _map_tool_choice(choice: Any) -> Any:
    if choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            # Responses API:        {"type": "function", "name": "foo"}
            # Chat Completions API: {"type": "function", "function": {"name": "foo"}}
            # vLLM only accepts the latter — rewrite if the client used the
            # Responses-style flat shape (Codex doesn't pin tool_choice this
            # way, but OpenAI SDK users hitting /v1/responses directly do).
            if "function" not in choice and "name" in choice:
                return {"type": "function", "function": {"name": choice["name"]}}
            return choice
        raise BadRequestError(f"unsupported Responses tool_choice type: {choice.get('type')}")
    return choice


def _map_text_format(req: Dict[str, Any], out: Dict[str, Any]) -> None:
    text = req.get("text")
    if not isinstance(text, dict):
        return
    fmt = text.get("format")
    if fmt is None:
        return
    if fmt == "text" or fmt == {"type": "text"}:
        return
    if isinstance(fmt, dict):
        # Responses API uses text.format; Chat Completions uses response_format.
        # vLLM supports structured output through the OpenAI-compatible
        # response_format path and vLLM-specific structured output extensions.
        out["response_format"] = fmt
        return
    raise BadRequestError("unsupported text.format value")


class ResponsesToChatAdapter(RequestAdapter):
    name = "responses_to_chat"

    async def transform(self, req: Dict[str, Any], ctx: RequestContext) -> Dict[str, Any]:
        if "model" not in req:
            raise BadRequestError("missing 'model' in request")

        # Responses API session continuation. The official OpenAI server
        # persists every response by id and lets a subsequent request resume
        # by passing `previous_response_id`. vLLM 0.x does NOT persist that
        # state — silently ignoring the field would cause clients (Codex,
        # Cursor, OpenAI SDK retry paths) to talk past each other: from their
        # POV the previous turn happened, but the model never sees it.
        # Fail loud so the client falls back to sending the full conversation
        # in `input`.
        if req.get("previous_response_id"):
            raise BadRequestError(
                "previous_response_id is not supported by this backend; "
                "client must include the full conversation in `input`"
            )

        sys_parts: List[str] = []
        if req.get("instructions"):
            sys_parts.append(req["instructions"])

        msgs: List[Dict[str, Any]] = []

        input_items = req.get("input", []) or []
        if isinstance(input_items, str):
            input_items = [input_items]

        for item in input_items:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            t = item.get("type", "message")
            role = item.get("role", "user")
            if role == "developer":
                role = "system"

            if t == "message":
                content = _extract_text_content(item.get("content", []))
                if role == "system":
                    sys_parts.append(content)
                else:
                    msgs.append({"role": role, "content": content})

            elif t == "function_call":
                msgs.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": item.get("call_id", item.get("id", "call_0")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }],
                })
            elif t == "function_call_output":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", "call_0"),
                    "content": item.get("output", ""),
                })

        if sys_parts:
            # We collapse every system/developer/instructions fragment into a
            # SINGLE leading system message. That preserves intent for Codex
            # (whose developer block always sits at the top) but rewrites the
            # temporal ordering when the client interleaves a `developer`
            # message *after* tool calls, e.g. "from now on, switch to patch
            # mode". Qwen reacts differently to "first instruction" vs
            # "latest instruction"; surface this so it's debuggable.
            #
            # Codex sends `instructions` + one `developer` block on EVERY
            # turn, so this branch fires constantly. Use DEBUG to avoid
            # spamming WARNING; promote to WARNING only when the developer
            # message is interleaved after non-developer messages (the case
            # that actually risks semantic drift).
            if len(sys_parts) > 1:
                first_non_dev_idx = next(
                    (i for i, m in enumerate(msgs) if m.get("role") != "system"),
                    None,
                )
                last_dev_in_input = sum(1 for m in msgs if m.get("role") == "system")
                # If we already moved every developer/system to the front by
                # collecting into sys_parts, there's no interleaving — quiet.
                # Only when we see the *order* of input as developer→user→developer
                # do we need to alert. We approximate this with: were any
                # developer fragments collected AFTER seeing a user/assistant
                # message? sys_parts is built strictly from input order, so a
                # user msg already in `msgs` followed by another sys_part is
                # the suspicious case.
                interleaved = first_non_dev_idx is not None and last_dev_in_input == 0 \
                              and any(_part for _part in sys_parts[1:])
                if interleaved:
                    log.warning(
                        "[%s] merged %d system/developer fragments after user/assistant "
                        "messages; original ordering relative to user/tool messages was lost",
                        getattr(ctx, "trace_id", "-"), len(sys_parts),
                    )
                else:
                    log.debug(
                        "[%s] merged %d system/developer fragments into one leading "
                        "system message", getattr(ctx, "trace_id", "-"), len(sys_parts),
                    )
            msgs.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})

        out: Dict[str, Any] = {
            "model": req["model"],
            "messages": msgs,
            "stream": bool(req.get("stream")),
        }

        for src, dst in _DIRECT_PARAM_MAP.items():
            if src in req:
                out[dst] = req[src]

        # `max_output_tokens` (Responses API) maps to `max_tokens` (Chat API).
        # When the client omits it, leave `max_tokens` UNSET here and let the
        # dedicated ``codex_max_tokens`` adapter apply a configurable fallback
        # (see ``adapters/codex_max_tokens.py``). Hard-coding the fallback in
        # this adapter would make ``codex_max_tokens`` a no-op and silently
        # ignore YAML overrides like
        # ``adapters.codex_max_tokens.options.fallback_max_tokens``.
        if "max_output_tokens" in req and req["max_output_tokens"]:
            out["max_tokens"] = req["max_output_tokens"]

        reasoning = req.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            out["reasoning_effort"] = reasoning["effort"]

        _map_text_format(req, out)
        if isinstance(req.get("response_format"), dict):
            out["response_format"] = req["response_format"]

        if req.get("tools"):
            tools = []
            unsupported = []
            for tdef in req["tools"]:
                if not isinstance(tdef, dict):
                    continue
                if tdef.get("type") == "function":
                    fn = {k: tdef[k] for k in ("name", "description", "parameters", "strict") if k in tdef}
                    tools.append({"type": "function", "function": fn})
                else:
                    unsupported.append(tdef.get("type", "<missing>"))
            if unsupported:
                # Codex 0.135+ injects internal tool types like `namespace`,
                # `web_search`, `local_shell`, etc. on every request. The
                # OpenAI Responses API silently ignores tools the backend
                # doesn't implement; mirror that behaviour. Raising here would
                # turn every Codex turn into a 400 ("ERROR: unsupported
                # Responses tool type"). Just drop them and continue with
                # whatever function tools (if any) survived.
                log.debug(
                    "[%s] dropping %d unsupported Responses tool type(s): %s",
                    getattr(ctx, "trace_id", "-"),
                    len(unsupported),
                    ", ".join(str(t) for t in unsupported),
                )
            if tools:
                out["tools"] = tools
                # Inject the Codex-with-non-OpenAI-model tool hint when
                # appropriate. Append to the leading system message we just
                # built (if any), or prepend a new one. This is what stops
                # Qwen from inventing apply_patch / mis-using write_stdin.
                hint = _maybe_codex_tool_hint({(t.get("function") or {}).get("name", "") for t in tools})
                if hint:
                    if msgs and msgs[0].get("role") == "system":
                        msgs[0]["content"] = (msgs[0].get("content") or "") + hint
                    else:
                        msgs.insert(0, {"role": "system", "content": hint.lstrip()})
        if req.get("tool_choice"):
            out["tool_choice"] = _map_tool_choice(req["tool_choice"])

        # Stash a flag so downstream stream/response adapters know to re-wrap.
        ctx.extras["protocol"] = "responses"
        return out
