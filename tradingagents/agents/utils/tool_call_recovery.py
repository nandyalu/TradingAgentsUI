"""Catch a model that writes its tool call instead of making it.

An analyst node decides it is finished when the model returns no tool calls:

    if len(result.tool_calls) == 0:
        report = result.content

That reads two different things as one. "I am finished, here is the report" and
"I tried to call a tool and typed it as prose" both arrive as an empty
``tool_calls`` list, so a model that narrates its intent gets that narration
filed as the finished report.

It is not hypothetical. On 2026-08-27 `gemma4-e4b-qat-128k` returned 532
characters beginning "**Step 1: Retrieve Necessary Stock Data** ... I will call
`get_stock_data` first" followed by a fenced ``{"tool_calls": [...]}`` block,
and that became the market report for the whole run. Nothing logged a problem.

**Ollama ignores ``tool_choice``.** Sending ``required`` behaves exactly like
``auto`` and like sending nothing, measured on the same model, so the standard
lever for forcing a call is not available on this stack. Detection is what is
left.

The model is not incapable. Given "Get me AAPL's price data from 2026-07-27 to
2026-08-27" it made a correct call every time; given "think aloud about your
plan first, in detail, before doing anything" it narrated every time. The
prompt decides it, which is why the recovery here retries rather than giving up.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Added to every analyst's preamble. The failure this addresses is not a
# capability problem: given "Get me AAPL's price data from 2026-07-27 to
# 2026-08-27" the same model made a correct call every time, and given "think
# aloud about your plan first, in detail, before doing anything" it narrated
# every time. The prompt decides it, so the prompt says so plainly.
CALL_DO_NOT_DESCRIBE = (
    " Call the tools. Do not write out a plan first, and do not describe the"
    " call you are about to make: a tool call written as text is not a tool"
    " call, the data never arrives, and your description becomes the report."
    " Fetch first, then write the report from what comes back."
)

# A fenced block, or the literal key an OpenAI-shaped call would use. Either on
# its own is weak evidence; paired with a bound tool's name it is not.
_STRUCTURE = re.compile(r"```\s*json|\"tool_calls\"|'tool_calls'|\"function\"\s*:", re.I)


def printed_a_tool_call(content: str | None, tool_names) -> bool:
    """Whether ``content`` looks like a tool call written out rather than made.

    Requires two things together: the name of a tool this agent actually bound,
    and JSON structure around it. Either alone gives false positives — a market
    report may reasonably mention ``get_stock_data`` in prose, and a report may
    contain a fenced JSON table.
    """
    if not content:
        return False
    text = str(content)
    if not _STRUCTURE.search(text):
        return False
    return any(name and name in text for name in tool_names)


def answered_without_ever_fetching(messages, result) -> bool:
    """Whether this answer is ungrounded because no tool ever ran.

    Stronger than reading the text, and it catches the failure that text
    cannot. A model sometimes writes a long plan — "### Phase 1: Detailed Swing
    Trade Planning and Methodology" — that names no tool and contains no JSON,
    so it is indistinguishable from a finished report by content alone. What
    gives it away is the conversation: if nothing has returned a tool result
    and the model is not asking for one now, then whatever it just wrote was
    composed from no data at all.

    An analyst that fetched and found nothing is a different case and must be
    left alone. It has tool results in its history, so it fails this check.
    """
    if result.tool_calls:
        return False
    for message in messages or []:
        role = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role in ("tool", "function"):
            return False
    return True


def parse_printed_tool_call(content: str | None, tool_names) -> list[dict] | None:
    """Pull a written-out tool call from prose, so the turn can be salvaged.

    The last resort, and the one with a cost worth naming: it teaches the loop
    to accept a shape the API never sent, so a model that keeps narrating keeps
    working and nobody notices it is doing the wrong thing. Retrying is
    preferred because it leaves the model's behaviour visible.

    It is here for the case retrying does not fix — a model that narrates every
    time, where the alternative is no data at all.

    Recognises the two shapes seen in real runs: an OpenAI-ish
    ``{"tool_calls": [{"function": ..., "args": ...}]}`` and a bare
    ``{"name": ..., "arguments": ...}``. Returns None when nothing usable is
    found, which is the common case and must stay cheap.
    """
    if not content:
        return None
    text = str(content)
    found: list[dict] = []
    # Balanced-brace scan rather than a regex. A non-greedy pattern stops at
    # the first closing brace, so {"tool_calls": [{"function": {...}}]} — a
    # shape real runs produce — never parsed at all.
    for blob in _json_objects(text):
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        for candidate in _candidates(parsed):
            name = candidate.get("name") or candidate.get("function")
            if isinstance(name, dict):          # {"function": {"name": ...}}
                name = name.get("name")
            if not name or name not in tool_names:
                continue
            args = (candidate.get("args") or candidate.get("arguments")
                    or candidate.get("parameters") or {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict):
                found.append({"name": name, "args": args, "id": f"recovered-{len(found)}"})
    return found or None


def _json_objects(text: str):
    """Every balanced ``{...}`` run in ``text``, outermost first.

    Quotes are tracked so a brace inside a string does not end the object, and
    a backslash escape does not end the string.
    """
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]
            elif depth < 0:
                depth = 0


def _candidates(parsed):
    """Every dict in a parsed blob that might be a call."""
    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            for item in parsed["tool_calls"]:
                if isinstance(item, dict):
                    yield item
        yield parsed
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                yield item


def invoke_with_tool_call_recovery(chain, messages, tool_names, agent_name: str):
    """Invoke ``chain``, retrying once if the model printed its call as prose.

    Returns the model's result. A retry that fails the same way is returned
    anyway — the caller decides what to do with it — but it is logged loudly,
    because the alternative is a run that looks like it worked.

    The retry re-invokes with the same messages rather than appending a nudge.
    At the sampling these models run at, the same prompt does not produce the
    same answer, and adding a correction turn changes the conversation every
    later stage reads.
    """
    def failed(res) -> str | None:
        if printed_a_tool_call(res.content, tool_names):
            return "wrote its tool call as text instead of calling it"
        if answered_without_ever_fetching(messages, res):
            return "answered without ever fetching anything, so the answer has no data behind it"
        return None

    result = chain.invoke(messages)
    why = failed(result)
    if why is None:
        return result

    logger.warning(
        "%s: %s; retrying once. Left alone this becomes the agent's report.",
        agent_name, why,
    )
    retry = chain.invoke(messages)
    if failed(retry) is None:
        return retry

    # Retrying was preferred and did not work. Salvage the written-out call
    # rather than let a narration become the report, and say so loudly: this
    # accepts a shape the API never sent, which is the thing that lets a model
    # keep narrating without anyone noticing.
    salvaged = parse_printed_tool_call(retry.content, tool_names)
    if salvaged:
        logger.error(
            "%s: printed its tool call twice; executing the written-out call "
            "%s rather than filing the narration as the report.",
            agent_name, [c["name"] for c in salvaged],
        )
        retry.tool_calls = salvaged
        return retry

    logger.error(
        "%s: did it twice and nothing could be salvaged. The report for this "
        "run will be ungrounded unless the caller discards it.",
        agent_name,
    )
    return retry
