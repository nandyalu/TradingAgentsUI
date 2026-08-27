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

import logging
import re

logger = logging.getLogger(__name__)

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
    if failed(retry) is not None:
        logger.error(
            "%s: did it twice. The report for this run will be ungrounded "
            "unless the caller discards it.",
            agent_name,
        )
    return retry
