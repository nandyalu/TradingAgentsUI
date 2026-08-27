"""Catching a model that writes its tool call instead of making it.

An analyst decides it is finished when tool_calls is empty, so a model that
narrates its intent gets that narration filed as the finished report. On
2026-08-27 that produced a 532-character market report reading "I will call
get_stock_data first" for a whole run, and nothing logged a problem.
"""
import logging

import pytest

from tradingagents.agents.utils.tool_call_recovery import (
    invoke_with_tool_call_recovery,
    printed_a_tool_call,
)

TOOLS = ["get_stock_data", "get_indicators", "get_verified_market_snapshot"]

# Verbatim from the run that failed, trimmed.
REAL_FAILURE = """**Step 1: Retrieve Necessary Stock Data**

To analyze AAPL for a swing trade I will start by retrieving recent stock data.
I will call `get_stock_data` first.

```json
{
  "tool_calls": [
    {"function": "get_stock_data", "args": {"symbol": "AAPL"}}
  ]
}
```"""


class Reply:
    def __init__(self, content="", tool_calls=None):
        self.content, self.tool_calls = content, tool_calls or []


class Chain:
    """Returns each queued reply in turn, and records how often it was asked."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, _messages):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


@pytest.mark.unit
class TestDetection:
    def test_the_real_failure_is_recognised(self):
        assert printed_a_tool_call(REAL_FAILURE, TOOLS) is True

    def test_a_genuine_report_is_not(self):
        report = ("## AAPL Technical Analysis\n\nThe 50 SMA sits at $308.12 and "
                  "RSI reads 61.4, so momentum is positive but not extended.")

        assert printed_a_tool_call(report, TOOLS) is False

    def test_a_report_that_merely_names_a_tool_is_not(self):
        """Prose about the data source is not a printed call. Requiring JSON
        structure as well as the name is what keeps this from firing on a
        perfectly good report."""
        report = "Data for this report came from get_stock_data over 30 sessions."

        assert printed_a_tool_call(report, TOOLS) is False

    def test_a_report_containing_a_json_table_is_not(self):
        """Structure without a tool name is not a printed call either."""
        report = '## Levels\n\n```json\n{"support": 305.0, "resistance": 318.0}\n```'

        assert printed_a_tool_call(report, TOOLS) is False

    def test_an_empty_answer_is_not(self):
        assert printed_a_tool_call("", TOOLS) is False
        assert printed_a_tool_call(None, TOOLS) is False


@pytest.mark.unit
class TestRecovery:
    def test_a_real_tool_call_is_returned_untouched(self):
        chain = Chain(Reply(tool_calls=[{"name": "get_stock_data"}]))

        result = invoke_with_tool_call_recovery(chain, [], TOOLS, "Market Analyst")

        assert chain.calls == 1 and result.tool_calls

    def test_a_finished_report_is_returned_untouched(self):
        """A real report arrives after tool results are in the history, which
        is what tells it apart from an answer composed from nothing."""
        chain = Chain(Reply(content="## AAPL\n\nRSI 61.4, trend intact."))
        history = [{"role": "tool", "content": "Close 313.45"}]

        result = invoke_with_tool_call_recovery(chain, history, TOOLS, "Market Analyst")

        assert chain.calls == 1 and "RSI" in result.content

    def test_a_printed_call_is_retried_once(self):
        chain = Chain(Reply(content=REAL_FAILURE),
                      Reply(tool_calls=[{"name": "get_stock_data"}]))

        result = invoke_with_tool_call_recovery(chain, [], TOOLS, "Market Analyst")

        assert chain.calls == 2
        assert result.tool_calls

    def test_it_retries_once_and_not_forever(self, caplog):
        """A model that does it twice is not going to do it right on the tenth
        try, and the run still has to finish."""
        chain = Chain(Reply(content=REAL_FAILURE))

        with caplog.at_level(logging.ERROR):
            invoke_with_tool_call_recovery(chain, [], TOOLS, "Market Analyst")

        assert chain.calls == 2
        assert "did it twice" in caplog.text

    def test_the_first_failure_is_logged(self, caplog):
        """The whole point: this used to happen in silence."""
        chain = Chain(Reply(content=REAL_FAILURE),
                      Reply(tool_calls=[{"name": "get_stock_data"}]))

        with caplog.at_level(logging.WARNING):
            invoke_with_tool_call_recovery(chain, [], TOOLS, "Market Analyst")

        assert "wrote its tool call as text" in caplog.text


@pytest.mark.unit
class TestAnsweredWithoutFetching:
    """The structural check, which catches what reading the text cannot.

    A model sometimes writes a long plan naming no tool and containing no JSON.
    By content that is indistinguishable from a finished report; by the
    conversation it is obvious, because nothing ever returned any data.
    """

    ESSAY = ("### Phase 1: Detailed Swing Trade Planning and Methodology\n\n"
             "A swing trade aims to capitalize on short-to-medium term price "
             "movements, often lasting a few days to several weeks.")

    def test_a_first_turn_answer_with_no_tool_call_is_ungrounded(self):
        chain = Chain(Reply(content=self.ESSAY),
                      Reply(tool_calls=[{"name": "get_stock_data"}]))

        result = invoke_with_tool_call_recovery(chain, [], TOOLS, "Market Analyst")

        assert chain.calls == 2 and result.tool_calls

    def test_an_analyst_that_fetched_and_found_nothing_is_left_alone(self):
        """The case this must not break. It has data — the data says there is
        none — and that is a real report, not an ungrounded one."""
        chain = Chain(Reply(content="No usable price data was returned for AAPL."))
        history = [{"role": "tool", "content": "DATA_UNAVAILABLE"}]

        result = invoke_with_tool_call_recovery(chain, history, TOOLS, "Market Analyst")

        assert chain.calls == 1
        assert "No usable price data" in result.content

    def test_langchain_message_objects_are_read_too(self):
        """State carries message objects, not dicts, so reading only .get()
        would make the check fire on every real report."""
        from langchain_core.messages import ToolMessage

        chain = Chain(Reply(content="## AAPL\n\nRSI 61.4."))
        history = [ToolMessage(content="Close 313.45", tool_call_id="t1")]

        result = invoke_with_tool_call_recovery(chain, history, TOOLS, "Market Analyst")

        assert chain.calls == 1
