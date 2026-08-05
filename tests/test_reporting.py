"""Report parity: the shared writer produces the report tree for the CLI and the
programmatic API alike (#1037)."""

from types import SimpleNamespace

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import extract_screen_summary, write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")


@pytest.mark.unit
def test_extract_screen_summary_parses_structured_fields():
    final_state = {
        "final_trade_decision": (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Strong momentum.\n\n"
            "**Investment Thesis**: Blah blah.\n\n"
            "**Price Target**: 215.5\n\n"
            "**Time Horizon**: 3-6 months"
        ),
        "sentiment_report": (
            "**Overall Sentiment:** **Bullish** (Score: 7.2/10)\n"
            "**Confidence:** High\n\n"
            "Narrative text here."
        ),
    }
    summary = extract_screen_summary(final_state)
    assert summary["direction"] == "Buy"
    assert summary["sentiment_band"] == "Bullish"
    assert summary["sentiment_score"] == 7.2
    assert summary["price_target"] == "215.5"
    assert summary["time_horizon"] == "3-6 months"


@pytest.mark.unit
def test_extract_screen_summary_missing_fields_are_none():
    summary = extract_screen_summary({})
    assert summary == {
        "direction": None,
        "sentiment_band": None,
        "sentiment_score": None,
        "price_target": None,
        "time_horizon": None,
    }
