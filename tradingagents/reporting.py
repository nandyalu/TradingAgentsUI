"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.rating import parse_rating

# Matches the deterministic header render_sentiment_report() prepends to
# sentiment_report, e.g. "**Overall Sentiment:** **Bullish** (Score: 7.2/10)".
_SENTIMENT_HEADER_RE = re.compile(
    r"\*\*Overall Sentiment:\*\*\s*\*\*([^*]+)\*\*\s*\(Score:\s*([\d.]+)/10\)"
)
# Matches render_pm_decision()'s optional lines, e.g.
# "**Price Target**: 215.5" / "**Time Horizon**: 3-6 months".
_PRICE_TARGET_RE = re.compile(r"\*\*Price Target\*\*:\s*([^\n]+)")
_TIME_HORIZON_RE = re.compile(r"\*\*Time Horizon\*\*:\s*([^\n]+)")


def extract_screen_summary(final_state: dict) -> dict[str, Any]:
    """Pull a compact, structured summary out of a completed run's final
    state — the fields a screener result table wants (direction, sentiment,
    price target / time horizon) without re-parsing prose by hand.

    Reads fields TradingAgents' agents already produce as structured output
    (``PortfolioDecision.price_target``/``time_horizon``, the Sentiment
    Analyst's ``SentimentReport`` header) rather than asking any agent for a
    new "6-month potential" field — those already exist, just rendered to
    markdown for the saved report. Every field is best-effort: missing or
    unparseable input yields ``None``, never an exception, so a screener run
    on many tickers isn't derailed by one odd report.
    """
    empty = {
        "direction": None,
        "sentiment_band": None,
        "sentiment_score": None,
        "price_target": None,
        "time_horizon": None,
    }
    if not isinstance(final_state, dict):
        return empty

    final_trade_decision = final_state.get("final_trade_decision") or ""
    sentiment_report = final_state.get("sentiment_report") or ""

    direction = parse_rating(final_trade_decision) if final_trade_decision else None

    sentiment_match = _SENTIMENT_HEADER_RE.search(sentiment_report)
    sentiment_band = sentiment_match.group(1).strip() if sentiment_match else None
    sentiment_score = None
    if sentiment_match:
        try:
            sentiment_score = float(sentiment_match.group(2))
        except ValueError:
            pass

    price_target_match = _PRICE_TARGET_RE.search(final_trade_decision)
    time_horizon_match = _TIME_HORIZON_RE.search(final_trade_decision)

    return {
        "direction": direction,
        "sentiment_band": sentiment_band,
        "sentiment_score": sentiment_score,
        "price_target": price_target_match.group(1).strip() if price_target_match else None,
        "time_horizon": time_horizon_match.group(1).strip() if time_horizon_match else None,
    }


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
