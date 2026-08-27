"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    bull_case: str = Field(
        description=(
            "Bull case: the 2-4 strongest arguments FOR upside / taking this "
            "position, each grounded in specific evidence (fundamentals, "
            "technicals, sentiment, catalysts)."
        ),
    )
    bear_case: str = Field(
        description=(
            "Bear case: the 2-4 strongest arguments AGAINST the position / "
            "pointing to downside, each grounded in specific evidence. Be "
            "genuinely critical; do not strawman the opposing view."
        ),
    )
    win_probability: float = Field(
        ge=0.0,
        le=100.0,
        description=(
            "Win probability (0-100): your estimated likelihood that the "
            "directional thesis plays out, weighing the bull case against the "
            "bear case. Do not default to 50 — commit to a considered estimate."
        ),
    )
    # **The model states distances, never prices.** It used to be asked for
    # entry, stop and target in the quote currency, and a small model filled
    # those fields from what it remembered of the ticker: $2,000 for GOOG on a
    # day it traded at $357, $30 for VERI at $1.26. Prompting against it helped
    # and did not stop it, because a model that can type a number can type the
    # wrong one.
    #
    # A multiple cannot be a remembered price. Python turns these into levels
    # from the verified close and ATR, so the arithmetic is checkable and the
    # only thing the model decides is how much room to give the trade — which
    # is the judgement worth having from it.
    stop_atr_multiple: float | None = Field(
        default=None,
        ge=0.25,
        le=10.0,
        description=(
            "How far the stop sits from the entry, counted in ATRs (average "
            "true range). Typical swing trades use 1.5 to 3. Smaller means a "
            "tighter stop that ordinary noise may trigger; larger risks more "
            "per share. Give this whenever the action is Buy or Sell. Do NOT "
            "give a price — the exact level is computed from the verified "
            "snapshot."
        ),
    )
    target_r_multiple: float | None = Field(
        default=None,
        ge=0.25,
        le=20.0,
        description=(
            "How much the trade aims to make, as a multiple of what it risks. "
            "2 means the target is twice as far from entry as the stop is, so "
            "the risk/reward is 2:1. Give this whenever the action is Buy or "
            "Sell. Do NOT give a price."
        ),
    )


    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("stop_atr_multiple", "target_r_multiple", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def resolve_levels(proposal: "TraderProposal", basis: dict | None) -> dict:
    """Turn the proposal's multiples into entry, stop and target prices.

    Python owns this arithmetic so the levels cannot be recalled from training.
    ``basis`` is ``{"close", "atr"}`` from ``verified_levels_basis`` — computed
    from the same OHLCV the analysts read.

    Returns all-``None`` when there is no basis or no multiples, which is the
    honest answer: a plan with no levels, rather than levels nobody can defend.
    Hold takes no levels either, since there is no trade to place them around.
    """
    empty = {"entry_price": None, "stop_loss": None, "target_price": None}
    if basis is None or proposal.action is TraderAction.HOLD:
        return empty
    stop_mult, target_mult = proposal.stop_atr_multiple, proposal.target_r_multiple
    if stop_mult is None or target_mult is None:
        return empty

    close, atr = basis["close"], basis["atr"]
    risk = stop_mult * atr
    # A stop wider than the price itself would put the level at or below zero.
    if risk >= close:
        return empty
    # Direction follows the action: a long stops below and targets above, a
    # short does the reverse. Getting this backwards would store a stop that
    # triggers the instant it is placed, which is the failure
    # _levels_on_the_wrong_side exists to catch downstream.
    sign = -1.0 if proposal.action is TraderAction.BUY else 1.0
    stop = close + sign * risk
    target = close - sign * risk * target_mult
    if target <= 0:
        return empty
    return {
        "entry_price": round(close, 2),
        "stop_loss": round(stop, 2),
        "target_price": round(target, 2),
    }


def _render_trade_review(proposal: TraderProposal, levels: dict) -> str:
    """Probability + risk/reward + expected-value review.

    The win probability comes from the model. Everything else — the levels in
    ``levels``, the risk/reward ratio, the expected value in R-multiples and
    the breakeven win-rate — is computed in Python, so the numbers stay
    arithmetically consistent and none of them can be recalled from training.
    """
    lines = ["### Probability & Risk/Reward"]
    prob = proposal.win_probability
    lines.append(f"- **Win Probability**: {prob:.0f}%")

    rr = None
    if (
        levels["entry_price"] is not None
        and levels["stop_loss"] is not None
        and levels["target_price"] is not None
    ):
        reward = abs(levels["target_price"] - levels["entry_price"])
        risk = abs(levels["entry_price"] - levels["stop_loss"])
        if risk > 0:
            rr = reward / risk
            lines.append(
                f"- **Risk/Reward Ratio**: {rr:.2f} : 1 "
                f"(potential +{reward:.2f} vs risk -{risk:.2f})"
            )

    if rr is not None:
        p = prob / 100.0
        ev_r = p * rr - (1.0 - p)  # expected value in units of risk (R-multiple)
        breakeven = 100.0 / (1.0 + rr)
        verdict = "favorable" if ev_r > 0 else "unfavorable"
        lines.append(f"- **Expected Value**: {ev_r:+.2f}R ({verdict})")
        lines.append(
            f"- **Breakeven Win-Rate**: {breakeven:.0f}% "
            f"(current {prob:.0f}% is {'above' if prob > breakeven else 'below'} breakeven)"
        )
    else:
        lines.append(
            "- **Risk/Reward Ratio**: n/a (needs entry / stop / target prices)"
        )
    return "\n".join(lines)


def render_trader_proposal(proposal: TraderProposal, levels: dict | None = None) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
        "",
        f"**Bull Case**: {proposal.bull_case}",
        "",
        f"**Bear Case**: {proposal.bear_case}",
    ]
    # The rendered shape is unchanged on purpose. Downstream consumers parse
    # these lines out of the markdown, so moving who computes the number must
    # not move where it appears.
    levels = levels or {"entry_price": None, "stop_loss": None, "target_price": None}
    if levels["entry_price"] is not None:
        parts.extend(["", f"**Entry Price**: {levels['entry_price']}"])
    if levels["stop_loss"] is not None:
        parts.extend(["", f"**Stop Loss**: {levels['stop_loss']}"])
    if levels["target_price"] is not None:
        parts.extend(["", f"**Target Price**: {levels['target_price']}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend(["", _render_trade_review(proposal, levels)])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
