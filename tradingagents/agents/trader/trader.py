"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        # The trader is asked for entry / stop / target prices but was given no
        # price to anchor them to — only the research manager's prose plan. A
        # small model fills those numeric fields from memory instead, and what
        # it remembers is the ticker's price during training: $2,000 for GOOG
        # (pre-split), $30 for VERI (its 2021 range), on stocks trading at $357
        # and $1.26. The reasoning around them was correct and specific, so
        # nothing else was wrong — the model simply never saw a price.
        #
        # The snapshot is computed in Python from the same OHLCV the analysts
        # used, never by a model, so it cannot itself be hallucinated.
        snapshot = ""
        try:
            snapshot = build_verified_market_snapshot(company_name, state["trade_date"])
        except Exception:  # noqa: BLE001 — a missing snapshot must not sink the run
            logger.warning("Trader: no verified snapshot for %s; levels will be unanchored", company_name)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Always argue BOTH sides explicitly — a bull case (arguments for) and a bear "
                    "case (arguments against) — then commit to a win probability, and when taking a "
                    "Buy/Sell give entry / stop-loss / target prices so the risk/reward ratio can "
                    "be computed.\n\n"
                    "CRITICAL — the entry, stop-loss, and target MUST be derived from the verified "
                    "market snapshot below, which is the only trustworthy price source in this "
                    "conversation. Read the latest close from it and place every level within a "
                    "few percent of that number. Do NOT use a price you recall for this ticker "
                    "from memory; it will be from the wrong year and the whole proposal will be "
                    "discarded. If the snapshot is missing, omit the prices rather than guessing."
                    + NO_EXTERNAL_TOOLS
                    + get_horizon_instruction(state)
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"{snapshot}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
