"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import (
    TraderProposal,
    render_trader_proposal,
    resolve_levels,
)
from tradingagents.dataflows.market_data_validator import (
    build_verified_market_snapshot,
    verified_levels_basis,
)
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

        # The same figures as numbers rather than markdown. The model reads the
        # snapshot to reason; Python reads this to compute the levels, so the
        # prices in the proposal are arithmetic rather than recall.
        # Guarded like the snapshot above it. Direct indexing raised KeyError on
        # a state without trade_date, which turned a missing optional into a
        # dead run.
        basis = None
        try:
            basis = verified_levels_basis(company_name, state["trade_date"])
        except Exception:  # noqa: BLE001 — a missing basis must not sink the run
            pass
        if basis is None:
            logger.warning(
                "Trader: no verified close/ATR for %s; the proposal will carry no levels",
                company_name,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Always argue BOTH sides explicitly — a bull case (arguments for) and a bear "
                    "case (arguments against) — then commit to a win probability.\n\n"
                    "You do NOT give prices. When taking a Buy or Sell, say how much room the "
                    "trade needs as two distances: stop_atr_multiple, how far the stop sits from "
                    "the entry counted in ATRs, and target_r_multiple, how much the trade aims to "
                    "make as a multiple of what it risks. A swing trade typically stops 1.5 to 3 "
                    "ATRs away and targets 1.5 to 3 times its risk. Choose them from the "
                    "volatility and the structure you see in the verified snapshot below: a "
                    "choppy chart needs a wider stop than a trending one.\n\n"
                    "The entry, stop and target prices are computed from those two numbers and "
                    "the snapshot's verified close and ATR. That is deliberate. Every price you "
                    "might recall for this ticker is from the wrong year, so there is no field "
                    "here for you to put one in."
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
            # The renderer needs the basis to turn the proposal's multiples into
            # prices, and invoke_structured_or_freetext passes it only the
            # proposal, so bind the basis here.
            lambda proposal: render_trader_proposal(
                proposal, resolve_levels(proposal, basis)
            ),
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
