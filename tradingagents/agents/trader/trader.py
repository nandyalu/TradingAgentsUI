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
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

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
                    # Upstream's grounding says where the numbers come from and
                    # is conditional on the market report existing. Ours says
                    # what the proposal must contain. Both apply.
                    + grounding
                    + "Anchor your reasoning in the analysts' reports and the research plan. "
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
                    # Two different things, and the trader needs both. The
                    # report section is the market analyst's prose. The snapshot
                    # is computed in Python from the same OHLCV — a verified
                    # close and ATR that no model wrote — and it is what stops
                    # the trader reaching for a price it remembers.
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    f"{snapshot}\n\n"
                    f"Make an informed, strategic trading decision."
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
