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
                    # Upstream's grounding says where the numbers come from and
                    # is conditional on the market report existing. Ours says
                    # what the proposal must contain. Both apply.
                    + grounding
                    + "Anchor your reasoning in the analysts' reports and the research plan. "
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
