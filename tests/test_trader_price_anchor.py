"""The trader must be given a price before it is asked for price levels.

The trader is asked for entry / stop-loss / target so a risk/reward ratio can
be computed, but its prompt contained only the research manager's prose plan —
no price anywhere. A small model fills those numeric fields from memory
instead, and what it remembers is the ticker's price during training. Observed
in production on 2026-08-06:

    GOOG at $356.62  ->  entry $2,000.00  (roughly its pre-2022-split price)
    VERI at $1.26    ->  entry $4.50
    VERI at $1.25    ->  entry $30.00     (roughly its 2021 range)

The reasoning around those numbers was correct and ticker-specific, so nothing
else was wrong. The model simply never saw a price.
"""
from unittest.mock import patch

from tradingagents.agents.trader import trader as trader_module

SNAPSHOT = "## Verified market data snapshot for VERI\n\n| Close | 1.25 |\n"


def _capture_prompt(state, snapshot=SNAPSHOT, snapshot_error=None):
    """Build the trader node and return the messages it would send."""
    captured = {}

    def fake_invoke(*args, **kwargs):
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], dict) and "role" in arg[0]:
                captured["messages"] = arg
        raise RuntimeError("stop before the model call")

    def fake_snapshot(*args, **kwargs):
        if snapshot_error is not None:
            raise snapshot_error
        return snapshot

    with patch.object(trader_module, "bind_structured", lambda *a, **k: object()), \
         patch.object(trader_module, "invoke_structured_or_freetext", fake_invoke), \
         patch.object(trader_module, "build_verified_market_snapshot", fake_snapshot):
        node = trader_module.create_trader(object())
        try:
            node(state)
        except RuntimeError:
            pass
    return captured["messages"]


def _state(**overrides):
    base = {
        "company_of_interest": "VERI",
        "trade_date": "2026-08-06",
        "investment_plan": "The research manager's plan.",
        "horizon": "swing",
        "messages": [],
    }
    base.update(overrides)
    return base


class TestTraderPriceAnchor:
    def test_the_verified_snapshot_reaches_the_prompt(self):
        messages = _capture_prompt(_state())
        assert SNAPSHOT in messages[1]["content"]

    def test_the_instruction_names_the_snapshot_as_the_price_source(self):
        system = _capture_prompt(_state())[0]["content"]
        assert "verified market snapshot" in system
        assert "from memory" in system

    def test_the_instruction_says_to_omit_rather_than_guess(self):
        # Guessing is the failure mode; an absent level costs far less than a
        # fabricated one, which arms a stop alert at a price that never comes.
        system = _capture_prompt(_state())[0]["content"]
        assert "omit the prices rather than guessing" in system

    def test_a_failed_snapshot_does_not_sink_the_run(self):
        # Market data can be unavailable. The trade plan degrades; the analysis
        # still completes and the decision is still recorded.
        messages = _capture_prompt(_state(), snapshot_error=RuntimeError("no data"))
        assert "The research manager's plan." in messages[1]["content"]
        assert "Verified market data snapshot" not in messages[1]["content"]

    def test_the_research_plan_is_still_included(self):
        messages = _capture_prompt(_state())
        assert "The research manager's plan." in messages[1]["content"]
