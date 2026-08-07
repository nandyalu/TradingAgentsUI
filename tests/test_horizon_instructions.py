"""Prompt guidance keyed to the run's trade horizon.

``get_horizon_instruction`` reaches the three decision-making stages (research
manager, trader, portfolio manager). ``get_indicator_instruction`` reaches only
the market analyst, whose indicator choice depends directly on the holding
period in a way the other analysts' prose does not.
"""
import pytest

from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_indicator_instruction,
)


class TestHorizonInstruction:
    def test_swing_states_a_one_to_two_week_window(self):
        text = get_horizon_instruction({"horizon": "swing"})
        assert "SWING" in text
        assert "1 to 2 weeks" in text
        assert "5 to 10 trading days" in text

    def test_swing_rules_out_a_six_month_thesis(self):
        text = get_horizon_instruction({"horizon": "swing"})
        assert "6-month thesis is out of scope" in text
        assert "never in months" in text

    def test_position_still_asks_for_a_multi_month_hold(self):
        text = get_horizon_instruction({"horizon": "position"})
        assert "POSITION" in text
        assert "multi-month" in text

    @pytest.mark.parametrize("state", [{}, None, {"horizon": None}, "not-a-mapping"])
    def test_defaults_to_position(self, state):
        assert "POSITION" in get_horizon_instruction(state)

    def test_case_and_whitespace_tolerated(self):
        assert "SWING" in get_horizon_instruction({"horizon": "  Swing  "})


class TestIndicatorInstruction:
    def test_position_adds_nothing(self):
        # The base market-analyst prompt already suits a multi-month hold, so
        # the default path must not spend tokens repeating it.
        assert get_indicator_instruction({"horizon": "position"}) == ""
        assert get_indicator_instruction({}) == ""
        assert get_indicator_instruction(None) == ""

    def test_swing_names_the_fast_indicators(self):
        text = get_indicator_instruction({"horizon": "swing"})
        for indicator in ("close_10_ema", "macd", "rsi", "boll", "atr", "vwma", "mfi"):
            assert indicator in text

    def test_swing_demotes_the_two_hundred_day_average(self):
        # The base prompt lists close_200_sma first and calls it a long-term
        # trend benchmark; over 10 trading days it carries almost no signal.
        text = get_indicator_instruction({"horizon": "swing"})
        assert "do not build the thesis on close_200_sma" in text

    def test_swing_asks_for_reachable_levels(self):
        text = get_indicator_instruction({"horizon": "swing"})
        assert "support and resistance" in text
        assert "10 to 20 sessions" in text
