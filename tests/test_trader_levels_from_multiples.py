"""Levels computed in Python from the trader's multiples, never typed by it.

The trader used to be asked for entry, stop and target prices. A small model
filled those fields from what it remembered of the ticker — $2,000 for GOOG on
a day it traded at $357 — and prompting against it helped without stopping it.
The schema now offers no price field at all, so there is nothing to fabricate
into: the model states two distances and this arithmetic turns them into levels
from the verified close and ATR.
"""
import pytest

from tradingagents.agents.schemas import TraderAction, TraderProposal, resolve_levels

BASIS = {"close": 100.0, "atr": 5.0}


def _proposal(action=TraderAction.BUY, stop=2.0, target=3.0):
    return TraderProposal(
        action=action, reasoning="r", bull_case="b", bear_case="c",
        win_probability=60, stop_atr_multiple=stop, target_r_multiple=target,
    )


@pytest.mark.unit
class TestResolveLevels:
    def test_the_model_is_never_shown_a_price_field(self):
        """The whole point. A field that does not exist cannot be filled from
        memory, which is stronger than any instruction not to."""
        shown = set(TraderProposal.model_json_schema()["properties"])

        assert not shown & {"entry_price", "stop_loss", "target_price"}
        assert {"stop_atr_multiple", "target_r_multiple"} <= shown

    def test_a_long_stops_below_and_targets_above(self):
        levels = resolve_levels(_proposal(), BASIS)

        assert levels == {"entry_price": 100.0, "stop_loss": 90.0, "target_price": 130.0}

    def test_a_short_stops_above_and_targets_below(self):
        """Getting the direction backwards would store a stop that triggers the
        instant it is placed."""
        levels = resolve_levels(_proposal(action=TraderAction.SELL), BASIS)

        assert levels == {"entry_price": 100.0, "stop_loss": 110.0, "target_price": 70.0}

    def test_the_r_multiple_measures_reward_against_risk(self):
        levels = resolve_levels(_proposal(stop=1.0, target=2.0), BASIS)
        risk = levels["entry_price"] - levels["stop_loss"]
        reward = levels["target_price"] - levels["entry_price"]

        assert reward / risk == pytest.approx(2.0)

    def test_hold_gets_no_levels(self):
        """There is no trade to place them around."""
        levels = resolve_levels(_proposal(action=TraderAction.HOLD), BASIS)

        assert levels == {"entry_price": None, "stop_loss": None, "target_price": None}

    def test_no_basis_means_no_levels_rather_than_guessed_ones(self):
        assert resolve_levels(_proposal(), None)["entry_price"] is None

    def test_missing_multiples_mean_no_levels(self):
        assert resolve_levels(_proposal(stop=None), BASIS)["stop_loss"] is None
        assert resolve_levels(_proposal(target=None), BASIS)["target_price"] is None

    def test_a_stop_wider_than_the_price_is_refused(self):
        """It would put the level at or below zero, which is not a stop."""
        levels = resolve_levels(_proposal(stop=10.0), {"close": 20.0, "atr": 5.0})

        assert levels["stop_loss"] is None

    def test_an_implausible_multiple_costs_the_levels_and_not_the_proposal(self):
        """A run answered 10.75 ATRs. The schema used to cap at 10, so Pydantic
        rejected the whole proposal, the structured call fell back to free text,
        and the reasoning and win probability went out with the one number that
        was unusable. The bound now sits in resolve_levels instead."""
        wide = _proposal(stop=10.75)

        assert wide.stop_atr_multiple == 10.75          # the proposal survives
        assert resolve_levels(wide, BASIS)["stop_loss"] is None   # the level does not

    def test_an_implausible_target_is_refused_the_same_way(self):
        far = _proposal(target=25.0)

        assert far.target_r_multiple == 25.0
        assert resolve_levels(far, BASIS)["target_price"] is None

    def test_the_schema_still_rejects_the_wrong_kind_of_number(self):
        """Loose is not absent. A negative or absurd multiple is not a bad plan,
        it is the wrong kind of thing."""
        with pytest.raises(Exception):
            _proposal(stop=-1.0)
        with pytest.raises(Exception):
            _proposal(target=500.0)
