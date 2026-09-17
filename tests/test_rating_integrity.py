"""A decision is recorded as the call that was made, or as needing review.

Two readers used to disagree about the same text: the signal said REVIEW while
the memory log wrote a fabricated Hold. Worse, prose that argued against a Buy
before concluding Underweight was read as Buy, because the parser took the first
rating word anywhere in the document. A wrong direction is worse than no
direction, so an unclear decision is REVIEW everywhere.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.rating import RATING_REVIEW, extract_rating, parse_rating

INVERTED = ("The aggressive analyst pushed hard for a Buy on the AI backlog, but the "
            "conservative case on margin compression carried the debate. "
            "Final rating — Underweight. Trim to half weight over the next two weeks.")
REFUSAL = "I'm sorry, I can't provide a rating for this security."


@pytest.mark.unit
@pytest.mark.parametrize("separator", [":", "-", "—", "–", "：", ": **"])
def test_the_labelled_rating_wins_whatever_separates_it(separator):
    text = f"Buy arguments were raised and rejected.\n\nRating{separator}Underweight\n\nTrim."
    assert extract_rating(text) == "Underweight"


@pytest.mark.unit
def test_a_rating_argued_against_is_not_read_as_the_decision():
    assert extract_rating(INVERTED) == "Underweight"


@pytest.mark.unit
def test_prose_naming_several_ratings_without_a_label_needs_review():
    """Nothing in the text says which one is the call, so guessing risks
    reporting the opposite of the decision."""
    text = "The bull wants Buy, the bear wants Sell, and the committee was split."
    assert extract_rating(text) is None


@pytest.mark.unit
def test_prose_naming_one_rating_is_taken_as_the_call():
    assert extract_rating("On balance we stay Underweight until margins recover.") == "Underweight"


@pytest.mark.unit
def test_a_refusal_has_no_rating_and_is_not_defaulted():
    assert extract_rating(REFUSAL) is None
    assert parse_rating(REFUSAL) == RATING_REVIEW


@pytest.mark.unit
def test_the_scale_quoted_in_a_prompt_does_not_become_the_rating():
    """A free-text answer that echoes the rating scale was read as the first
    tier listed in it."""
    text = ("**Rating Scale**: Buy, Overweight, Hold, Underweight, Sell.\n\n"
            "**Rating**: Sell\n\nExit the position.")
    assert extract_rating(text) == "Sell"


# --- the readers agree ------------------------------------------------------

@pytest.mark.unit
def test_the_memory_log_records_review_rather_than_a_tradeable_hold(tmp_path):
    from tradingagents.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    log.store_decision("NVDA", "2026-01-05", REFUSAL)

    entry = log.load_entries()[0]
    assert entry["rating"] == RATING_REVIEW


@pytest.mark.unit
def test_the_signal_and_the_log_agree_on_the_same_decision(tmp_path):
    from tradingagents.agents.utils.memory import TradingMemoryLog
    from tradingagents.graph.signal_processing import SignalProcessor

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    for text in (INVERTED, REFUSAL, "**Rating**: Buy\n\nAccumulate."):
        log.store_decision("NVDA", f"2026-01-0{len(log.load_entries()) + 1}", text)

    signals = [SignalProcessor.process_signal(None, text)
               for text in (INVERTED, REFUSAL, "**Rating**: Buy\n\nAccumulate.")]
    assert [e["rating"] for e in log.load_entries()] == signals


# Two upstream tests are not here: one grades a backtest run and one drives the
# CLI. This fork ships neither surface — the app calls the graph directly, and
# the backtest module was left upstream until a backtester is built here.
