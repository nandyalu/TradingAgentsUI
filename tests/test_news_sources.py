"""The extra news sources and the laya grade on each item."""
import pytest

from tradingagents.dataflows import news_sources


@pytest.fixture
def graded(monkeypatch):
    """laya is on, and grades by keyword: a text naming ACME mentions it."""
    monkeypatch.setenv("LAYA_URL", "http://laya.test")
    monkeypatch.setattr(news_sources, "company_name", lambda ticker: "Acme")

    def ask(state, questions, model=None):
        text = state["text"]
        if "mentions" in questions:
            return {"mentions": {"noul": 0.9 if "ACME" in text.upper() else 0.05}}
        return {"tone": {"score": 4.0 if "up" in text else 0.0}}

    monkeypatch.setattr(news_sources.laya, "ask", ask)


YAHOO = """## ACME News, from 2026-09-19 to 2026-09-26:

### Acme shares up on contract (source: Reuters)
A long summary line.
Link: https://example.com/a

### Fed holds rates (source: AP)
"""

STOCKTWITS = """Bullish: 1 (50%) · Total: 2 most-recent messages

[2026-09-26 · @a · Bullish] $ACME going up
[2026-09-26 · @b · no-label] nothing here"""


def test_each_item_gets_its_grade_at_its_start(graded):
    out = news_sources.annotate(YAHOO, "ACME")
    assert "### [mentions 0.90 · tone +1.00] Acme shares up on contract (source: Reuters)" in out
    assert "### [mentions 0.05 · tone -1.00] Fed holds rates (source: AP)" in out
    assert "A long summary line." in out  # the rest of the item is kept


def test_a_post_line_keeps_its_own_brackets(graded):
    out = news_sources.annotate(STOCKTWITS, "ACME")
    assert "[mentions 0.90 · tone +1.00] [2026-09-26 · @a · Bullish] $ACME going up" in out
    assert out.startswith("Bullish: 1 (50%)")  # a summary line is not an item


def test_nothing_is_dropped_unless_asked(graded, monkeypatch):
    assert "Fed holds rates" in news_sources.annotate(YAHOO, "ACME")
    monkeypatch.setenv("LAYA_DROP_BELOW", "0.5")
    out = news_sources.annotate(YAHOO, "ACME")
    assert "Fed holds rates" not in out
    assert "Acme shares up" in out
    assert out.endswith("(1 item(s) left out: the classifier found no mention of ACME.)")


def test_without_laya_the_block_is_unchanged(monkeypatch):
    monkeypatch.delenv("LAYA_URL", raising=False)
    assert news_sources.annotate(YAHOO, "ACME") == YAHOO


def test_a_laya_failure_leaves_the_block_unchanged(graded, monkeypatch):
    monkeypatch.setattr(news_sources.laya, "ask", lambda *a, **k: None)
    assert news_sources.annotate(YAHOO, "ACME") == YAHOO


def test_the_legend_comes_once_and_only_with_a_grade(graded, monkeypatch):
    for f in ("google_news", "finnhub_news", "sec_8k"):
        monkeypatch.setattr(news_sources, f, lambda t, s, e: "### Acme item (source: X)")
    news_sources._cache.clear()
    out = news_sources.with_extra(YAHOO, "ACME", "2026-09-19", "2026-09-26")
    assert out.count(news_sources.GRADE_LEGEND) == 1
    assert out.startswith(news_sources.GRADE_LEGEND)
    assert news_sources.with_legend("no grades here") == "no grades here"


def test_an_8k_names_what_its_items_report(monkeypatch):
    from tradingagents.dataflows import sec_edgar

    monkeypatch.setattr(sec_edgar, "cik_for", lambda t: "0000000001")
    monkeypatch.setattr(sec_edgar, "_fetch_json", lambda url: {"filings": {"recent": {
        "form": ["8-K", "10-Q", "8-K"],
        "filingDate": ["2026-09-24", "2026-09-20", "2026-08-01"],
        "items": ["2.02,9.01", "", "5.02"],
    }}})
    out = news_sources.sec_8k("ACME", "2026-09-19", "2026-09-26")
    assert "### 8-K filed 2026-09-24: results of operations (earnings); financial statements and exhibits" in out
    assert "10-Q" not in out and "2026-08-01" not in out


def test_finnhub_without_a_key_says_nothing(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert news_sources.finnhub_news("ACME", "2026-09-19", "2026-09-26") == ""


def test_a_company_suffix_is_cut_for_the_search(monkeypatch):
    import yfinance

    news_sources.company_name.cache_clear()
    monkeypatch.setattr(yfinance, "Ticker", lambda t: type("T", (), {"info": {"shortName": "NVIDIA Corporation"}})())
    assert news_sources.company_name("NVDA") == "NVIDIA"
    news_sources.company_name.cache_clear()
