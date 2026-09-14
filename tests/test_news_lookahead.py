"""yfinance news must not leak future-dated (or undated, in a backtest) articles
into a historical window.

Regressions for #992 (flat articles bypassed the date filter), #1007 (global
news injected future articles), #993 (empty-after-filter returned a blank body),
and #1126 (inclusive upper bound leaked the midnight-after article; host-local
timestamp parsing made filtering machine-dependent).
"""
from datetime import datetime, timezone

import pytest

import tradingagents.dataflows.yfinance_news as ynews
from tradingagents.dataflows.date_window import in_window


def _epoch(date_str):
    """Epoch seconds for UTC midnight of ``date_str`` (host-timezone independent)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992: flat articles now carry a pub_date (was always None -> unfilterable).
    # #1126: parsed as UTC-aware, so the date can't shift with the host timezone.
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].tzinfo is not None
    assert data["pub_date"] == datetime(2025, 5, 9, tzinfo=timezone.utc)


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert in_window(inside, start, end) is True
    assert in_window(future, start, end) is False     # look-ahead blocked
    assert in_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # Live window (reaches today): undated articles can't be "future", so keep them.
    now = datetime.now(timezone.utc)
    assert in_window(None, now, now) is True


@pytest.mark.unit
def test_upper_bound_is_exclusive():
    # #1126: an article stamped exactly midnight AFTER end_date leaked in under
    # the old inclusive bound; the whole of end_date itself must still be kept.
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    midnight_after = datetime(2025, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    last_moment = datetime(2025, 5, 9, 23, 59, 59, tzinfo=timezone.utc)
    assert in_window(midnight_after, start, end) is False
    assert in_window(last_moment, start, end) is True


@pytest.mark.unit
def test_offset_aware_timestamp_is_converted_not_truncated():
    # #1126: 2025-05-10T01:00+05:00 is really 2025-05-09T20:00Z -> inside the
    # window. Stripping tzinfo (old behavior) misread it as 05-10 and dropped it.
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)
    aware = datetime.fromisoformat("2025-05-10T01:00:00+05:00")
    assert in_window(aware, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007: a flat, future-dated global article must not appear in a historical run.
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993: everything filtered out -> a clear message, not a blank-bodied report.
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "###" not in out  # no empty article body


# --- every macro query must be represented -------------------------------------


def _query_bucket_search(per_query_titles):
    """A Search stub that returns different articles for each query, so a test
    can tell which queries actually ran."""

    class FakeSearch:
        def __init__(self, query=None, **kwargs):
            titles = per_query_titles.get(query, [])
            self.news = [
                {"title": t, "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2025-05-05")}
                for t in titles
            ]

    return FakeSearch


@pytest.mark.unit
def test_every_configured_query_is_fetched(monkeypatch):
    """The first query alone returns about as many articles as the limit, so a
    loop that fills one list and stops never reaches the later ones. With the
    shipped configuration that silently dropped geopolitics, central banks and
    commodities — macro news was Federal Reserve headlines under a broader
    name."""
    from tradingagents.dataflows.config import get_config

    queries = get_config()["global_news_queries"]
    monkeypatch.setattr(
        ynews.yf, "Search",
        _query_bucket_search({q: [f"{q[:12]} article {i}" for i in range(10)] for q in queries}),
    )

    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)

    for query in queries:
        assert query[:12] in out, f"nothing from {query!r} reached the report"


@pytest.mark.unit
def test_a_query_returning_nothing_does_not_cost_the_others_their_share(monkeypatch):
    """One of the shipped queries returns zero articles in practice. Its unused
    slots have to go to the topics that did return something."""
    monkeypatch.setattr(
        ynews.yf, "Search",
        _query_bucket_search({
            "a": ["a1", "a2", "a3", "a4"],
            "b": [],
            "c": ["c1", "c2", "c3", "c4"],
        }),
    )
    monkeypatch.setattr(
        ynews, "get_config", lambda: {"global_news_queries": ["a", "b", "c"],
                                      "global_news_lookback_days": 7,
                                      "global_news_article_limit": 6}
    )

    out = ynews.get_global_news_yfinance("2025-05-09")

    assert out.count("###") == 6


@pytest.mark.unit
def test_one_failing_query_does_not_lose_every_other_topic(monkeypatch):
    """A macro report missing commodities is worth more than no macro report."""
    def flaky(query=None, **kwargs):
        if query == "boom":
            raise RuntimeError("upstream is down")

        class Ok:
            news = [{"title": "GOOD", "publisher": "P", "link": "l",
                     "providerPublishTime": _epoch("2025-05-05")}]

        return Ok()

    monkeypatch.setattr(ynews.yf, "Search", flaky)
    monkeypatch.setattr(
        ynews, "get_config", lambda: {"global_news_queries": ["boom", "fine"],
                                      "global_news_lookback_days": 7,
                                      "global_news_article_limit": 10}
    )

    out = ynews.get_global_news_yfinance("2025-05-09")

    assert "GOOD" in out


@pytest.mark.unit
def test_out_of_window_articles_do_not_consume_a_slot(monkeypatch):
    """Truncating to the limit and filtering afterwards lets future-dated
    articles eat slots that in-window ones would have filled."""
    def search(query=None, **kwargs):
        class S:
            news = [
                {"title": "FUTURE", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2025-06-01")},
                {"title": "KEEPER", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2025-05-05")},
            ]

        return S()

    monkeypatch.setattr(ynews.yf, "Search", search)
    monkeypatch.setattr(
        ynews, "get_config", lambda: {"global_news_queries": ["only"],
                                      "global_news_lookback_days": 7,
                                      "global_news_article_limit": 1}
    )

    out = ynews.get_global_news_yfinance("2025-05-09")

    assert "KEEPER" in out
    assert "FUTURE" not in out


def _ticker_with(articles, monkeypatch):
    class FakeTicker:
        def __init__(self, *a, **k):
            pass

        def get_news(self, count=20):
            return articles

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)


@pytest.mark.unit
def test_ticker_news_window_before_feed_coverage_is_unavailable(monkeypatch):
    # Yahoo serves only recent articles: a historical window gets none of them,
    # which must read as "cannot answer", not "no news happened".
    recent = [{"title": "RECENT", "publisher": "P", "link": "l",
               "providerPublishTime": _epoch("2026-09-10")}]
    _ticker_with(recent, monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "RECENT" not in out
    assert "unavailable" in out and "not an absence" in out
    assert "2026-09-10" in out  # says how far back the feed actually reaches


@pytest.mark.unit
def test_ticker_news_covered_but_empty_window_is_a_real_absence(monkeypatch):
    articles = [{"title": "RECENT", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2026-09-10")},
                {"title": "OLDER", "publisher": "P", "link": "l",
                 "providerPublishTime": _epoch("2026-07-01")}]
    _ticker_with(articles, monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "No news found" in out
    assert "unavailable" not in out


@pytest.mark.unit
@pytest.mark.parametrize("dates, expect_gap", [
    ([], True),                                              # empty feed: covers at most now
    ([None], True),                                          # undated only: same
    ([datetime(2026, 5, 20, tzinfo=timezone.utc)], True),    # all after the window
    ([datetime(2026, 5, 4, tzinfo=timezone.utc)], True),     # starts mid-window: partial
    ([datetime(2026, 5, 1, 18, tzinfo=timezone.utc)], False),  # reaches the first day
    ([datetime(2026, 5, 20, tzinfo=timezone.utc),
      datetime(2026, 4, 1, tzinfo=timezone.utc)], False),    # coverage reaches back
])
def test_coverage_gap_boundaries(dates, expect_gap):
    from tradingagents.dataflows.date_window import coverage_gap

    out = coverage_gap(dates, "2026-05-01", "2026-05-08", "Feed", "items")
    assert (out is not None) is expect_gap
    if expect_gap:
        assert "unavailable for 2026-05-01..2026-05-08" in out and "not an absence" in out



@pytest.mark.unit
def test_ticker_news_empty_feed_for_a_past_window_is_unavailable(monkeypatch):
    _ticker_with([], monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_ticker_news_null_feed_is_handled(monkeypatch):
    # Yahoo can return None instead of a list; that is unavailability, not an error.
    _ticker_with(None, monkeypatch)
    out = ynews.get_news_yfinance("AAPL", "2026-08-07", "2026-08-14")
    assert "unavailable" in out and "Error" not in out


@pytest.mark.unit
def test_global_news_empty_feed_for_a_past_window_is_unavailable(monkeypatch):
    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = []

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_global_news_does_not_infer_coverage_from_a_stale_search_hit(monkeypatch):
    # Global news merges fuzzy searches; one old hit before the window says
    # nothing about the days in between, so the window stays unavailable.
    stale = {"title": "STALE", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-01-01")}
    fresh = {"title": "FRESH", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [fresh, stale]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "unavailable" in out and "No global news found" not in out


@pytest.mark.unit
def test_coverage_gap_future_window_is_unavailable():
    from datetime import timedelta

    from tradingagents.dataflows.date_window import coverage_gap
    today = datetime.now(timezone.utc).date()
    out = coverage_gap([], str(today), str(today + timedelta(days=3)), "Feed", "items")
    assert out is not None and "past today" in out
