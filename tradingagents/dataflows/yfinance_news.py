"""yfinance-based news data fetching functions."""

import contextlib
from datetime import datetime, timedelta, timezone

import yfinance as yf
from dateutil.relativedelta import relativedelta

from .config import get_config
from .stockstats_utils import yf_retry
from .symbol_utils import normalize_symbol


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; a naive value is assumed to be UTC.

    Window bounds arrive naive (parsed from ``yyyy-mm-dd``) while article
    timestamps may be offset-aware, so every operand is normalized before
    comparison. Without this the filter depends on the host timezone (#1126).
    """
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    # Handle nested content structure
    if "content" in article:
        content = article["content"]
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")

        # Get URL from canonicalUrl or clickThroughUrl
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "")

        # Get publish date
        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            with contextlib.suppress(ValueError, AttributeError):
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))

        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "link": link,
            "pub_date": pub_date,
        }
    else:
        # Fallback for flat structure. Parse the epoch publish time so flat
        # articles are date-filterable too (otherwise they bypass the
        # historical window and leak future news, #992/#1007).
        pub_date = None
        ts = article.get("providerPublishTime")
        if ts:
            # Epoch seconds are UTC; parse them as UTC-aware so filtering does
            # not shift with the host timezone (#1126).
            with contextlib.suppress(ValueError, OSError, TypeError):
                pub_date = datetime.fromtimestamp(ts, tz=timezone.utc)
        return {
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "pub_date": pub_date,
        }


def _in_news_window(pub_date, start_dt, end_dt) -> bool:
    """Whether an article belongs in the half-open window ``[start, end + 1 day)``.

    Every operand is normalized to UTC, and the upper bound is exclusive so an
    article stamped exactly at midnight after ``end_dt`` cannot leak into a
    historical run (#1126). An undated article is kept only when the window
    reaches the present (live run) — in a historical/backtest window it's
    excluded, since we can't prove it isn't future news (#992/#1007).
    """
    end = _as_utc(end_dt)
    if pub_date is not None:
        return _as_utc(start_dt) <= _as_utc(pub_date) < end + timedelta(days=1)
    return end >= datetime.now(timezone.utc) - timedelta(days=1)


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    article_limit = get_config()["news_article_limit"]
    # Query Yahoo with the canonical symbol, like every other yfinance path —
    # a raw broker/forex/crypto alias (XAUUSD, BTCUSD) otherwise silently
    # returns no news. Keep the user's ticker in the report header.
    canonical = normalize_symbol(ticker)
    resolved = "" if canonical == ticker else f" (resolved to {canonical})"
    try:
        stock = yf.Ticker(canonical)
        news = yf_retry(lambda: stock.get_news(count=article_limit))

        if not news:
            return f"No news found for {ticker}{resolved}"

        # Parse date range for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_str = ""
        filtered_count = 0

        for article in news:
            data = _extract_article_data(article)

            # Keep only articles within the requested window (look-ahead safe).
            if not _in_news_window(data["pub_date"], start_dt, end_dt):
                continue

            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            filtered_count += 1

        if filtered_count == 0:
            return f"No news found for {ticker}{resolved} between {start_date} and {end_date}"

        return f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back. ``None`` falls back to
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles to return. ``None`` falls back to
            ``global_news_article_limit`` from the active config.

    Returns:
        Formatted string containing global news articles
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]
    search_queries = config["global_news_queries"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    # One bucket per query, then a round-robin across them. Filling a single
    # list and stopping at ``limit`` looks equivalent and is not: the first
    # query alone returns about ten articles, so the loop used to break before
    # the later queries ever ran. With the shipped configuration that meant
    # "geopolitical risk trade war sanctions", the central-bank query and the
    # commodities query were never fetched at all, and macro news was Federal
    # Reserve headlines wearing the name of something broader.
    #
    # Each query is still asked for ``limit`` articles rather than its share.
    # It is the same single request either way, and the headroom matters
    # because the window filter below discards some of what comes back.
    buckets: list[list[dict]] = []
    seen_titles = set()
    errors = 0

    for query in search_queries:
        try:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                news_count=limit,
                enable_fuzzy_query=True,
            ))
        except Exception:
            # One failing query must not cost every other topic. A macro report
            # missing commodities is worth more than no macro report at all.
            errors += 1
            continue

        bucket: list[dict] = []
        for article in (getattr(search, "news", None) or []):
            data = _extract_article_data(article)
            title = data["title"]
            if not title or title in seen_titles:
                continue
            # Filtered here rather than after the cut. Truncating first and
            # filtering second lets out-of-window articles consume slots that
            # in-window ones would have filled (#993, #1007).
            if not _in_news_window(data["pub_date"], start_dt, curr_dt):
                continue
            seen_titles.add(title)
            bucket.append(data)
        buckets.append(bucket)

    if errors and errors == len(search_queries):
        return f"Error fetching global news: all {errors} queries failed"

    selected: list[dict] = []
    for rank in range(max((len(b) for b in buckets), default=0)):
        for bucket in buckets:
            if rank < len(bucket):
                selected.append(bucket[rank])
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break

    if not selected:
        return f"No global news found between {start_date} and {curr_date}"

    news_str = ""
    for data in selected:
        news_str += f"### {data['title']} (source: {data['publisher']})\n"
        if data["summary"]:
            news_str += f"{data['summary']}\n"
        if data["link"]:
            news_str += f"Link: {data['link']}\n"
        news_str += "\n"

    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"
