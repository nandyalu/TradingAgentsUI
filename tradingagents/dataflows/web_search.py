"""Web search fetcher for supplementary financial context.

Uses DuckDuckGo (no API key required) to fetch recent web results when
primary data sources (StockTwits, Reddit) are unavailable. Returns
formatted plaintext blocks ready for prompt injection and degrades
gracefully — returns a placeholder string rather than raising.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UNAVAILABLE = "<web_search_unavailable>"


def web_search_financial(ticker: str, limit: int = 5) -> str:
    """Search the web for recent financial news/sentiment about *ticker*.

    Returns a formatted multi-result string, or a placeholder on failure.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed; web search unavailable")
        return _UNAVAILABLE

    query = f"{ticker} stock news sentiment"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=limit))
            if not results:
                results = list(ddgs.text(query, max_results=limit))
    except Exception as exc:
        logger.warning("Web search failed for %s: %s", ticker, exc)
        return _UNAVAILABLE

    if not results:
        return _UNAVAILABLE

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", r.get("snippet", ""))
        source = r.get("source", r.get("href", ""))
        date = r.get("date", "")
        lines.append(f"[{i}] {title}")
        if date:
            lines.append(f"    Date: {date}")
        if source:
            lines.append(f"    Source: {source}")
        if body:
            lines.append(f"    {body}")
        lines.append("")

    return "\n".join(lines).strip()
