"""More news for an analysis of a known ticker, and a laya grade on every item.

Yahoo Finance was the only news source. Three more are added here, each as a
block of its own after Yahoo's:

1. **Google News RSS.** Free, no key. The query is the company's name and the
   ticker, because Google News indexes headlines, and a headline names the
   company, not the ticker. The name comes from a profile lookup on the ticker
   the analysis already has. This is not a guess from a headline.
2. **Finnhub company news**, when ``FINNHUB_API_KEY`` is set. Finnhub tags each
   article with the ticker itself.
3. **SEC 8-K filings.** A company files an 8-K for a material event, such as
   results, a merger or an officer who leaves. The filing is the event itself,
   not a report about it.

**A grade is an annotation.** When ``LAYA_URL`` is set, ``annotate`` asks a laya
sidecar two questions about each item: does it mention this company, and what
does it suggest for the stock. It writes the two answers at the start of the
item. It drops nothing, unless ``LAYA_DROP_BELOW`` is set to a value between 0
and 1. Leave that unset until a person has read the grades on real items and
agrees with them.

**Why "mentions" and not "is about" (measured 2026-09-26).** On eight real
CoreWeave headlines and four made-up headlines about other companies, laya's
answer to "is this about CoreWeave" ranged from 0.18 to 0.94 on the real ones.
"Does this mention CoreWeave" gave 0.72 or more on every real one and 0.10 or
less on the others. So the grade measures a mention, and says so. A post that
lists five cashtags and talks about something else mentions all five.

The tone question goes to the ``typed-decisions`` checkpoint. On the same eight
headlines it had the right sign on seven, and the default checkpoint on five.

Every fetcher here returns a string and never raises. A source that fails says
so in one line, and the analysis continues with the others.
"""
from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import quote_plus

import requests

from . import laya
from .date_window import in_window

logger = logging.getLogger(__name__)

_TIMEOUT = 20
_GOOGLE_LIMIT = 15
_FINNHUB_LIMIT = 15

# 8-K item numbers and what each one reports. The list is SEC's own, from
# Form 8-K's General Instructions. Only the items a trader reads are named; an
# item missing here is shown by its number.
_8K_ITEMS = {
    "1.01": "a material agreement was signed",
    "1.02": "a material agreement ended",
    "1.03": "bankruptcy or receivership",
    "2.01": "an acquisition or a sale of assets was completed",
    "2.02": "results of operations (earnings)",
    "2.03": "a new direct financial obligation",
    "2.05": "costs of an exit or a restructuring",
    "2.06": "a material impairment",
    "3.01": "a delisting notice or a failed listing rule",
    "4.02": "earlier financial statements can no longer be relied on",
    "5.02": "a director or an officer left, or was appointed",
    "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}


@lru_cache(maxsize=256)
def company_name(ticker: str) -> str:
    """The company's short name from Yahoo's profile, or the ticker itself."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        name = (info.get("shortName") or info.get("longName") or "").strip()
    except Exception as exc:  # a profile lookup must never stop an analysis
        logger.warning("No company name for %s: %s", ticker, exc)
        name = ""
    # "NVIDIA Corporation" finds fewer headlines than "NVIDIA".
    name = re.sub(r",?\s+(Inc\.?|Corp\.?|Corporation|Co\.?|Ltd\.?|plc|N\.V\.|S\.A\.|Holdings?)$", "", name, flags=re.I)
    return name or ticker


def _window(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    return datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")


def google_news(ticker: str, start_date: str, end_date: str) -> str:
    """Headlines from Google News that name the company or the ticker."""
    name = company_name(ticker)
    days = max(1, (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days)
    query = quote_plus(f'"{name}" OR {ticker} stock when:{days}d')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        return f"## Google News for {name} ({ticker})\n\n<unavailable: {type(exc).__name__}>"
    start_dt, end_dt = _window(start_date, end_date)
    lines = []
    for item in root.iter("item"):
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            published = None
        if not in_window(published, start_dt, end_dt):
            continue
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        # Google appends " - Source" to the title. The source is shown once.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        when = published.strftime("%Y-%m-%d") if published else "?"
        lines.append(f"### {title} (source: {source or 'unknown'}, {when})")
        if len(lines) >= _GOOGLE_LIMIT:
            break
    if not lines:
        return f"## Google News for {name} ({ticker})\n\nNo headlines between {start_date} and {end_date}."
    return f"## Google News for {name} ({ticker}), from {start_date} to {end_date}\n\n" + "\n\n".join(lines)


def finnhub_news(ticker: str, start_date: str, end_date: str) -> str:
    """Finnhub's ticker-tagged company news, or "" when no key is set."""
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key:
        return ""
    url = (
        "https://finnhub.io/api/v1/company-news"
        f"?symbol={quote_plus(ticker)}&from={start_date}&to={end_date}&token={key}"
    )
    try:
        response = requests.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        articles = response.json() or []
    except Exception as exc:
        # Never the URL: it carries the key.
        return f"## Finnhub news for {ticker}\n\n<unavailable: {type(exc).__name__}>"
    start_dt, end_dt = _window(start_date, end_date)
    lines = []
    for a in sorted(articles, key=lambda a: a.get("datetime") or 0, reverse=True):
        published = datetime.fromtimestamp(a["datetime"], timezone.utc) if a.get("datetime") else None
        if not in_window(published, start_dt, end_dt):
            continue
        when = published.strftime("%Y-%m-%d") if published else "?"
        lines.append(f"### {(a.get('headline') or '').strip()} (source: {a.get('source') or 'unknown'}, {when})")
        summary = (a.get("summary") or "").replace("\n", " ").strip()
        if summary:
            lines[-1] += "\n" + (summary[:400] + "…" if len(summary) > 400 else summary)
        if len(lines) >= _FINNHUB_LIMIT:
            break
    if not lines:
        return f"## Finnhub news for {ticker}\n\nNo articles between {start_date} and {end_date}."
    return f"## Finnhub news for {ticker}, from {start_date} to {end_date}\n\n" + "\n\n".join(lines)


def sec_8k(ticker: str, start_date: str, end_date: str) -> str:
    """The company's 8-K filings in the window, each with what its items report."""
    from . import sec_edgar

    try:
        cik = sec_edgar.cik_for(ticker)
        if not cik:
            return ""  # not a US filer; nothing to say
        filings = sec_edgar._fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")["filings"]["recent"]
    except Exception as exc:
        return f"## SEC 8-K filings for {ticker}\n\n<unavailable: {type(exc).__name__}>"
    start_dt, end_dt = _window(start_date, end_date)
    lines = []
    for form, filed, items in zip(filings.get("form", []), filings.get("filingDate", []), filings.get("items", [])):
        if form not in ("8-K", "8-K/A"):
            continue
        filed_dt = datetime.strptime(filed, "%Y-%m-%d")
        if not in_window(filed_dt, start_dt, end_dt):
            continue
        what = "; ".join(_8K_ITEMS.get(i.strip(), f"item {i.strip()}") for i in (items or "").split(",") if i.strip())
        lines.append(f"### {form} filed {filed}: {what or 'no items listed'}")
    if not lines:
        return f"## SEC 8-K filings for {ticker}\n\nNone between {start_date} and {end_date}."
    return f"## SEC 8-K filings for {ticker}, from {start_date} to {end_date}\n\n" + "\n\n".join(lines)


# The news analyst and the sentiment analyst both ask for the same ticker's
# news in one analysis, a few seconds apart. Half an hour is long enough to
# serve the second from the first, and short enough that a second analysis of
# the same ticker later that day fetches again.
_CACHE_SECONDS = 1800
_cache: dict[tuple[str, str, str], tuple[float, str]] = {}


def with_extra(yahoo: str, ticker: str, start_date: str, end_date: str) -> str:
    """Yahoo's block, then Google News, Finnhub and 8-K, all graded.

    The legend that explains the grades comes first, once, when laya graded
    anything.
    """
    key = (ticker.upper(), start_date, end_date)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_SECONDS:
        extra = hit[1]
    else:
        blocks = [f(ticker, start_date, end_date) for f in (google_news, finnhub_news, sec_8k)]
        extra = "\n\n".join(annotate(b, ticker) for b in blocks if b)
        _cache[key] = (time.monotonic(), extra)
    text = "\n\n".join(part for part in (annotate(yahoo, ticker), extra) if part)
    return with_legend(text)


def with_legend(text: str) -> str:
    """The text with the grade legend first, when it carries a grade."""
    return f"{GRADE_LEGEND}\n\n{text}" if "[mentions " in text else text


# ---------------------------------------------------------------------------
# Grading

# An item starts at a "### " headline (Yahoo, Google, Finnhub, 8-K) or at a
# "[...]" line (StockTwits, Reddit). It runs until a blank line or the next item.
_ITEM_START = re.compile(r"^(\s*)(### |\[)")
_SENTIMENT_LEVELS = [
    "clearly bearish",
    "somewhat bearish",
    "neutral or unclear",
    "somewhat bullish",
    "clearly bullish",
]
GRADE_LEGEND = (
    "Each item below starts with a grade from a small classifier, not from an analyst: "
    "`mentions` is the chance the item mentions this company, and `tone` runs from -1 "
    "(bearish for the stock) to +1 (bullish). Weigh the item, not the grade."
)


def _drop_below() -> float | None:
    raw = os.environ.get("LAYA_DROP_BELOW", "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def grade(text: str, ticker: str) -> tuple[float, float] | None:
    """(mentions, tone) for one item, or None when laya is off or fails."""
    name = company_name(ticker)
    state = {"text": text[:1500]}
    mention = laya.ask(state, {"mentions": {
        "type": "noul",
        "instructions": f"Does this text mention {name} or {ticker}?",
    }})
    tone = laya.ask(state, {"tone": {
        "type": "score",
        "instructions": f"What does this text suggest for the stock price of {name} ({ticker})?",
        "criteria": _SENTIMENT_LEVELS,
    }}, model="typed-decisions")
    if not mention or not tone:
        return None
    # laya's score runs 0 to 4 across the five levels; 2 is neutral.
    return float(mention["mentions"]["noul"]), (float(tone["tone"]["score"]) - 2) / 2


def _items(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line index of each item in a block."""
    spans, start = [], None
    for i, line in enumerate(lines):
        if _ITEM_START.match(line):
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif not line.strip() and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(lines) - 1))
    return spans


def annotate(block: str, ticker: str) -> str:
    """The block with a grade at the start of each item, or unchanged.

    Unchanged when ``LAYA_URL`` is unset, when the block has no items, or when
    laya fails on the first item. With ``LAYA_DROP_BELOW`` set, an item whose
    ``mentions`` is below it is left out, and a last line says how many were.
    """
    if not laya.laya_url() or not block:
        return block
    lines = block.splitlines()
    spans = _items(lines)
    if not spans:
        return block
    floor = _drop_below()
    out = list(lines)
    dropped = 0
    # Right to left, so an earlier span's indices stay valid while later lines change.
    for first, last in reversed(spans):
        scored = grade("\n".join(lines[first : last + 1]), ticker)
        if scored is None:
            return block
        mentions, tone = scored
        if floor is not None and mentions < floor:
            del out[first : last + 1]
            dropped += 1
            continue
        match = _ITEM_START.match(out[first])
        indent, marker = match.group(1), match.group(2)
        rest = out[first][len(indent) + len(marker):]
        tag = f"[mentions {mentions:.2f} · tone {tone:+.2f}]"
        out[first] = f"{indent}### {tag} {rest}" if marker == "### " else f"{indent}{tag} [{rest}"
    text = "\n".join(out)
    if dropped:
        text += f"\n\n({dropped} item(s) left out: the classifier found no mention of {ticker}.)"
    return text
