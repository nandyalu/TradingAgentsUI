"""Reddit search fetcher for ticker-specific discussion posts.

Supports two modes:

1. **OAuth (preferred)**: When ``REDDIT_CLIENT_ID`` and ``REDDIT_CLIENT_SECRET``
   env vars are set, uses Reddit's OAuth2 app-only flow to hit
   ``oauth.reddit.com/r/{sub}/search.json`` — gives 100 QPM, score/comment
   counts, and avoids the WAF 403 that blocks unauthenticated JSON requests.

2. **trawl**: When ``REDDIT_TRAWL_URL`` is set, loads Reddit's HTML search page
   in a self-hosted trawl browser (``POST {REDDIT_TRAWL_URL}/scrape``). That
   page loads where the RSS feed returns 429, and it carries score and comment
   counts. It has no post bodies, so each post shown gets its body from the
   post's own JSON. A trawl failure falls back to RSS.

3. **RSS fallback**: When neither is configured, falls back to the
   public Atom/RSS search feed (``reddit.com/r/{sub}/search.rss``). Subject
   to aggressive per-IP rate limiting (~1 QPM as of June 2026). RSS lacks
   score/comment counts.

On a 429 we back off once (honouring ``Retry-After``).

A fetch that fails is reported as ``<unavailable>``, never as "no posts found":
the two are different claims, and passing a rate-limited fetch off as silence
hands the sentiment analyst a signal that was never observed (#1295).

Returns formatted plaintext blocks ready for prompt injection and degrades
gracefully — returns a placeholder string rather than raising, so callers never
special-case missing data.
"""

from __future__ import annotations

import html
import http.client
import json
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .date_window import in_window
from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)


def _within_window(posts, start_date, end_date):
    """Keep only posts published in [start_date, end_date] (look-ahead safe).

    No window (both None) leaves the list untouched for live callers. A post with
    no ``created_utc`` epoch is dropped in a historical window (#1220).
    """
    if not (start_date and end_date):
        return posts
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    kept = []
    for p in posts:
        ts = p.get("created_utc")
        created = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        if in_window(created, start_dt, end_dt):
            kept.append(p)
    return kept

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_OAUTH_API = "https://oauth.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_SEARCH_PAGE = "https://www.reddit.com/r/{sub}/search/?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# ---------------------------------------------------------------------------
# OAuth2 app-only token management
# ---------------------------------------------------------------------------
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_oauth_token: str | None = None
_oauth_expires_at: float = 0.0


def _get_oauth_credentials() -> tuple[str, str] | None:
    """Return (client_id, client_secret) from env vars, or None if not set."""
    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _get_oauth_token() -> str | None:
    """Obtain or reuse a Reddit OAuth2 app-only bearer token.

    Uses the "Application Only OAuth" flow (grant_type=client_credentials)
    which doesn't require a user account — just a registered Reddit app
    (script type). Token is cached until 60s before expiry.
    """
    global _oauth_token, _oauth_expires_at

    if _oauth_token and time.time() < _oauth_expires_at:
        return _oauth_token

    creds = _get_oauth_credentials()
    if creds is None:
        return None

    client_id, client_secret = creds
    data = urlencode({"grant_type": "client_credentials"}).encode()
    req = Request(_TOKEN_URL, data=data, method="POST")
    req.add_header("User-Agent", _UA)

    import base64
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req.add_header("Authorization", f"Basic {auth}")

    try:
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        token = body.get("access_token")
        expires_in = body.get("expires_in", 3600)
        if token:
            _oauth_token = token
            _oauth_expires_at = time.time() + expires_in - 60
            logger.debug("Reddit OAuth token acquired, expires in %ds", expires_in)
            return _oauth_token
    except Exception as exc:
        logger.warning("Reddit OAuth token request failed: %s", exc)

    return None


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


# Headerless-429 backoff when Reddit gives no Retry-After. Measured against
# /r/{sub}/search.rss, a retry still 429s at 8s, 10s and 30s of spacing and
# succeeds at 60s, so a shorter wait spends the one retry on a request that
# cannot succeed (#1295). Jittered so several analyses sharing an IP don't
# retry in lockstep and re-collide on the limit.
_RETRY_FALLBACK_SECONDS = 60.0


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """Return ``seconds`` with +/-``frac`` random jitter, to desynchronize
    concurrent runs pacing against the same per-IP limit."""
    return seconds * (1.0 + random.uniform(-frac, frac))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 60s.

    The cap matches ``_RETRY_FALLBACK_SECONDS``: honouring less than we would
    wait on our own would spend the one retry on a request we already know is
    too early.

    Returns ``None`` only when the header is absent or unparseable; a valid
    ``Retry-After: 0`` returns ``0.0`` (retry at once), not ``None``.
    """
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 60.0) if val is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


# Reddit search feeds are small (a page of results); cap the read so a
# compromised or misbehaving endpoint can't stream an unbounded body into
# memory before we parse it. Overflow raises http.client.HTTPException, which
# both fetch paths already treat as a failed fetch (degrade to empty / RSS).
_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp) -> bytes:
    """Read a response body bounded to ``_MAX_FEED_BYTES``, raising on overflow."""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(
            f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse"
        )
    return data


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict] | None:
    """Default path: parse the public Atom search feed for a subreddit.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.

    Returns ``[]`` when the search ran and matched nothing, and ``None`` when
    the fetch itself failed. The caller must keep these apart: rendering a
    failed fetch as "no posts found" hands the sentiment analyst an absence of
    discussion that was never observed (#1295).
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            # Honour a server-supplied Retry-After exactly (including 0); jitter
            # only our own fallback so concurrent runs don't retry in lockstep.
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Richer JSON search path (carries score / comment counts).

    Reddit's WAF currently returns ``403 Blocked`` on this endpoint for
    non-OAuth clients (issue #862), so it is NOT used by default — calling it on
    every request only doubled our volume against the per-IP rate limit and
    triggered 429s on the RSS fallback. Kept for the day the WAF relaxes or an
    OAuth token is wired in; degrades to RSS on failure.
    """
    url = _API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(_read_capped(resp))
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit_oauth(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    token: str,
    _retry: bool = True,
) -> list[dict] | None:
    """OAuth-authenticated JSON search (100 QPM, includes score/comments)."""
    url = _OAUTH_API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={
        "User-Agent": _UA,
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            wait = _retry_after_seconds(exc) or 2.0
            logger.warning(
                "Reddit OAuth 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_oauth(ticker, sub, limit, timeout, token, _retry=False)
        if exc.code == 401:
            global _oauth_token
            _oauth_token = None
            logger.warning("Reddit OAuth token expired mid-request, falling back to RSS")
            return _fetch_subreddit_rss(ticker, sub, limit, timeout)
        logger.warning("Reddit OAuth fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit OAuth fetch failed for r/%s · %s: %s — falling back to RSS",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


# ---------------------------------------------------------------------------
# trawl: Reddit's HTML search page through a self-hosted browser
# ---------------------------------------------------------------------------
# Measured 2026-09-13 from one IP: the RSS feed returned 429 after its first
# request, and 18 HTML search pages loaded through trawl 2 s apart all
# succeeded. The JSON search endpoint also loads through trawl, but it returns
# far fewer results (0 posts where the HTML page shows 14). It is not used,
# because a short answer that looks like success is worse than a 429.
_TRAWL_TIMEOUT_SECONDS = 60.0
_RECENT_SECONDS = 7 * 24 * 3600

_POST_UNIT = 'data-testid="search-post-unit"'
_COUNTER_ROW = 'data-testid="search-counter-row"'
_NO_RESULTS_MARKER = 'data-testid="search-error-message"'
_NO_RESULTS_TEXT = "find any results for"
_TITLE_TAG = re.compile(r'<a\b[^>]*\bdata-testid="post-title"[^>]*>')
_HREF = re.compile(r'\bhref="([^"]*)"')
_ARIA_LABEL = re.compile(r'\baria-label="([^"]*)"')
_TIMEAGO_TS = re.compile(r'<faceplate-timeago\b[^>]*\bts="([^"]*)"')
_NUMBER = re.compile(r'<faceplate-number\b[^>]*\bnumber="(-?\d+)"')
_PRE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S)


def _trawl_url() -> str | None:
    """Return the trawl address from ``REDDIT_TRAWL_URL``, or None if unset."""
    return os.environ.get("REDDIT_TRAWL_URL", "").strip().rstrip("/") or None


def _trawl_scrape(base: str, url: str) -> str | None:
    """Load ``url`` in trawl's browser. Return the page, or None if it failed."""
    body = json.dumps({
        "url": url,
        "maxTimeout": int(_TRAWL_TIMEOUT_SECONDS * 1000),
        # Tier 1 is a plain fetch from this IP, and Reddit throttles it like
        # the RSS feed. Tier 4 needs a residential proxy.
        "skipHttp": True,
        "maxTier": 3,
    }).encode()
    req = Request(f"{base}/scrape", data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=_TRAWL_TIMEOUT_SECONDS + 10) as resp:
            payload = json.loads(_read_capped(resp))
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # OSError includes HTTPError, which is how trawl reports a failed scrape.
        logger.warning("trawl fetch failed for %s: %s", url, exc)
        return None
    if not isinstance(payload, dict) or payload.get("statusCode") != 200 or not payload.get("html"):
        logger.warning(
            "trawl fetch for %s returned status %s: %s",
            url, payload.get("statusCode") if isinstance(payload, dict) else None,
            payload.get("error") if isinstance(payload, dict) else payload,
        )
        return None
    return payload["html"]


def _parse_search_page(page: str, limit: int | None = None) -> list[dict] | None:
    """Read the posts from Reddit's HTML search page.

    Returns the posts, ``[]`` only when Reddit says it found no results, and
    ``None`` for any other page. A redesigned page must read as a failed fetch:
    a parser that matches nothing would report a silence that was never
    observed (#1295).
    """
    starts = [m.start() for m in re.finditer(re.escape(_POST_UNIT), page)]
    if not starts:
        if _NO_RESULTS_MARKER in page and _NO_RESULTS_TEXT in page:
            return []
        return None
    posts = []
    for start, end in zip(starts, starts[1:] + [len(page)]):
        block = page[start:end]
        tag = _TITLE_TAG.search(block)
        href = _HREF.search(tag.group(0)) if tag else None
        label = _ARIA_LABEL.search(tag.group(0)) if tag else None
        ts = _TIMEAGO_TS.search(block)
        created = _iso_to_timestamp(ts.group(1)) if ts else None
        if not (href and label and created):
            # One post in an unknown shape means the page shape is unknown. The
            # time is required, because the fetcher drops posts by age.
            return None
        row = block.find(_COUNTER_ROW)
        numbers = _NUMBER.findall(block, row) if row >= 0 else []
        # Reddit hides the score of some new posts. Show no counts then, not zeros.
        score, comments = (int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else (None, None)
        posts.append({
            "title": html.unescape(label.group(1)),
            "score": score,
            "num_comments": comments,
            "created_utc": created,
            "selftext": "",
            "permalink": html.unescape(href.group(1)),
            "source": "trawl",
        })
        if limit is not None and len(posts) >= limit:
            break
    return posts


def _fetch_post_body(base: str, permalink: str) -> str:
    """Return a post's body through trawl, or "" if it cannot be read."""
    page = _trawl_scrape(base, f"https://www.reddit.com{permalink.rstrip('/')}.json")
    if page is None:
        return ""
    # The browser can wrap a JSON document in a viewer page. The JSON is in <pre>.
    m = _PRE.search(page)
    raw = html.unescape(m.group(1)) if m else page
    try:
        return json.loads(raw)[0]["data"]["children"][0]["data"].get("selftext") or ""
    except (ValueError, LookupError, TypeError, AttributeError) as exc:
        logger.warning("Reddit post body unreadable for %s: %s", permalink, exc)
        return ""


def _fetch_subreddit_trawl(
    ticker: str,
    sub: str,
    limit: int,
    base: str,
    now: float | None = None,
) -> list[dict] | None:
    """Search one subreddit through trawl. ``None`` means the fetch failed.

    Keeps only posts from the past 7 days, then the newest ``limit`` of them.
    """
    page = _trawl_scrape(base, _SEARCH_PAGE.format(sub=sub, qs=_search_qs(ticker, limit)))
    if page is None:
        return None
    parsed = _parse_search_page(page)
    if parsed is None:
        logger.warning(
            "Reddit search page for r/%s · %s does not match the expected markup", sub, ticker
        )
        return None
    # On 2026-09-13 the page ignored t=week with sort=new and returned posts
    # 17 days old. The analyst is told these posts are from the past 7 days.
    cutoff = (time.time() if now is None else now) - _RECENT_SECONDS
    posts = [p for p in parsed if p["created_utc"] >= cutoff][:limit]
    # The RSS feed carries a body for every post, so every post shown gets one.
    for post in posts:
        if post["permalink"].startswith("/r/"):
            post["selftext"] = _fetch_post_body(base, post["permalink"])
    return posts


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict] | None:
    """Fetch one subreddit: OAuth, then trawl, then the RSS feed.

    ``None`` means the fetch failed.

    With OAuth (REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET): uses the JSON
    endpoint at oauth.reddit.com — 100 QPM, includes score/comment counts.
    With REDDIT_TRAWL_URL: loads the HTML search page through trawl, and falls
    back to the RSS feed if that fails.
    Otherwise: the public RSS feed (~1 QPM, no metrics).
    """
    token = _get_oauth_token()
    if token:
        return _fetch_subreddit_oauth(ticker, sub, limit, timeout, token)
    trawl = _trawl_url()
    if trawl:
        posts = _fetch_subreddit_trawl(ticker, sub, limit, trawl)
        if posts is not None:
            return posts
        logger.warning("trawl could not read r/%s · %s — falling back to the RSS feed", sub, ticker)
    return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=_retry)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` paces the RSS-only per-subreddit requests to stay
    under Reddit's public per-IP rate limit; combined with the RSS-first path
    it makes 429s rare even when several analyses run back-to-back.

    With ``REDDIT_TRAWL_URL`` set, subreddits are fetched at once instead of
    one after another. trawl's per-IP rate limit does not apply the same way
    to a real browser session, and each subreddit is an independent trawl
    session anyway, so there is nothing to pace between them. Measured
    2026-09-15 on 3 subreddits: 39.6s one after another against 15.2s at
    once, no fetch failures. See ``market-data.md`` for the trawl setup
    this assumes.

    When ``start_date``/``end_date`` (yyyy-mm-dd) are given, posts are trimmed to
    that window so a historical run does not leak current discussion into a
    backtest (#1220).
    """
    # Crypto reaches us as a Yahoo pair (BTC-USD); search Reddit for the base
    # ("BTC") so the query actually matches discussion instead of near-nothing.
    ticker = crypto_base(ticker) or ticker
    subreddits = list(subreddits)
    blocks = []
    total_posts = 0
    unavailable = []

    if _trawl_url():
        with ThreadPoolExecutor(max_workers=len(subreddits)) as pool:
            fetched_by_sub = dict(zip(
                subreddits,
                pool.map(lambda sub: _fetch_subreddit(ticker, sub, limit_per_sub, timeout), subreddits),
            ))
    else:
        fetched_by_sub = {}
        allow_retry = True
        for i, sub in enumerate(subreddits):
            if i > 0 and inter_request_delay:
                time.sleep(_jitter(inter_request_delay))
            fetched = _fetch_subreddit(ticker, sub, limit_per_sub, timeout, _retry=allow_retry)
            if fetched is None:
                # One failure means the per-IP budget is likely gone, so skip
                # the (now 60s) back-off on the remaining subreddits rather
                # than stalling the run on retries that cannot succeed;
                # #1286 tracks coordinating this properly.
                allow_retry = False
            fetched_by_sub[sub] = fetched

    for sub in subreddits:
        fetched = fetched_by_sub[sub]
        if fetched is None:
            # A failed fetch is not an absence of discussion, so it must not be
            # rendered as "no posts found" (#1295).
            unavailable.append(sub)
            blocks.append(f"r/{sub}: <unavailable: fetch failed, not an absence of posts>")
            continue
        posts = _within_window(fetched, start_date, end_date)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        searched = [s for s in subreddits if s not in unavailable]
        if not searched:
            # Every source failed: claiming "no posts" here would assert a
            # silence we never observed.
            return (
                f"<Reddit unavailable: every source failed to fetch "
                f"({', '.join(f'r/{s}' for s in unavailable)}); this is not an "
                f"absence of discussion>"
            )
        summary = (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in searched)} in the past 7 days>"
        )
        if unavailable:
            summary += (
                f"\n<unavailable (fetch failed): "
                f"{', '.join(f'r/{s}' for s in unavailable)}>"
            )
        return summary
    return "\n\n".join(blocks)
