"""Tests for the trawl path: Reddit's HTML search page loaded in a self-hosted
browser, with post bodies from each top post's JSON.

The fixture markup is cut down from real pages fetched on 2026-09-13.
"""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit


def _unit(post_id, title_attr, ts, votes=None, comments=None):
    """One search result, in the shape of Reddit's search-post-unit markup."""
    counters = (
        f'<div data-testid="search-counter-row"><span><faceplate-number number="{votes}" pretty="">'
        f"<!---->{votes}</faceplate-number> votes</span><span>·</span><span>"
        f'<faceplate-number number="{comments}" pretty=""><!---->{comments}</faceplate-number>'
        " comments</span></div>"
        if votes is not None
        else '<div data-testid="search-counter-row"><span>Vote</span></div>'
    )
    href = f"/r/stocks/comments/{post_id}/slug/"
    return f"""<div data-testid="search-post-unit">
  <search-telemetry-tracker data-faceplate-tracking-context="{{}}"><h2>
    <a data-testid="post-title" href="{href}" class="absolute inset-0" aria-label="{title_attr}">
      <faceplate-screen-reader-content>title</faceplate-screen-reader-content>
    </a>
  </h2><!----></search-telemetry-tracker>
  <div data-testid="sdui-post-unit">
    <span>r/stocks</span><span>·</span><faceplate-timeago format="narrow" ts="{ts}">
      <time datetime="2026-09-13T19:37:00.997Z">3h ago</time></faceplate-timeago>
    <a data-testid="post-title-text" id="search-post-title-t3_{post_id}" href="{href}">title</a>
    {counters}
  </div>
</div>"""


def _page(*units):
    return "<html><body><main>" + "".join(units) + "</main></body></html>"


_RESULTS = _page(
    _unit("aaa111", "Will there be a &quot;TSMC&quot; of robotics?",
          "2026-09-13T19:37:00.997000+0000", votes=53, comments=73),
    _unit("bbb222", "NVDA post-earnings", "2026-09-12T10:00:00.000000+0000",
          votes=210, comments=12),
    _unit("ccc333", "Too new to score", "2026-09-13T21:00:00.000000+0000"),
)

_NO_RESULTS = _page(
    '<div data-testid="search-error-message"><img alt="" src="snoo_thinking.svg">'
    "<div>Hm...we couldn’t find any results for ZQXJWV</div>"
    "<div>Double-check your spelling or try different keywords</div></div>"
)


_NOW = reddit._iso_to_timestamp("2026-09-13T22:00:00+00:00")


def _post_json(selftext):
    return json.dumps([{"data": {"children": [{"data": {"selftext": selftext}}]}}, {}])


@pytest.mark.unit
class TestParseSearchPage:
    def test_reads_title_counts_time_and_link(self):
        posts = reddit._parse_search_page(_RESULTS, limit=10)
        assert [p["title"] for p in posts] == [
            'Will there be a "TSMC" of robotics?', "NVDA post-earnings", "Too new to score",
        ]
        first = posts[0]
        assert (first["score"], first["num_comments"]) == (53, 73)
        assert first["created_utc"] == reddit._iso_to_timestamp("2026-09-13T19:37:00.997+00:00")
        assert first["permalink"] == "/r/stocks/comments/aaa111/slug/"
        assert first["source"] == "trawl"

    def test_hidden_score_is_none_not_zero(self):
        post = reddit._parse_search_page(_RESULTS, limit=10)[2]
        assert post["score"] is None and post["num_comments"] is None

    def test_limit_is_respected(self):
        assert len(reddit._parse_search_page(_RESULTS, limit=2)) == 2

    def test_no_results_page_is_genuine_silence(self):
        assert reddit._parse_search_page(_NO_RESULTS, limit=5) == []

    def test_unknown_page_is_a_failed_fetch(self):
        # A redesign, a login wall or a block page must not read as "no posts".
        assert reddit._parse_search_page("<html><body>Blocked</body></html>", limit=5) is None

    def test_error_marker_without_no_results_text_is_a_failed_fetch(self):
        page = _page('<div data-testid="search-error-message">Something went wrong</div>')
        assert reddit._parse_search_page(page, limit=5) is None

    def test_post_block_without_title_is_a_failed_fetch(self):
        broken = _RESULTS.replace('data-testid="post-title" ', 'data-testid="renamed" ', 1)
        assert reddit._parse_search_page(broken, limit=10) is None

    def test_unreadable_time_is_a_failed_fetch(self):
        # The fetcher drops posts by age, so a post without a readable time
        # cannot be kept or dropped correctly.
        broken = _RESULTS.replace('ts="2026-09-13T19:37:00.997000+0000"', 'ts="3h ago"', 1)
        assert reddit._parse_search_page(broken, limit=10) is None

    def test_no_limit_reads_every_post(self):
        assert len(reddit._parse_search_page(_RESULTS)) == 3


@pytest.mark.unit
class TestTrawlScrape:
    class _Resp:
        def __init__(self, payload):
            self._data = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, size=-1):
            return self._data if size is None or size < 0 else self._data[:size]

    def test_asks_for_the_browser_and_returns_the_page(self):
        seen = {}

        def fake_urlopen(req, timeout):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data)
            return self._Resp({"statusCode": 200, "html": "<html>ok</html>", "tier": 2})

        with patch.object(reddit, "urlopen", side_effect=fake_urlopen):
            page = reddit._trawl_scrape("http://trawl:8191", "https://www.reddit.com/r/stocks/search/?q=NVDA")
        assert page == "<html>ok</html>"
        assert seen["url"] == "http://trawl:8191/scrape"
        assert seen["body"]["skipHttp"] is True
        assert seen["body"]["maxTier"] == 3

    def test_http_error_from_trawl_returns_none(self):
        err = HTTPError("http://trawl:8191/scrape", 500, "Internal Server Error", {}, None)
        with patch.object(reddit, "urlopen", side_effect=err):
            assert reddit._trawl_scrape("http://trawl:8191", "https://x") is None

    def test_non_200_page_returns_none(self):
        resp = self._Resp({"statusCode": 429, "html": "<html>Too Many Requests</html>"})
        with patch.object(reddit, "urlopen", return_value=resp):
            assert reddit._trawl_scrape("http://trawl:8191", "https://x") is None

    def test_connection_refused_returns_none(self):
        with patch.object(reddit, "urlopen", side_effect=ConnectionRefusedError()):
            assert reddit._trawl_scrape("http://trawl:8191", "https://x") is None


@pytest.mark.unit
class TestFetchPostBody:
    def test_reads_json_wrapped_in_pre(self):
        page = f"<html><body><pre>{_post_json('Title says it all. &amp; more')}</pre></body></html>"
        with patch.object(reddit, "_trawl_scrape", return_value=page):
            assert reddit._fetch_post_body("http://t", "/r/stocks/comments/a/") == "Title says it all. & more"

    def test_reads_raw_json(self):
        with patch.object(reddit, "_trawl_scrape", return_value=_post_json("raw body")):
            assert reddit._fetch_post_body("http://t", "/r/stocks/comments/a/") == "raw body"

    def test_requests_the_post_json(self):
        with patch.object(reddit, "_trawl_scrape", return_value=_post_json("")) as scrape:
            reddit._fetch_post_body("http://t", "/r/stocks/comments/a/slug/")
        assert scrape.call_args.args[1] == "https://www.reddit.com/r/stocks/comments/a/slug.json"

    def test_unreadable_body_is_empty(self):
        with patch.object(reddit, "_trawl_scrape", return_value="<html>not json</html>"):
            assert reddit._fetch_post_body("http://t", "/r/stocks/comments/a/") == ""

    def test_failed_fetch_is_empty(self):
        with patch.object(reddit, "_trawl_scrape", return_value=None):
            assert reddit._fetch_post_body("http://t", "/r/stocks/comments/a/") == ""


@pytest.mark.unit
class TestFetchSubredditTrawl:
    def test_every_post_shown_gets_a_body(self):
        # The RSS feed carries a body for every post, so trawl must too.
        bodies = {
            "/r/stocks/comments/aaa111/slug/": "body of 53",
            "/r/stocks/comments/bbb222/slug/": "body of 210",
            "/r/stocks/comments/ccc333/slug/": "body of the unscored post",
        }
        asked = []

        def fake_body(base, permalink):
            asked.append(permalink)
            return bodies[permalink]

        with patch.object(reddit, "_trawl_scrape", return_value=_RESULTS), \
                patch.object(reddit, "_fetch_post_body", side_effect=fake_body):
            posts = reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t", now=_NOW)
        assert asked == list(bodies)
        assert [p["selftext"] for p in posts] == list(bodies.values())

    def test_posts_cut_by_the_limit_get_no_body(self):
        with patch.object(reddit, "_trawl_scrape", return_value=_RESULTS), \
                patch.object(reddit, "_fetch_post_body", return_value="body") as body:
            reddit._fetch_subreddit_trawl("NVDA", "stocks", 2, "http://t", now=_NOW)
        assert body.call_count == 2

    def test_posts_older_than_a_week_are_dropped_before_bodies(self):
        # Seen live: the page ignored t=week and returned a 17-day-old post
        # with 3063 votes, which would also have taken a body request.
        page = _page(
            _unit("new111", "Fresh", "2026-09-12T10:00:00.000000+0000", votes=5, comments=1),
            _unit("old222", "Seventeen days old", "2026-08-27T10:00:00.000000+0000",
                  votes=3063, comments=293),
        )
        with patch.object(reddit, "_trawl_scrape", return_value=page), \
                patch.object(reddit, "_fetch_post_body", return_value="body") as body:
            posts = reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t", now=_NOW)
        assert [p["title"] for p in posts] == ["Fresh"]
        assert [c.args[1] for c in body.call_args_list] == ["/r/stocks/comments/new111/slug/"]

    def test_only_old_posts_is_silence_for_the_week(self):
        page = _page(_unit("old222", "Old", "2026-08-27T10:00:00.000000+0000", votes=1, comments=1))
        with patch.object(reddit, "_trawl_scrape", return_value=page), \
                patch.object(reddit, "_fetch_post_body") as body:
            assert reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t", now=_NOW) == []
        body.assert_not_called()

    def test_limit_applies_after_the_week_filter(self):
        with patch.object(reddit, "_trawl_scrape", return_value=_RESULTS), \
                patch.object(reddit, "_fetch_post_body", return_value=""):
            posts = reddit._fetch_subreddit_trawl("NVDA", "stocks", 1, "http://t", now=_NOW)
        assert [p["title"] for p in posts] == ['Will there be a "TSMC" of robotics?']

    def test_search_page_failure_is_none(self):
        with patch.object(reddit, "_trawl_scrape", return_value=None):
            assert reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t") is None

    def test_unrecognised_page_is_none(self):
        with patch.object(reddit, "_trawl_scrape", return_value="<html>new design</html>"):
            assert reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t") is None

    def test_no_results_is_empty_and_fetches_no_bodies(self):
        with patch.object(reddit, "_trawl_scrape", return_value=_NO_RESULTS), \
                patch.object(reddit, "_fetch_post_body") as body:
            assert reddit._fetch_subreddit_trawl("NVDA", "stocks", 5, "http://t") == []
        body.assert_not_called()


@pytest.mark.unit
class TestFetchSubredditRouting:
    @pytest.fixture(autouse=True)
    def _no_oauth(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        monkeypatch.setattr(reddit, "_oauth_token", None)

    def test_unset_url_goes_straight_to_rss(self, monkeypatch):
        monkeypatch.delenv("REDDIT_TRAWL_URL", raising=False)
        with patch.object(reddit, "_fetch_subreddit_trawl") as trawl, \
                patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            reddit._fetch_subreddit("NVDA", "stocks", 5, 10.0)
        trawl.assert_not_called()
        rss.assert_called_once()

    def test_trawl_result_is_used_without_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://host.docker.internal:8191/")
        with patch.object(reddit, "_fetch_subreddit_trawl", return_value=[{"title": "x"}]) as trawl, \
                patch.object(reddit, "_fetch_subreddit_rss") as rss:
            assert reddit._fetch_subreddit("NVDA", "stocks", 5, 10.0) == [{"title": "x"}]
        assert trawl.call_args.args[3] == "http://host.docker.internal:8191"
        rss.assert_not_called()

    def test_trawl_silence_is_used_without_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://t")
        with patch.object(reddit, "_fetch_subreddit_trawl", return_value=[]), \
                patch.object(reddit, "_fetch_subreddit_rss") as rss:
            assert reddit._fetch_subreddit("NVDA", "stocks", 5, 10.0) == []
        rss.assert_not_called()

    def test_trawl_failure_falls_back_to_rss(self, monkeypatch):
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://t")
        with patch.object(reddit, "_fetch_subreddit_trawl", return_value=None), \
                patch.object(reddit, "_fetch_subreddit_rss", return_value=None) as rss:
            assert reddit._fetch_subreddit("NVDA", "stocks", 5, 10.0, _retry=False) is None
        rss.assert_called_once_with("NVDA", "stocks", 5, 10.0, _retry=False)


@pytest.mark.unit
class TestFormatterShowsTrawlPosts:
    def test_counts_and_top_bodies_are_rendered(self):
        posts = reddit._parse_search_page(_RESULTS, limit=10)
        posts[1]["selftext"] = "Datacenter revenue beat"
        with patch.object(reddit, "_fetch_subreddit", return_value=posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "210↑" in out and "12c" in out
        assert "body excerpt: Datacenter revenue beat" in out
        assert "via RSS" not in out
        assert "Too new to score" in out


@pytest.mark.unit
class TestTrawlAsksForOneSubredditAtATime:
    """Reddit's HTML search page has no `r/a+b+c` form: it answers "no results",
    which the parser reads as a real absence. Measured 2026-09-20 for NVDA over
    one week: 13 posts on the combined RSS feed, 3 on a single subreddit through
    trawl, 0 on the combined page through trawl."""

    def test_a_combined_search_is_split_into_one_page_each(self, monkeypatch):
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://t")
        asked = []

        def record(ticker, sub, limit, base):
            asked.append(sub)
            return [{"title": f"from {sub}", "created_utc": 1, "subreddit": sub}]

        with patch.object(reddit, "_fetch_subreddit_trawl", side_effect=record):
            posts = reddit._fetch_subreddit("NVDA", "a+b+c", 100, 10.0)

        assert asked == ["a", "b", "c"]
        assert [p["subreddit"] for p in posts] == ["a", "b", "c"]

    def test_a_page_trawl_cannot_read_is_asked_of_the_feed_alone(self, monkeypatch):
        """trawl answers 500 for one page under load. Sending the whole search
        to the feed would discard the pages that did load."""
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://t")
        trawled = {"a": [{"title": "a", "created_utc": 1}], "b": None, "c": []}

        with patch.object(reddit, "_fetch_subreddit_trawl",
                          side_effect=lambda t, sub, limit, base: trawled[sub]), \
                patch.object(reddit, "_fetch_subreddit_rss",
                             return_value=[{"title": "b", "created_utc": 1}]) as rss:
            posts = reddit._fetch_subreddit("NVDA", "a+b+c", 100, 10.0)

        rss.assert_called_once_with("NVDA", "b", 100, 10.0, _retry=True)
        assert [p["title"] for p in posts] == ["a", "b"]

    def test_a_subreddit_neither_path_can_read_makes_the_search_unavailable(self, monkeypatch):
        """A partial set would render the missing subreddit as "no posts
        found", which is the silence this path exists to avoid."""
        monkeypatch.setenv("REDDIT_TRAWL_URL", "http://t")
        trawled = {"a": [{"title": "a", "created_utc": 1}], "b": None, "c": []}

        with patch.object(reddit, "_fetch_subreddit_trawl",
                          side_effect=lambda t, sub, limit, base: trawled[sub]), \
                patch.object(reddit, "_fetch_subreddit_rss", side_effect=[None, ["combined"]]) as rss:
            assert reddit._fetch_subreddit("NVDA", "a+b+c", 100, 10.0) == ["combined"]

        assert rss.call_args_list[-1].args[1] == "a+b+c"
