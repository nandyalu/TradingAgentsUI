# TradingAgents: Multi-Agents LLM Financial Trading Framework

## About this fork

This is a fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). The [trading-helper](https://github.com/nandyalu/trading-helper) project uses it to run one autonomous trading agent on a simulated account.

The default branch, `trading-helper-custom`, is rebased onto upstream v0.6.0. It adds these changes.

**Cherry-picked from open upstream pull requests:**

- A circuit breaker for vendor failures (#1071)
- A retry when a JSON response does not decode (#1074)
- A probability and risk/reward review on every trader proposal (#1082)
- Reddit OAuth2, with an RSS feed as the fallback (#1134)
- A candidate screener script, with trade-horizon-aware prompts (#1122)
- A StockTwits read that stops at 5 MiB (#1328)
- A guide for custom Ollama Modelfiles (#1149)

**Custom to this fork:**

- The Trader states a stop and a target as ATR (Average True Range) multiples. Code computes the prices, because model-stated prices proved unreliable.
- A tool error reaches the model, not only the logs.
- The market and fundamentals analysts get their data before the model call, and make one model call with no tools.
- Every macro news query in a run is fetched, not only the first.
- A sentiment score on the wrong scale is rescaled before it reaches the report.
- Reddit posts load through a trawl browser when `REDDIT_TRAWL_URL` is set. Reddit no longer allows self-serve API apps, so the OAuth2 path does not work for a personal project.
- The news tool adds Google News, Finnhub and SEC 8-K filings, and each item can carry a grade from an optional laya sidecar (`LAYA_URL`).
- The news analyst can use Gemini's search grounding (`TRADINGAGENTS_GOOGLE_SEARCH_GROUNDING`), optionally on a separate Google model.
- `propagate()` takes a trade `horizon` and an `on_chunk` callback.

This fork declines one upstream change: the Trader states absolute entry and stop prices (upstream commit `1c44dd1`). This fork computes prices from a verified closing price and ATR instead.

For the full decision log, and for how to compare this branch with upstream, see the trading-helper docs.
