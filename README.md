# TradingAgents: Multi-Agents LLM Financial Trading Framework

## About this fork

This is a fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). The [trading-helper](https://github.com/nandyalu/trading-helper) project uses it to run one autonomous trading agent on a simulated account.

The default branch, `trading-helper-custom`, starts from upstream v0.4.1. It adds these changes on top.

**Cherry-picked from upstream pull requests:**

- Simpler vendor routing, with a circuit breaker for vendor failures (#1071)
- A retry when a JSON response does not decode (#1074)
- A probability and risk/reward review on every trader proposal (#1082)
- Reddit OAuth2, with an RSS feed as the fallback (#1134)
- A candidate screener script, with trade-horizon-aware prompts (#1122)
- The FRED API key removed from error text (#1324)
- A StockTwits read that stops at 5 MiB (#1328)
- A guide for custom Ollama Modelfiles (#1149)
- An unsettled last price bar no longer marks a symbol as unavailable
- A failed Reddit fetch reports as unavailable, not as "no posts found"

**Custom to this fork:**

- The Trader states a stop price and a target price as ATR (Average True Range) multiples. Code computes the actual price. The model never states a price, because model-stated prices proved unreliable.
- A tool error reaches the model, not only the application logs.
- The four analysts in one run execute at the same time, not one after another.
- Every macro data query in a run is fetched, not only the first.
- A sentiment score on the wrong scale is rescaled before it reaches the report.
- Reddit posts load through a trawl browser when the `REDDIT_TRAWL_URL` variable is set. Reddit's Responsible Builder Policy ended self-serve API app creation, so the OAuth2 path above no longer works for a personal project.

One upstream change is declined on purpose: a pull request that tells the Trader to state its entry and stop as absolute prices. This fork computes prices from a verified closing price and ATR instead, for the reason stated above. See the `trading-helper` project's docs for the full decision log and for how to compare this branch against upstream.
