from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_indicator_instruction,
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.agents.utils.structured import NO_EXTERNAL_TOOLS

# The analyst used to pick its indicators and fetch them in 4 to 6 tool rounds.
# The data is the same set each time, so it is fetched before the one model
# call. Each horizon gets the indicators that resolve inside its window.
INDICATORS = {
    "swing": ("close_10_ema", "close_50_sma", "macd", "macds", "macdh", "rsi",
              "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi"),
    "position": ("close_50_sma", "close_200_sma", "close_10_ema", "macd", "macds", "macdh",
                 "rsi", "boll", "boll_ub", "boll_lb", "atr", "vwma"),
}
DESCRIPTIONS = """Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."""
# (days of daily prices, days of each indicator)
WINDOWS = {"swing": (60, 20), "position": (120, 30)}


def _fetch(tool, **args) -> str:
    """A tool's output, or a note that it failed: the report must say what is missing."""
    try:
        return tool.invoke(args)
    except Exception as exc:  # noqa: BLE001 - any vendor error becomes text for the model
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def fetch_market_data(ticker: str, trade_date: str, horizon: str) -> str:
    horizon = horizon if horizon in INDICATORS else "position"
    price_days, indicator_days = WINDOWS[horizon]
    start = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=price_days)).strftime("%Y-%m-%d")
    blocks = [
        f"<start_of_verified_snapshot>\n{_fetch(get_verified_market_snapshot, symbol=ticker, curr_date=trade_date)}\n<end_of_verified_snapshot>",
        f"<start_of_daily_prices>\n{_fetch(get_stock_data, symbol=ticker, start_date=start, end_date=trade_date)}\n<end_of_daily_prices>",
    ]
    for name in INDICATORS[horizon]:
        data = _fetch(get_indicators, symbol=ticker, indicator=name, curr_date=trade_date,
                      look_back_days=indicator_days)
        blocks.append(f"<start_of_indicator {name}>\n{data}\n<end_of_indicator {name}>")
    return "\n\n".join(blocks)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        horizon = str(state.get("horizon") or "position").strip().lower()
        data = fetch_market_data(state["company_of_interest"], current_date, horizon)

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. The data below was collected for you: a verified market snapshot, the daily prices, and these technical indicators. Categories and each category's indicators are:

""" + DESCRIPTIONS + """

Treat the verified snapshot as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If other data conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by the data with concrete dates and prices. If a block is marked unavailable, say so and do not fill it in.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_indicator_instruction(state)
            + get_language_instruction()
            + "\n\n" + data
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what the data supports; another agent decides the trade."
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        report = (prompt | llm).invoke({"messages": state["messages"]}).content

        return {
            "messages": [AIMessage(content=report)],
            "market_report": report,
        }

    return market_analyst_node
