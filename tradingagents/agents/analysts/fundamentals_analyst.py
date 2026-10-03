from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.market_analyst import _fetch
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)

# No tools are offered: the data is fetched before the model call, so the
# analyst's graph is one model turn (see graph/analyst_execution.py).
TOOLS = ()


def fetch_fundamentals_data(ticker: str, trade_date: str) -> str:
    """The analyst called the four statement tools in almost every run, one
    model round each, so they are fetched before its one model call. Insider
    transactions are fetched with them: upstream gave this analyst that tool
    (15b8276), and no other analyst here reads insider trades."""
    blocks = {
        "company_overview": _fetch(get_fundamentals, ticker=ticker, curr_date=trade_date, trade_date=trade_date),
        "balance_sheet": _fetch(get_balance_sheet, ticker=ticker, freq="quarterly", curr_date=trade_date,
                                trade_date=trade_date),
        "cash_flow": _fetch(get_cashflow, ticker=ticker, freq="quarterly", curr_date=trade_date,
                            trade_date=trade_date),
        "income_statement": _fetch(get_income_statement, ticker=ticker, freq="quarterly", curr_date=trade_date,
                                   trade_date=trade_date),
        "insider_transactions": _fetch(get_insider_transactions, ticker=ticker, trade_date=trade_date),
    }
    return "\n\n".join(f"<start_of_{k}>\n{v}\n<end_of_{k}>" for k, v in blocks.items())


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        data = fetch_fundamentals_data(state["company_of_interest"], current_date)

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " The data below was collected for you: a company overview, the quarterly balance sheet, cash flow and income statements, and recent insider buying and selling. If a block is marked unavailable, say so and do not fill it in."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
