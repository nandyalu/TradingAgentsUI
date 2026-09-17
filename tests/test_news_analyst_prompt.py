"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)`` while the tool takes a
``ticker``, tricking the LLM into hallucinating free-text query calls.
"""
import inspect

import pytest

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.utils.news_data_tools import get_news


@pytest.mark.unit
def test_get_news_takes_ticker_not_query():
    arg_names = set(get_news.args.keys())
    assert "ticker" in arg_names
    assert "query" not in arg_names


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(ticker, start_date, end_date)" in src
    assert "get_news(query" not in src


@pytest.mark.unit
def test_google_search_grounding_bound_for_google_llm():
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    class FakeGoogleLLM(Runnable):
        __module__ = "tradingagents.llm_clients.google_client"

        def __init__(self):
            self.bound_tools = None

        def bind_tools(self, tools, **kwargs):
            self.bound_tools = tools
            self.bound_tool_config = kwargs.get("tool_config")
            return self

        def invoke(self, messages, config=None, **kwargs):
            return AIMessage(content="Google Grounded Report", tool_calls=[])

    fake_google = FakeGoogleLLM()
    node = na.create_news_analyst(fake_google, google_search_grounding=True)

    state = {
        "trade_date": "2026-03-31",
        "asset_type": "stock",
        "company_of_interest": "AAPL",
        "messages": [],
    }

    res = node(state)

    assert fake_google.bound_tools is not None
    assert any(isinstance(t, dict) and "google_search" in t for t in fake_google.bound_tools)
    # Gemini rejects the built-in search tool alongside function tools
    # without this set. Confirmed against the live API 2026-09-17.
    assert fake_google.bound_tool_config == {"include_server_side_tool_invocations": True}
    assert res["news_report"] == "Google Grounded Report"


@pytest.mark.unit
def test_google_search_grounding_off_by_default():
    """Grounding needs a Google Cloud billing account (Tier 1) linked --
    on the Free tier, Gemini 3 has zero grounding quota and every call
    fails with 429 RESOURCE_EXHAUSTED. Confirmed live 2026-09-17. So it
    must stay opt-in, not automatic for every Gemini run."""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    class FakeGoogleLLM(Runnable):
        __module__ = "tradingagents.llm_clients.google_client"

        def __init__(self):
            self.bound_tools = None

        def bind_tools(self, tools, **kwargs):
            self.bound_tools = tools
            return self

        def invoke(self, messages, config=None, **kwargs):
            return AIMessage(content="Ungrounded Report", tool_calls=[])

    fake_google = FakeGoogleLLM()
    node = na.create_news_analyst(fake_google)  # google_search_grounding defaults to False

    state = {
        "trade_date": "2026-03-31",
        "asset_type": "stock",
        "company_of_interest": "AAPL",
        "messages": [],
    }

    res = node(state)

    assert fake_google.bound_tools is not None
    assert not any(isinstance(t, dict) and "google_search" in t for t in fake_google.bound_tools)
    assert res["news_report"] == "Ungrounded Report"


@pytest.mark.unit
def test_google_search_grounding_not_bound_for_non_google_llm():
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import Runnable

    class FakeOpenAILLM(Runnable):
        __module__ = "tradingagents.llm_clients.openai_client"

        def __init__(self):
            self.bound_tools = None

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        def invoke(self, messages, config=None, **kwargs):
            return AIMessage(content="Standard OpenAI Report", tool_calls=[])

    fake_openai = FakeOpenAILLM()
    # google_search_grounding=True should have no effect on a non-Google LLM.
    node = na.create_news_analyst(fake_openai, google_search_grounding=True)

    state = {
        "trade_date": "2026-03-31",
        "asset_type": "stock",
        "company_of_interest": "AAPL",
        "messages": [],
    }

    res = node(state)

    assert fake_openai.bound_tools is not None
    assert not any(isinstance(t, dict) and "google_search" in t for t in fake_openai.bound_tools)
    assert res["news_report"] == "Standard OpenAI Report"
