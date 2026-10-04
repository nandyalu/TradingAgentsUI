"""llm_request_label names each request after the graph node that sends it."""

import importlib

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tradingagents.llm_clients import build_llm_kwargs, create_llm_client
from tradingagents.llm_clients.request_label import HEADER, node, ticker

COMPLETION = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _llm(seen: list):
    kwargs = build_llm_kwargs({"llm_provider": "ollama", "llm_request_label": "app"})

    # openai 2 builds on httpx and openai 3 on httpx2, so the mock comes from
    # whichever package the client is.
    client = kwargs["http_client"]
    http = importlib.import_module(type(client).__mro__[1].__module__.split(".")[0])

    def handler(request):
        seen.append(request.headers.get(HEADER))
        return http.Response(200, json=COMPLETION)

    client._transport = http.MockTransport(handler)
    return create_llm_client("ollama", "m", base_url="http://pool/v1", api_key="k", **kwargs).get_llm()


def test_a_node_call_carries_the_node_name_and_a_plain_call_the_app_name():
    seen: list = []
    llm = _llm(seen)

    class State(TypedDict):
        out: str

    graph = StateGraph(State)
    graph.add_node("Market Analyst", lambda s: {"out": llm.invoke("hi").content})
    graph.add_edge(START, "Market Analyst")
    graph.add_edge("Market Analyst", END)
    graph.compile().invoke({"out": ""})
    llm.invoke("hi")

    assert seen == ["app: Market Analyst", "app"]


def test_a_label_the_caller_names_is_kept():
    seen: list = []
    llm = _llm(seen)
    llm.client.create(model="m", messages=[{"role": "user", "content": "hi"}],
                      extra_headers={HEADER: "app: agent"})
    assert seen == ["app: agent"]


def test_no_label_and_other_providers_add_nothing():
    assert "http_client" not in build_llm_kwargs({"llm_provider": "ollama"})
    assert "http_client" not in build_llm_kwargs({"llm_provider": "google", "llm_request_label": "app"})


def test_node_names_a_call_outside_a_graph_and_survives_a_langchain_call():
    seen: list = []
    llm = _llm(seen)
    with node("agent"):
        llm.client.create(model="m", messages=[{"role": "user", "content": "hi"}])
        llm.invoke("hi")
    llm.invoke("hi")
    assert seen == ["app: agent", "app: agent", "app"]


def test_the_ticker_reaches_the_label_inside_the_graph():
    seen: list = []
    llm = _llm(seen)

    class State(TypedDict):
        out: str

    graph = StateGraph(State)
    graph.add_node("Trader", lambda s: {"out": llm.invoke("hi").content})
    graph.add_edge(START, "Trader")
    graph.add_edge("Trader", END)
    with ticker("NVDA"):
        graph.compile().invoke({"out": ""})
        llm.invoke("hi")
    assert seen == ["app NVDA: Trader", "app NVDA"]
