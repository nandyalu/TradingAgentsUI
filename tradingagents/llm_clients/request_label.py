"""Name each LLM request after the graph node that sends it, in an HTTP header.

A proxy in front of a pool of local model servers can then show which agent
sent a request, where it would otherwise see one client address for all of
them. Set ``config["llm_request_label"]`` to the app's name. Each request then
carries ``X-Pool-Label: <app> <ticker>: <node>``, for example ``ten-acre NVDA:
Market Analyst``. ``propagate()`` sets the ticker. A request sent outside a
graph node leaves out the node, or carries the name a caller gives with
``node()``.

Only OpenAI-compatible providers get the header. The model sees nothing of it.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler

HEADER = "X-Pool-Label"

# The node of the model call that runs now in this context. LangGraph runs each
# node in a copy of the context, so parallel analysts do not see each other's.
_node: ContextVar[str | None] = ContextVar("llm_request_node", default=None)
# The ticker of the run, set by propagate(). The nodes run in copies of its context.
_ticker: ContextVar[str | None] = ContextVar("llm_request_ticker", default=None)


@contextmanager
def ticker(name: str) -> Iterator[None]:
    """Put the ticker into the label of the requests sent inside this block."""
    token = _ticker.set(name)
    try:
        yield
    finally:
        _ticker.reset(token)


@contextmanager
def node(name: str) -> Iterator[None]:
    """Label the requests sent inside this block, for a call outside a graph."""
    token = _node.set(name)
    try:
        yield
    finally:
        _node.reset(token)


class NodeLabelHandler(BaseCallbackHandler):
    """Set ``_node`` to LangGraph's node name for the length of one model call."""

    # Inline, so that the callback sets the variable in the same context as
    # the HTTP request that follows it.
    run_inline = True

    def __init__(self) -> None:
        self._tokens: dict = {}

    def _start(self, metadata: dict | None, run_id) -> None:
        metadata = metadata or {}
        # An analyst runs as a subgraph inside its own node, so langgraph_node
        # names the inner node ("agent"). The namespace starts with the outer
        # one: "Market Analyst:<id>|agent:<id>".
        namespace = metadata.get("langgraph_checkpoint_ns") or ""
        name = namespace.split("|", 1)[0].split(":", 1)[0] or metadata.get("langgraph_node")
        if name:
            self._tokens[run_id] = _node.set(name)

    def _end(self, run_id) -> None:
        # Put back what was there before, so a label from node() survives.
        token = self._tokens.pop(run_id, None)
        if token is not None:
            try:
                _node.reset(token)
            except ValueError:
                # The call ended in another context; that context goes too.
                pass

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._start(metadata, run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs) -> None:
        self._start(metadata, run_id)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._end(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._end(run_id)


def label_for(app: str) -> str:
    node, symbol = _node.get(), _ticker.get()
    label = f"{app} {symbol}" if symbol else app
    if node:
        label = f"{label}: {node}"
    # A header value must be ASCII.
    return label.encode("ascii", "replace").decode("ascii")


def label_kwargs(app: str) -> dict[str, Any]:
    """The ``callbacks`` and HTTP clients that make ChatOpenAI send the label."""
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

    def hook(request) -> None:
        # A caller that names its own label, with extra_headers, keeps it.
        if HEADER not in request.headers:
            request.headers[HEADER] = label_for(app)

    async def async_hook(request) -> None:
        hook(request)

    return {
        "callbacks": [NodeLabelHandler()],
        "http_client": DefaultHttpxClient(event_hooks={"request": [hook]}),
        "http_async_client": DefaultAsyncHttpxClient(event_hooks={"request": [async_hook]}),
    }
