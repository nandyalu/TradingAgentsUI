"""propagate(on_chunk=...) streams each state to the caller and still
returns the merged final state."""
from unittest.mock import MagicMock

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


def _graph(chunks):
    g = MagicMock(spec=TradingAgentsGraph)
    g.debug = False
    g.callbacks = ["handler"]
    g.memory_log = MagicMock()
    g.propagator = MagicMock()
    g.propagator.get_graph_args.return_value = {"config": {}}
    g.graph = MagicMock()
    g.graph.stream.return_value = iter(chunks)
    g.process_signal.return_value = "Buy"
    return g


def test_on_chunk_sees_every_state_and_final_state_is_merged():
    chunks = [{"market_report": "m"}, {"market_report": "m", "final_trade_decision": "Rating: Buy"}]
    g = _graph(chunks)
    seen = []
    final, signal = TradingAgentsGraph._run_graph(g, "AAPL", "2026-09-25", on_chunk=seen.append)
    assert seen == chunks
    assert final == {"market_report": "m", "final_trade_decision": "Rating: Buy"}
    assert signal == "Buy"
    g.graph.invoke.assert_not_called()
    g.propagator.get_graph_args.assert_called_once_with(callbacks=["handler"])


def test_on_chunk_exception_stops_the_run():
    g = _graph([{"market_report": "m"}])

    def stop(_):
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError):
        TradingAgentsGraph._run_graph(g, "AAPL", "2026-09-25", on_chunk=stop)
    g.memory_log.store_decision.assert_not_called()
