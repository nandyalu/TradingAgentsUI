import threading
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.utils.agent_utils import analyst_placeholder
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup

REPORT_KEYS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


class ParallelAnalystTests(unittest.TestCase):
    def setUp(self):
        # Each analyst waits here until all four arrive. A graph that runs the
        # analysts one after another never fills the barrier, so it fails.
        self.barrier = threading.Barrier(4, timeout=10)
        self.first_messages = {}
        self.bull_state = None

    def _analyst(self, key):
        def factory(_llm):
            def node(state):
                self.first_messages[key] = state["messages"][0].content
                self.barrier.wait()
                return {
                    "messages": [AIMessage(content=f"{key} report")],
                    REPORT_KEYS[key]: f"{key} report",
                }

            return node

        return factory

    def _run_graph(self):
        def bull(_llm):
            def node(state):
                self.bull_state = dict(state)
                return {"investment_debate_state": {"count": 2, "current_response": "Bull: go"}}

            return node

        def nothing(_llm):
            return lambda state: {}

        def aggressive(_llm):
            return lambda state: {"risk_debate_state": {"count": 3, "latest_speaker": "Aggressive"}}

        patches = {
            "create_market_analyst": self._analyst("market"),
            "create_sentiment_analyst": self._analyst("social"),
            "create_news_analyst": self._analyst("news"),
            "create_fundamentals_analyst": self._analyst("fundamentals"),
            "create_bull_researcher": bull,
            "create_bear_researcher": nothing,
            "create_research_manager": nothing,
            "create_trader": nothing,
            "create_aggressive_debator": aggressive,
            "create_conservative_debator": nothing,
            "create_neutral_debator": nothing,
            "create_portfolio_manager": nothing,
        }
        with mock.patch.multiple("tradingagents.graph.setup", **patches):
            tool_nodes = {key: (lambda state: {}) for key in REPORT_KEYS}
            setup = GraphSetup(None, None, tool_nodes, ConditionalLogic())
            graph = setup.setup_graph(("market", "social", "news", "fundamentals")).compile()
            self.state = {
                "messages": [HumanMessage(content="NVDA")],
                "company_of_interest": "NVDA",
                "instrument_context": "The instrument to analyze is `NVDA`.",
                "trade_date": "2026-09-13",
            }
            return graph.invoke(self.state)

    def test_the_analysts_run_at_the_same_time(self):
        final = self._run_graph()
        for key, report_key in REPORT_KEYS.items():
            self.assertEqual(final[report_key], f"{key} report")

    def test_each_analyst_gets_the_first_message_of_the_sequential_graph(self):
        self._run_graph()
        placeholder = analyst_placeholder(self.state).content
        self.assertEqual(self.first_messages["market"], "NVDA")
        for key in ("social", "news", "fundamentals"):
            self.assertEqual(self.first_messages[key], placeholder)

    def test_the_bull_researcher_gets_every_report_and_no_analyst_messages(self):
        self._run_graph()
        for key, report_key in REPORT_KEYS.items():
            self.assertEqual(self.bull_state[report_key], f"{key} report")
        self.assertEqual([m.content for m in self.bull_state["messages"]], ["NVDA"])


if __name__ == "__main__":
    unittest.main()
