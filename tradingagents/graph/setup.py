# TradingAgents/graph/setup.py

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import analyst_placeholder

from .analyst_execution import AnalystNodeSpec, build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        google_search_grounding: bool = False,
        news_analyst_llm: Any = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.google_search_grounding = google_search_grounding
        # Falls back to quick_thinking_llm when no override was built
        # (see TradingAgentsGraph.__init__ for when one is).
        self.news_analyst_llm = news_analyst_llm or quick_thinking_llm

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(
                self.news_analyst_llm,
                google_search_grounding=self.google_search_grounding,
            ),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # One node for each analyst. Each node runs that analyst's own graph.
        for index, spec in enumerate(plan.specs):
            workflow.add_node(
                spec.agent_node,
                self._analyst_branch(spec, analyst_factories[spec.key](), first=index == 0),
            )

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # All analysts start at the same time. The Bull Researcher starts when
        # every analyst has written its report.
        for spec in plan.specs:
            workflow.add_edge(START, spec.agent_node)
        workflow.add_edge([spec.agent_node for spec in plan.specs], "Bull Researcher")

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow

    def _analyst_branch(self, spec: AnalystNodeSpec, analyst_node, first: bool):
        """Return a node that runs one analyst and its tool calls to the end.

        The analysts run at the same time, so each one needs its own list of
        messages. The node runs a small graph with its own state and gives back
        only the report. The shared message list does not change.

        Each analyst gets the same first message as in the old sequential
        graph. The first analyst gets the start message of the run. Every other
        analyst gets the placeholder that the message-clear step wrote. So the
        model gets the same prompts as before.
        """
        branch = StateGraph(AgentState)
        branch.add_node(spec.agent_node, analyst_node)
        branch.add_node(spec.tool_node, self.tool_nodes[spec.key])
        branch.add_edge(START, spec.agent_node)
        branch.add_conditional_edges(
            spec.agent_node,
            getattr(self.conditional_logic, f"should_continue_{spec.key}"),
            {spec.tool_node: spec.tool_node, spec.clear_node: END},
        )
        branch.add_edge(spec.tool_node, spec.agent_node)
        compiled = branch.compile()

        def run_analyst(state):
            branch_input = dict(state)
            if not first:
                branch_input["messages"] = [analyst_placeholder(state)]
            result = compiled.invoke(branch_input)
            return {spec.report_key: result.get(spec.report_key, "")}

        return run_analyst
