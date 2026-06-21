"""
LangGraph StateGraph definition.
Builds the multi-agent research pipeline with conditional routing.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from agents import supervisor_node, researcher_node, analyst_node, writer_node, critic_node
from loguru import logger


def route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: supervisor decides next node."""
    next_agent = state.get("next_agent", "END")
    if next_agent == "END" or state.get("is_complete"):
        return END
    return next_agent


def build_graph(enable_human_in_loop: bool = False):
    """Build and compile the multi-agent graph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor routes to any agent or END
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "critic": "critic",
            END: END,
        },
    )

    # All agents return to supervisor
    for node in ["researcher", "analyst", "writer", "critic"]:
        builder.add_edge(node, "supervisor")

    # Memory for persistence across turns
    memory = MemorySaver()

    if enable_human_in_loop:
        # Pause before critic evaluates for human review
        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["critic"],
        )
    else:
        graph = builder.compile(checkpointer=memory)

    logger.info(f"Graph compiled (human_in_loop={enable_human_in_loop})")
    return graph


# Singleton graph instances
_graph = None
_hitl_graph = None


def get_graph(human_in_loop: bool = False):
    global _graph, _hitl_graph
    if human_in_loop:
        if _hitl_graph is None:
            _hitl_graph = build_graph(enable_human_in_loop=True)
        return _hitl_graph
    if _graph is None:
        _graph = build_graph(enable_human_in_loop=False)
    return _graph
