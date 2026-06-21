from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Shared state across all agents in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    next_agent: str
    research_notes: list[str]
    analysis_results: list[str]
    draft_answer: str
    critic_feedback: str
    iteration: int
    max_iterations: int
    final_answer: str
    is_complete: bool
    thread_id: str
