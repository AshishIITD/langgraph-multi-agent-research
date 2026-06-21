"""
Agent nodes for the LangGraph multi-agent system.
Each returns a partial state update dict.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import AgentState
from tools import web_search, python_repl, format_report, TOOL_MAP
from loguru import logger
import json


def get_llm(bind_tools: list = None):
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1,
        )
    else:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
        )
    if bind_tools:
        llm = llm.bind_tools(bind_tools)
    return llm


# ── Supervisor ────────────────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> dict:
    """Decides which agent to run next based on current state."""
    logger.info("[Supervisor] Deciding next agent...")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    if state.get("is_complete") or iteration >= max_iter:
        return {"next_agent": "END", "is_complete": True}

    if not state.get("research_notes"):
        return {"next_agent": "researcher", "iteration": iteration + 1}
    if not state.get("draft_answer"):
        return {"next_agent": "writer", "iteration": iteration + 1}
    if not state.get("critic_feedback"):
        return {"next_agent": "critic", "iteration": iteration + 1}

    llm = get_llm()
    system = """You are a supervisor managing a research team. Based on the current state, 
    decide the next step. Respond ONLY with one of: researcher, analyst, writer, critic, END
    - researcher: need more information
    - analyst: need to compute or analyze data
    - writer: ready to write/improve the answer
    - critic: answer written, needs evaluation
    - END: answer is complete and high quality"""

    summary = f"""Task: {state['task']}
Research notes: {len(state.get('research_notes', []))} items collected
Draft answer: {'Yes' if state.get('draft_answer') else 'No'}
Critic feedback: {state.get('critic_feedback', 'None')}
Iteration: {iteration}/{max_iter}"""

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=summary)])
    next_agent = response.content.strip().lower()
    if next_agent not in ["researcher", "analyst", "writer", "critic", "end"]:
        next_agent = "END"

    logger.info(f"[Supervisor] → {next_agent}")
    return {"next_agent": next_agent, "iteration": iteration + 1}


# ── Researcher ────────────────────────────────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    """Uses web search to gather information on the task."""
    logger.info("[Researcher] Starting research...")
    llm = get_llm(bind_tools=[web_search])
    system = """You are a research specialist. Search for comprehensive, accurate information 
    on the given task. Make 2-3 targeted searches. Be thorough."""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Research this topic thoroughly: {state['task']}")
    ]
    response = llm.invoke(messages)
    notes = []

    # Process tool calls if any
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                notes.append(f"Search '{tc['args'].get('query', '')}': {result[:800]}")
    else:
        notes.append(response.content)

    logger.info(f"[Researcher] Gathered {len(notes)} research notes")
    return {
        "research_notes": state.get("research_notes", []) + notes,
        "messages": [AIMessage(content=f"Research complete: {len(notes)} findings")],
    }


# ── Analyst ───────────────────────────────────────────────────────────────────
def analyst_node(state: AgentState) -> dict:
    """Analyzes research notes, may run Python for calculations."""
    logger.info("[Analyst] Analyzing research...")
    llm = get_llm(bind_tools=[python_repl])
    context = "\n".join(state.get("research_notes", [])[:5])
    system = """You are a data analyst. Analyze the research findings. 
    If numerical analysis is needed, write Python code to compute it."""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Task: {state['task']}\n\nResearch:\n{context}\n\nAnalyze and extract key insights."),
    ])
    analysis = [response.content]
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "python_repl":
                result = python_repl.invoke(tc["args"])
                analysis.append(f"Computation result: {result}")

    return {
        "analysis_results": state.get("analysis_results", []) + analysis,
        "messages": [AIMessage(content="Analysis complete")],
    }


# ── Writer ────────────────────────────────────────────────────────────────────
def writer_node(state: AgentState) -> dict:
    """Synthesizes research + analysis into a coherent answer."""
    logger.info("[Writer] Drafting answer...")
    llm = get_llm()
    context = "\n\n".join([
        "RESEARCH:\n" + "\n".join(state.get("research_notes", [])[:4]),
        "ANALYSIS:\n" + "\n".join(state.get("analysis_results", [])[:2]),
        "PREVIOUS FEEDBACK:\n" + (state.get("critic_feedback") or "None"),
    ])
    system = """You are a technical writer. Write a clear, comprehensive, well-structured answer.
    Use headings, bullet points where appropriate. Cite sources when available."""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Task: {state['task']}\n\nContext:\n{context}\n\nWrite the final answer:"),
    ])
    logger.info("[Writer] Draft complete")
    return {
        "draft_answer": response.content,
        "messages": [AIMessage(content="Draft written")],
    }


# ── Critic ────────────────────────────────────────────────────────────────────
def critic_node(state: AgentState) -> dict:
    """Evaluates the draft and decides if it's complete or needs revision."""
    logger.info("[Critic] Evaluating draft...")
    llm = get_llm()
    system = """You are a critical reviewer. Evaluate the answer on:
    1. Accuracy and factual correctness
    2. Completeness (does it fully answer the task?)
    3. Clarity and structure
    
    Respond with: APPROVED: <brief comment> OR REVISION NEEDED: <specific improvements>"""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Task: {state['task']}\n\nAnswer:\n{state.get('draft_answer', '')}\n\nEvaluate:"),
    ])
    feedback = response.content
    is_complete = feedback.upper().startswith("APPROVED")
    logger.info(f"[Critic] {'Approved' if is_complete else 'Revision needed'}")

    return {
        "critic_feedback": feedback,
        "final_answer": state.get("draft_answer", "") if is_complete else "",
        "is_complete": is_complete,
        "messages": [AIMessage(content=feedback)],
    }
