import os
from typing import Annotated, Sequence, TypedDict, Literal
import operator
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

load_dotenv()

# Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    feedback: str
    status: str # e.g. "running", "awaiting_human", "completed"

# Tools Initialization
tavily_tool = TavilySearchResults(max_results=3)
python_repl_tool = PythonREPLTool()

# LLM Initialization
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Helper function to create agents
def create_agent(llm, tools, system_prompt: str):
    prompt = SystemMessage(content=system_prompt)
    if tools:
        llm = llm.bind_tools(tools)
    
    def agent_node(state: AgentState):
        messages = [prompt] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    return agent_node

# --- Define the 5 Agents ---

# 1. Supervisor
def supervisor_node(state: AgentState):
    system_prompt = """You are a Supervisor managing a team of AI agents:
    - Researcher (Web Search)
    - Analyst (Data & Python execution)
    - Writer (Drafting reports)
    - Critic (Reviewing and suggesting improvements)
    
    Review the conversation history. Decide who should act next.
    If a report is final and approved by the Critic, respond EXACTLY with 'FINISH'.
    Otherwise, respond EXACTLY with the name of the next agent: 'Researcher', 'Analyst', 'Writer', or 'Critic'.
    Do NOT output anything else.
    """
    prompt = SystemMessage(content=system_prompt)
    messages = [prompt] + state["messages"]
    response = llm.invoke(messages)
    
    next_step = response.content.strip()
    
    if "FINISH" in next_step.upper():
        return {"next_agent": "FINISH", "status": "completed"}
    elif "RESEARCHER" in next_step.upper():
        return {"next_agent": "Researcher"}
    elif "ANALYST" in next_step.upper():
        return {"next_agent": "Analyst"}
    elif "WRITER" in next_step.upper():
        return {"next_agent": "Writer"}
    elif "CRITIC" in next_step.upper():
        return {"next_agent": "Critic"}
    else:
        # Default fallback
        return {"next_agent": "Researcher"}

# 2. Researcher (uses Tavily)
research_prompt = "You are a Researcher. Use the Tavily search tool to find information requested by the user. Gather thorough details."
researcher_node = create_agent(llm, [tavily_tool], research_prompt)

# 3. Analyst (uses Python REPL)
analyst_prompt = "You are an Analyst. Use the Python REPL tool to execute calculations, data processing, or logical checks. Return the results."
analyst_node = create_agent(llm, [python_repl_tool], analyst_prompt)

# 4. Writer (No tools, just drafting)
writer_prompt = "You are a Writer. Synthesize the findings from the Researcher and Analyst into a cohesive, professional report. Incorporate any feedback from the Critic."
writer_node = create_agent(llm, [], writer_prompt)

# 5. Critic (Human-in-the-loop checkpoint precedes this, or it acts automatically)
critic_prompt = "You are a Critic. Review the Writer's draft. If it is excellent and meets the user's needs, state 'APPROVED'. If it needs work, provide specific 'FEEDBACK' on what to fix."
critic_node = create_agent(llm, [], critic_prompt)

# Tools execution node
tools = [tavily_tool, python_repl_tool]
tool_node = ToolNode(tools)

# --- Graph Construction ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("tools", tool_node)

# Conditional router for tools vs supervisor
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# Add Edges
workflow.set_entry_point("Supervisor")

# Supervisor routing
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next_agent"],
    {
        "Researcher": "Researcher",
        "Analyst": "Analyst",
        "Writer": "Writer",
        "Critic": "Critic",
        "FINISH": END
    }
)

# Agent to tools or back to supervisor
workflow.add_conditional_edges("Researcher", should_continue, {"continue": "tools", "end": "Supervisor"})
workflow.add_conditional_edges("Analyst", should_continue, {"continue": "tools", "end": "Supervisor"})

# Tools always return to the agent that called them. 
# LangGraph ToolNode inherently returns to the caller if wired correctly, 
# but for a custom multi-agent setup, we route tools back to Supervisor to re-assess.
workflow.add_edge("tools", "Supervisor")

workflow.add_edge("Writer", "Supervisor")

# CRITICAL: Human-in-the-loop checkpoint edge. We pause BEFORE the Critic node.
# The user can review the Writer's draft before the Critic finalizes it.
workflow.add_edge("Critic", "Supervisor")

# Initialize MemorySaver for persistent thread memory and human-in-the-loop
memory = MemorySaver()

# Compile graph with a breakpoint BEFORE the Critic node for human review
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["Critic"]
)
