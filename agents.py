import os
from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL = "llama3.1:8b"   # Runs via Ollama — no API key needed

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    status: str

# Only use Tavily if API key is available, otherwise skip web search
try:
    tavily_tool = TavilySearchResults(max_results=3)
    tools = [tavily_tool, PythonREPLTool()]
except Exception:
    tools = [PythonREPLTool()]

python_repl_tool = PythonREPLTool()
llm = ChatOllama(model=LOCAL_MODEL, temperature=0)

def create_agent(llm, agent_tools, system_prompt: str):
    prompt = SystemMessage(content=system_prompt)
    bound_llm = llm.bind_tools(agent_tools) if agent_tools else llm

    def agent_node(state: AgentState):
        messages = [prompt] + list(state["messages"])
        response = bound_llm.invoke(messages)
        return {"messages": [response]}
    return agent_node

def supervisor_node(state: AgentState):
    system_prompt = """You are a Supervisor managing: Researcher, Analyst, Writer, Critic.
Review the conversation and decide who acts next.
If the report is final and approved, respond EXACTLY 'FINISH'.
Otherwise respond EXACTLY with one of: 'Researcher', 'Analyst', 'Writer', 'Critic'."""
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    next_step = response.content.strip()
    if "FINISH" in next_step.upper():
        return {"next_agent": "FINISH", "status": "completed"}
    for agent in ["Researcher", "Analyst", "Writer", "Critic"]:
        if agent.upper() in next_step.upper():
            return {"next_agent": agent}
    return {"next_agent": "Researcher"}

researcher_node = create_agent(llm, tools, "You are a Researcher. Search for and gather information on the topic.")
analyst_node    = create_agent(llm, [python_repl_tool], "You are an Analyst. Analyze data using Python when needed.")
writer_node     = create_agent(llm, [], "You are a Writer. Synthesize findings into a clear professional report.")
critic_node     = create_agent(llm, [], "You are a Critic. Review the draft. Reply 'APPROVED' if excellent, else give 'FEEDBACK'.")

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst",    analyst_node)
workflow.add_node("Writer",     writer_node)
workflow.add_node("Critic",     critic_node)
workflow.add_node("tools",      tool_node)

workflow.set_entry_point("Supervisor")
workflow.add_conditional_edges("Supervisor", lambda x: x["next_agent"], {
    "Researcher": "Researcher", "Analyst": "Analyst",
    "Writer": "Writer", "Critic": "Critic", "FINISH": END
})
workflow.add_conditional_edges("Researcher", should_continue, {"continue": "tools", "end": "Supervisor"})
workflow.add_conditional_edges("Analyst",    should_continue, {"continue": "tools", "end": "Supervisor"})
workflow.add_edge("tools",   "Supervisor")
workflow.add_edge("Writer",  "Supervisor")
workflow.add_edge("Critic",  "Supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["Critic"])
