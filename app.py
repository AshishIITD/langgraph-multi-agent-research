"""
FastAPI backend for the LangGraph Multi-Agent Research Assistant.
Supports streaming via Server-Sent Events (SSE).
"""
import uuid
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from loguru import logger
from graph import get_graph
from state import AgentState

app = FastAPI(
    title="LangGraph Multi-Agent Research Assistant",
    description="Multi-agent research system built with LangGraph. By Ashish Singh.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResearchRequest(BaseModel):
    task: str
    human_in_loop: bool = False
    thread_id: Optional[str] = None
    max_iterations: int = 6


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True
    feedback: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/research")
async def research(request: ResearchRequest):
    """Run the full multi-agent research pipeline. Returns complete result."""
    thread_id = request.thread_id or str(uuid.uuid4())
    graph = get_graph(human_in_loop=False)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [],
        "task": request.task,
        "next_agent": "supervisor",
        "research_notes": [],
        "analysis_results": [],
        "draft_answer": "",
        "critic_feedback": "",
        "iteration": 0,
        "max_iterations": request.max_iterations,
        "final_answer": "",
        "is_complete": False,
        "thread_id": thread_id,
    }

    try:
        result = graph.invoke(initial_state, config)
        return {
            "thread_id": thread_id,
            "task": request.task,
            "final_answer": result.get("final_answer") or result.get("draft_answer", ""),
            "research_notes_count": len(result.get("research_notes", [])),
            "iterations": result.get("iteration", 0),
            "is_complete": result.get("is_complete", False),
            "critic_feedback": result.get("critic_feedback", ""),
        }
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    """Stream agent events via SSE as they happen."""
    thread_id = request.thread_id or str(uuid.uuid4())
    graph = get_graph(human_in_loop=request.human_in_loop)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [],
        "task": request.task,
        "next_agent": "supervisor",
        "research_notes": [],
        "analysis_results": [],
        "draft_answer": "",
        "critic_feedback": "",
        "iteration": 0,
        "max_iterations": request.max_iterations,
        "final_answer": "",
        "is_complete": False,
        "thread_id": thread_id,
    }

    async def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'thread_id': thread_id, 'task': request.task})}\n\n"
        try:
            for chunk in graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    event_data = {
                        "event": "agent_update",
                        "agent": node_name,
                        "iteration": node_output.get("iteration", 0),
                        "next": node_output.get("next_agent", ""),
                        "has_research": bool(node_output.get("research_notes")),
                        "has_draft": bool(node_output.get("draft_answer")),
                        "critic_feedback": node_output.get("critic_feedback", ""),
                    }
                    if node_output.get("final_answer"):
                        event_data["final_answer"] = node_output["final_answer"]
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0)  # yield control

            # If human-in-loop, pause here
            if request.human_in_loop:
                yield f"data: {json.dumps({'event': 'awaiting_human_approval', 'thread_id': thread_id})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'complete', 'thread_id': thread_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/research/resume")
async def resume_research(request: ResumeRequest):
    """Resume a human-in-loop paused graph after approval."""
    graph = get_graph(human_in_loop=True)
    config = {"configurable": {"thread_id": request.thread_id}}

    # Get current state
    current = graph.get_state(config)
    if not current:
        raise HTTPException(404, "Thread not found or already complete")

    if not request.approved:
        # Inject feedback into state and re-run writer
        graph.update_state(config, {
            "critic_feedback": f"Human reviewer: {request.feedback or 'Needs revision'}",
            "draft_answer": "",
        })

    result = graph.invoke(None, config)
    return {
        "thread_id": request.thread_id,
        "final_answer": result.get("final_answer") or result.get("draft_answer", ""),
        "is_complete": result.get("is_complete", False),
    }


@app.get("/research/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Inspect the current state of a research thread."""
    graph = get_graph(human_in_loop=True)
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(404, "Thread not found")
    s = state.values
    return {
        "thread_id": thread_id,
        "task": s.get("task"),
        "iteration": s.get("iteration"),
        "is_complete": s.get("is_complete"),
        "has_draft": bool(s.get("draft_answer")),
        "research_notes_count": len(s.get("research_notes", [])),
        "next_agent": s.get("next_agent"),
    }
