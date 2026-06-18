import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from agents import app as agent_app

app = FastAPI(title="LangGraph Multi-Agent API")

class ResearchRequest(BaseModel):
    thread_id: str
    message: str

class ApprovalRequest(BaseModel):
    thread_id: str
    feedback: str
    approve: bool

@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """
    Endpoint to start a research task and stream the agent actions via SSE (<50ms delivery).
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Initialize state with human message
    initial_state = {"messages": [HumanMessage(content=request.message)]}
    
    async def event_generator():
        try:
            # LangGraph async stream
            async for event in agent_app.astream(initial_state, config=config, stream_mode="updates"):
                # Format event for SSE
                yield {
                    "event": "update",
                    "data": json.dumps({k: str(v) for k, v in event.items()})
                }
                await asyncio.sleep(0.01) # Ensure <50ms yielding
                
            # Check if it paused for human review
            state = agent_app.get_state(config)
            if state.next and "Critic" in state.next:
                yield {
                    "event": "checkpoint",
                    "data": json.dumps({"status": "PAUSED", "message": "Awaiting human review before Critic node."})
                }
            else:
                yield {
                    "event": "done",
                    "data": json.dumps({"status": "COMPLETED"})
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
            
    return EventSourceResponse(event_generator())

@app.post("/research/review")
async def human_review(request: ApprovalRequest):
    """
    Endpoint for Human-in-the-loop. Resumes the graph execution after the breakpoint.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    state = agent_app.get_state(config)
    
    if not state.next:
        raise HTTPException(status_code=400, detail="No active breakpoint found for this thread.")
        
    if request.approve:
        human_msg = HumanMessage(content="HUMAN APPROVAL: Proceed to final review.")
    else:
        human_msg = HumanMessage(content=f"HUMAN FEEDBACK: Please revise. {request.feedback}")
        
    # Resume the graph with the new human message injected
    agent_app.update_state(config, {"messages": [human_msg]})
    
    # Continue execution
    async def resume_generator():
        async for event in agent_app.astream(None, config=config, stream_mode="updates"):
            yield {
                "event": "update",
                "data": json.dumps({k: str(v) for k, v in event.items()})
            }
            await asyncio.sleep(0.01)
            
        yield {
            "event": "done",
            "data": json.dumps({"status": "COMPLETED"})
        }
        
    return EventSourceResponse(resume_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
