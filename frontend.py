"""
Streamlit frontend for the LangGraph Multi-Agent Research Assistant.
Shows live agent trace with streaming updates.
"""
import streamlit as st
import httpx
import json
import time

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🤖", layout="wide")

API_URL = "http://localhost:8000"

st.title("🤖 Multi-Agent Research Assistant")
st.caption("Powered by LangGraph | 4 Agents: Researcher → Analyst → Writer → Critic")

AGENT_ICONS = {
    "supervisor": "🎯",
    "researcher": "🔍",
    "analyst": "📊",
    "writer": "✍️",
    "critic": "🔎",
}

col1, col2 = st.columns([2, 1])

with col1:
    task = st.text_area(
        "Research Task",
        placeholder="e.g. 'Compare LoRA vs QLoRA for LLM fine-tuning on limited GPU resources'",
        height=100,
    )

with col2:
    max_iter = st.slider("Max Iterations", 3, 10, 6)
    human_loop = st.checkbox("Human-in-the-loop", value=False,
                              help="Pause before final evaluation for your review")
    stream_mode = st.checkbox("Stream agent trace", value=True)

run_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

if run_btn and task.strip():
    trace_container = st.container()
    result_container = st.container()

    with trace_container:
        st.subheader("🔄 Agent Trace")
        trace_placeholder = st.empty()
        events = []

    with result_container:
        st.subheader("📄 Final Answer")
        answer_placeholder = st.empty()

    if stream_mode:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{API_URL}/research/stream", json={
                "task": task,
                "human_in_loop": human_loop,
                "max_iterations": max_iter,
            }) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            event_type = data.get("event", "")
                            agent = data.get("agent", "")

                            if event_type == "start":
                                events.append(f"▶️ **Started** — Task: _{task[:60]}..._")
                            elif event_type == "agent_update" and agent:
                                icon = AGENT_ICONS.get(agent, "⚙️")
                                status_parts = []
                                if data.get("next"):
                                    status_parts.append(f"→ {data['next']}")
                                if data.get("has_research"):
                                    status_parts.append("📚 research collected")
                                if data.get("has_draft"):
                                    status_parts.append("📝 draft ready")
                                if data.get("critic_feedback"):
                                    fb = data["critic_feedback"][:80]
                                    status_parts.append(f"💬 _{fb}_")
                                status = " | ".join(status_parts) if status_parts else ""
                                events.append(f"{icon} **{agent.capitalize()}** (iter {data.get('iteration',0)}) {status}")
                                if data.get("final_answer"):
                                    answer_placeholder.markdown(data["final_answer"])
                            elif event_type == "awaiting_human_approval":
                                events.append("⏸️ **Paused for human review** — Use /research/resume to continue")
                            elif event_type == "complete":
                                events.append("✅ **Research complete**")
                            elif event_type == "error":
                                events.append(f"❌ Error: {data.get('message')}")

                            trace_placeholder.markdown("\n\n".join(events))
                        except json.JSONDecodeError:
                            pass
    else:
        with st.spinner("Agents working..."):
            resp = httpx.post(f"{API_URL}/research", json={
                "task": task,
                "max_iterations": max_iter,
            }, timeout=120)
            data = resp.json()
            answer_placeholder.markdown(data.get("final_answer", "No answer generated"))
            st.info(f"Iterations: {data.get('iterations')} | Notes collected: {data.get('research_notes_count')}")

elif run_btn:
    st.warning("Please enter a research task.")

# Thread inspector
with st.expander("🔧 Thread Inspector"):
    thread_id_input = st.text_input("Thread ID")
    if st.button("Inspect") and thread_id_input:
        try:
            r = httpx.get(f"{API_URL}/research/{thread_id_input}/state")
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
