import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import create_agent_graph
from src.agent import is_fast_query
from src.attachments import attachment_context, load_task_attachments
from src.modes import MODES
from src.report import build_markdown_report

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon=":material/auto_awesome:", layout="centered")
st.session_state.setdefault("conversation", [])
st.session_state.setdefault("last_result", None)

st.title("Research Agent")
st.caption("Plan, execute, review, and continue a research conversation.")

with st.sidebar:
    st.header("Research settings")
    mode = st.selectbox("Research mode", list(MODES), index=1, key="research_mode")
    st.caption(MODES[mode].instruction)
    st.info("Attachments are temporary task context. They are not written to or inserted into Qdrant.")

for turn in st.session_state.conversation:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

submission = st.chat_input(
    "Ask a research question or follow up...",
    accept_file="multiple",
    file_type=["txt", "md"],
)


def _submission_text(value) -> str:
    return str(getattr(value, "text", "") or "").strip()


if submission:
    query = _submission_text(submission)
    files = list(getattr(submission, "files", ()) or ())
    if not query:
        st.error("Enter a research question before submitting.")
        st.stop()

    try:
        attachments = load_task_attachments(files)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.session_state.conversation.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        if attachments:
            st.caption("Task context: " + ", ".join(item.name for item in attachments))

    try:
        graph = create_agent_graph()
    except Exception as error:
        st.error(f"Failed to initialize the agent: {error}")
        st.stop()

    messages = []
    for turn in st.session_state.conversation:
        message_type = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_type(content=turn["content"]))

    trace = []
    final_answer = ""
    run_status = "running"
    critic_status = "pending"
    with st.chat_message("assistant"):
        status = st.status("Running research workflow", expanded=True)
        try:
            for event in graph.stream(
                {
                    "messages": messages,
                    "mode": mode,
                    "context": attachment_context(attachments),
                    "fast_path": not attachments and is_fast_query(query),
                }
            ):
                for node_name, update in event.items():
                    update = update or {}
                    run_status = update.get("run_status", run_status)
                    critic_status = update.get("critic_status", critic_status)
                    trace.extend(update.get("trace", []))
                    status.write(f"{node_name}: {run_status}")

                    if node_name in {"fast_answer", "executor", "synthesis"}:
                        for message in update.get("messages", []):
                            if isinstance(message, AIMessage) and not message.tool_calls and message.content:
                                final_answer = str(message.content)
                    elif node_name == "tools":
                        for message in update.get("messages", []):
                            if isinstance(message, ToolMessage):
                                status.write(f"tool completed: {message.name}")
            status.update(label=f"Research {run_status}", state="complete")
        except Exception as error:
            status.update(label="Research failed", state="error")
            st.error(f"Research run failed: {error}")
            st.stop()

        if final_answer:
            st.markdown(final_answer)
        if critic_status == "unavailable":
            st.warning("Automated review was unavailable; treat this answer as unverified.")
        elif run_status in {"planner_failed", "executor_failed", "synthesis_failed"}:
            st.error(f"Research ended with status: `{run_status}`")

    if final_answer:
        st.session_state.conversation.append({"role": "assistant", "content": final_answer})
    st.session_state.last_result = {
        "query": query,
        "answer": final_answer,
        "mode": mode,
        "trace": trace,
        "attachments": [item.name for item in attachments],
    }
    st.rerun()

if st.session_state.last_result:
    result = st.session_state.last_result
    report = build_markdown_report(
        result["query"],
        result["answer"],
        result["mode"],
        result["trace"],
        result["attachments"],
    )
    st.download_button(
        "Download Markdown report",
        data=report,
        file_name="research-report.md",
        mime="text/markdown",
        icon=":material/download:",
    )
