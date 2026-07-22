import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent import create_agent_graph

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🤖", layout="centered")

st.title("Research Agent (Plan & Execute)")
st.write("Ask complex questions. The agent will plan, execute tools, and self-correct.")

query = st.text_input("Enter your research question:")

if st.button("Run Agent"):
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            app = create_agent_graph()
        except Exception as e:
            st.error(f"❌ Failed to initialize the agent: {e}")
            st.info("Make sure your API keys are correctly set in the `.env` file.")
            st.stop()
        
        initial_state = {"messages": [HumanMessage(content=query)]}
        
        st.markdown("### Execution Trace")
        
        final_answer = ""
        
        try:
            # We will use st.status blocks for each major step
            with st.spinner("🔄 Agent is thinking... This may take a minute on Groq's free tier."):
                for event in app.stream(initial_state):
                    for node_name, state_update in event.items():
                        if state_update is None:
                            state_update = {}
                            
                        if node_name == "planner":
                            with st.expander("📝 Planner: Generated Sub-tasks", expanded=True):
                                st.markdown(state_update.get("plan", "No plan generated."))
                                
                        elif node_name == "executor":
                            messages = state_update.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, AIMessage):
                                    if msg.tool_calls:
                                        with st.expander("🛠️ Executor: Tool Calls", expanded=False):
                                            for tool_call in msg.tool_calls:
                                                st.write(f"**Tool:** `{tool_call['name']}`")
                                                st.write(f"**Args:** `{tool_call['args']}`")
                                    else:
                                        with st.expander("✍️ Executor: Draft Answer", expanded=False):
                                            st.markdown(msg.content)
                                            final_answer = msg.content
                                            
                        elif node_name == "critic":
                            messages = state_update.get("messages", [])
                            if messages and messages[-1].type == "human" and "Critic Feedback:" in str(messages[-1].content):
                                with st.status("❌ Critic: Revision Requested", state="error"):
                                    st.write(messages[-1].content)
                            else:
                                with st.status("✅ Critic: Approved!", state="complete"):
                                    st.write("The draft is complete and correctly cited.")
                                    
                        elif node_name == "tools":
                            messages = state_update.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, ToolMessage):
                                    with st.expander(f"📥 Tool Output: {msg.name}", expanded=False):
                                        content = str(msg.content)
                                        if len(content) > 500:
                                            content = content[:500] + "\n\n... [truncated]"
                                        st.text(content)
            
            st.markdown("---")
            st.markdown("### Final Answer")
            st.markdown(final_answer)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                st.error("⏳ Rate limit hit on Groq's free tier. Please wait 60 seconds and try again.")
            elif "400" in error_msg or "BadRequest" in error_msg:
                st.error("⚠️ The LLM returned a formatting error. Try rephrasing your question more concisely.")
            elif "401" in error_msg or "auth" in error_msg.lower():
                st.error("🔑 Authentication failed. Check your GROQ_API_KEY in the `.env` file.")
            else:
                st.error(f"❌ An error occurred: {error_msg}")
            st.info("💡 Tip: Try a simpler question, or check that your API keys are valid.")

