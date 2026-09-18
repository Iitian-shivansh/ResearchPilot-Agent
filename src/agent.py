"""
Agent logic using LangGraph.

This module defines the Plan-and-Execute agent:
1. Planner: Breaks query into sub-tasks.
2. Executor: Uses tools to solve the plan sequentially.
3. Critic: Reviews draft for completeness and citations.
"""

import time
from typing import Annotated, Literal, TypedDict
import groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.tools import get_tools

# --- Agent-level constants ---
# Maximum number of executor → tools → executor round-trips before forcing
# the executor to produce a final answer. Prevents infinite looping when the
# LLM keeps requesting tools instead of synthesizing an answer.
MAX_TOOL_ROUNDS = 4

# Seconds to wait before retrying after a Groq rate-limit (429) error.
RATE_LIMIT_RETRY_DELAY = 5

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    revision_count: int

def create_agent_graph():
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    def planner_node(state: AgentState):
        messages = state["messages"]
        query = messages[0].content if messages else ""
        
        sys_msg = SystemMessage(
            content=(
                "You are a planning assistant. Break the user's research question "
                "into concrete sub-tasks.\n"
                "- For simple factual questions (e.g. 'What is X?', 'Who is Y?'), "
                "output just 1 sub-task.\n"
                "- For complex multi-part questions, output 2 to 4 sub-tasks.\n"
                "Output ONLY the numbered list of sub-tasks, nothing else. "
                "Keep it brief to save tokens."
            )
        )
        
        try:
            response = llm.invoke([sys_msg, HumanMessage(content=query)])
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                print(f"\n[Warning] Rate limit hit in planner, retrying in {RATE_LIMIT_RETRY_DELAY}s...")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                response = llm.invoke([sys_msg, HumanMessage(content=query)])
            else:
                raise
        return {"plan": response.content, "revision_count": 0}
        
    def executor_node(state: AgentState):
        messages = state["messages"]
        plan = state.get("plan", "")
        
        sys_msg = SystemMessage(
            content=(
                "You are an executor assistant. Follow this plan sequentially to solve the user's original query:\n"
                f"{plan}\n\n"
                "Only use tools when you do not know the answer with high confidence.\n"
                "For simple factual questions you can answer directly without tools.\n"
                "When using execute_python based on knowledge base text, pass the text directly as a string literal. "
                "Never call tools from inside the Python code.\n"
                "Once you have completed the plan, synthesize a final cohesive draft answer."
            )
        )
        
        full_messages = [sys_msg] + messages
        
        try:
            response = llm_with_tools.invoke(full_messages)
        except groq.BadRequestError as e:
            print(f"\n[Warning] Groq formatting error caught: {e}. Retrying once with simplified constraints...")
            retry_msg = HumanMessage(content="System Note: Your previous tool call failed due to malformed JSON formatting. Keep any string arguments concise and avoid unnecessary special characters or extremely long inline text.")
            try:
                response = llm_with_tools.invoke(full_messages + [retry_msg])
            except Exception as e2:
                print(f"\n[Error] Tool call formatting failed after retry.")
                return {"messages": [AIMessage(content="Tool call formatting failed after retry \u2014 try rephrasing your question or breaking it into smaller steps.")]}
        except Exception as e:
            error_str = str(e)
            # Handle rate-limit (429) errors with a retry + backoff
            if "429" in error_str or "rate" in error_str.lower():
                print(f"\n[Warning] Rate limit hit, retrying in {RATE_LIMIT_RETRY_DELAY}s...")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                try:
                    response = llm_with_tools.invoke(full_messages)
                except Exception as e2:
                    print(f"\n[Error] LLM call failed after rate-limit retry: {e2}")
                    return {"messages": [AIMessage(content="Rate limit exceeded on the LLM API. Please wait a moment and try again.")]}
            else:
                print(f"\n[Error] LLM call failed: {e}")
                return {"messages": [AIMessage(content=f"An unexpected error occurred during the LLM call: {type(e).__name__}")]}
            
        return {"messages": [response]}
        
    def should_continue_executor(state: AgentState) -> Literal["tools", "critic", END]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            # --- Hard cap on total tool-call rounds ---
            # Count how many tool-call round-trips have occurred so far.
            # Each round is: executor sends tool_calls → tools node returns results.
            tool_round_count = sum(
                1 for msg in messages if msg.type == "tool"
            )
            if tool_round_count >= MAX_TOOL_ROUNDS:
                print(f"\n[Safeguard] Tool rounds exceeded limit ({MAX_TOOL_ROUNDS} max). Stopping loop.")
                return END

            # --- Consecutive error cap (original safeguard) ---
            error_count = 0
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                if msg.type == "tool":
                    if "Execution error:" in str(msg.content) or "Error querying" in str(msg.content):
                        error_count += 1
                    else:
                        break  # Found a successful tool call, stop counting
                elif msg.type == "ai":
                    continue
                else:
                    break
            
            if error_count >= 2:
                print("\n[Safeguard] Tool errors exceeded retry limit (2 max). Stopping loop.")
                return END
                
            return "tools"
            
        return "critic"
        
    def critic_node(state: AgentState):
        messages = state["messages"]
        original_query = messages[0].content
        draft = messages[-1].content
        plan = state.get("plan", "")
        
        sys_msg = SystemMessage(
            content=(
                "You are a reviewer. Review the Draft Answer against the Original Query and Plan.\n"
                "Tasks:\n"
                "1. Check if the query was answered correctly and completely.\n"
                "2. For complex research questions, verify claims are supported by citations from tools.\n"
                "3. For simple factual questions (e.g. 'What is X?'), well-known facts do NOT need tool citations.\n"
                "If the draft answers the query correctly, output EXACTLY: 'APPROVED'.\n"
                "Only reject if the answer is factually wrong, incomplete, or a complex question lacks sources. "
                "If rejecting, provide a BRIEF 1-sentence critique. Do NOT output 'APPROVED'."
            )
        )
        
        prompt = f"Original Query: {original_query}\nPlan: {plan}\nDraft Answer: {draft}"
        try:
            response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                print(f"\n[Warning] Rate limit hit in critic, retrying in {RATE_LIMIT_RETRY_DELAY}s...")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
                try:
                    response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
                except Exception:
                    # If still failing, just approve to avoid an infinite error loop
                    print("\n[Warning] Critic rate-limited after retry, auto-approving.")
                    return {}
            else:
                print(f"\n[Warning] Critic failed: {e}, auto-approving to avoid crash.")
                return {}
        
        review = response.content.strip()
        if "APPROVED" in review.upper() or state.get("revision_count", 0) >= 1:
            return {} # No state changes, proceed to END
            
        # Needs revision
        return {
            "messages": [HumanMessage(content=f"Critic Feedback: {review}. Please fix these issues and provide an updated final answer.")],
            "revision_count": state.get("revision_count", 0) + 1
        }
        
    def should_loop_critic(state: AgentState) -> Literal["executor", END]:
        # If the last message is a HumanMessage from the critic, we loop back
        if state["messages"][-1].type == "human" and "Critic Feedback:" in str(state["messages"][-1].content):
            return "executor"
        return END

    # Construct the Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("critic", critic_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    
    # Executor loops with tools until it returns a draft, then goes to critic
    workflow.add_conditional_edges("executor", should_continue_executor)
    workflow.add_edge("tools", "executor")
    
    # Critic evaluates the draft, either ends or loops back to executor once
    workflow.add_conditional_edges("critic", should_loop_critic)
    
    app = workflow.compile()
    
    return app

