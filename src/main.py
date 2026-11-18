"""
Main entry point for the LangGraph agent CLI.

This script demonstrates how to interact with the agent, printing its step-by-step
reasoning and tool calls before reporting the final, verified outcome.
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import create_agent_graph
from src.modes import MODES

VERIFIED_STATUSES = {"completed", "completed_after_revision", "fast_completed"}


def print_message_trace(event) -> None:
    """Format one streamed LangGraph event so the reasoning steps are visible."""

    for node_name, state_update in event.items():
        print(f"\n--- Output from Node: {node_name} ---")
        state_update = state_update or {}
        messages = state_update.get("messages", [])

        if node_name == "planner":
            print(f"Generated Plan:\n{state_update.get('plan', '')}")

        elif node_name == "executor":
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"Executor decided to use tool: {tool_call['name']}")
                            print(f"   With arguments: {tool_call['args']}")
                    else:
                        print(f"Executor draft answer:\n{msg.content}")

        elif node_name in {"synthesis", "fast_answer"}:
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    label = "Forced synthesis" if node_name == "synthesis" else "Direct answer"
                    print(f"{label}:\n{msg.content}")

        elif node_name == "critic":
            critic_status = state_update.get("critic_status")
            if messages and messages[-1].type == "human" and "Critic Feedback:" in str(messages[-1].content):
                print(f"Critic requested revision:\n{messages[-1].content}")
            elif critic_status == "approved":
                print("Critic APPROVED the answer.")
            elif critic_status == "rejected":
                print("Critic REJECTED the answer: it could not be verified.")
            elif critic_status == "revision_limit":
                print("Critic did not approve the revised answer (revision limit reached).")
            else:
                print("Critic unavailable or returned an invalid decision: draft is unverified.")

        elif node_name == "tools":
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    print(f"Tool '{msg.name}' returned:")
                    content = str(msg.content)
                    if len(content) > 300:
                        content = content[:300] + "... [truncated]"
                    print(f"   {content}")


def main() -> int:
    load_dotenv()

    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        print("Error: Missing GROQ_API_KEY, TAVILY_API_KEY, or GEMINI_API_KEY in .env file.")
        print("Please check your .env configuration.")
        return 2

    parser = argparse.ArgumentParser(description="Run the ResearchPilot research agent.")
    parser.add_argument("query", type=str, nargs="?", help="The research question to ask the agent.")
    parser.add_argument("--mode", choices=list(MODES), default="Deep", help="Research mode (default: Deep).")
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Welcome to the LangGraph Research Agent!")
        query = input("Please enter your research question: ")

    if not query.strip():
        print("Empty question provided. Exiting.")
        return 2

    print(f"\nProcessing query: '{query}' (mode: {args.mode})\n")
    app = create_agent_graph()
    initial_state = {"messages": [HumanMessage(content=query)], "mode": args.mode}

    print("Starting execution trace...\n" + "=" * 40)
    run_status = "running"
    for event in app.stream(initial_state):
        print_message_trace(event)
        for update in event.values():
            run_status = (update or {}).get("run_status", run_status)

    verified = run_status in VERIFIED_STATUSES
    print("\n" + "=" * 40)
    print(f"Execution complete. Final status: {run_status} ({'verified' if verified else 'NOT verified'})")
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
