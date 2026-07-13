"""
Main entry point for the LangGraph agent CLI.

This script demonstrates how to interact with the agent, printing its step-by-step
reasoning and tool calls before yielding the final answer.
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent import create_agent_graph

def print_message_trace(event):
    """
    Helper function to nicely format and print the stream of events
    from the LangGraph execution to show reasoning steps.
    """
    for node_name, state_update in event.items():
        print(f"\n--- Output from Node: {node_name} ---")
        
        if state_update is None:
            state_update = {}
            
        if node_name == "planner":
            print(f"Generated Plan:\n{state_update.get('plan', '')}")
            
        elif node_name == "executor":
            messages = state_update.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"Executor decided to use tool: {tool_call['name']}")
                            print(f"   With arguments: {tool_call['args']}")
                    else:
                        print(f"Executor draft answer:\n{msg.content}")
                        
        elif node_name == "critic":
            messages = state_update.get("messages", [])
            if messages and messages[-1].type == "human" and "Critic Feedback:" in str(messages[-1].content):
                print(f"Critic requested revision:\n{messages[-1].content}")
            else:
                print(f"Critic APPROVED the answer.")
                
        elif node_name == "tools":
            messages = state_update.get("messages", [])
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    print(f"Tool '{msg.name}' returned:")
                    # Truncate long results for readability
                    content = str(msg.content)
                    if len(content) > 300:
                        content = content[:300] + "... [truncated]"
                    print(f"   {content}")

def main():
    # Load environment variables (GROQ_API_KEY, TAVILY_API_KEY, GEMINI_API_KEY)
    load_dotenv()
    
    # Ensure keys are loaded
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        print("Error: Missing GROQ_API_KEY, TAVILY_API_KEY, or GEMINI_API_KEY in .env file.")
        print("Please check your .env configuration.")
        return

    parser = argparse.ArgumentParser(description="Run the Research ReAct Agent.")
    parser.add_argument(
        "query", 
        type=str, 
        nargs="?", 
        help="The research question to ask the agent."
    )
    args = parser.parse_args()

    # Get query from args or prompt the user
    query = args.query
    if not query:
        print("Welcome to the LangGraph Research Agent!")
        query = input("Please enter your research question: ")

    if not query.strip():
        print("Empty question provided. Exiting.")
        return

    print(f"\nProcessing query: '{query}'\n")

    # Initialize the agent graph
    app = create_agent_graph()
    
    # The initial state is just the user's message
    initial_state = {"messages": [HumanMessage(content=query)]}
    
    # Stream the execution to trace reasoning
    print("Starting execution trace...\n" + "="*40)
    
    final_state = None
    for event in app.stream(initial_state):
        print_message_trace(event)
        
    print("\n" + "="*40 + "\nExecution complete.")

if __name__ == "__main__":
    main()
