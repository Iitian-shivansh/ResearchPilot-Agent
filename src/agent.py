"""
Agent logic using LangGraph.

This module defines the ReAct agent, which consists of:
1. An LLM node (the agent reasoning).
2. A tool execution node (for web search).

The agent uses a loop: 
- It receives a message (e.g., a user query).
- The LLM decides whether to answer directly or call a tool.
- If it calls a tool, the ToolNode executes it, appends the result to the state, and passes it back to the LLM.
- Once the LLM generates a final response (without a tool call), the execution ends.
"""

from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from src.tools import get_tools

def create_agent_graph():
    """
    Creates and compiles the LangGraph application for the ReAct agent.
    
    Decision flow:
    - START -> agent: The entry point passes the initial state to the agent node.
    - agent: The LLM examines the messages. If it needs to search the web, it outputs a tool call.
             If it has enough info, it outputs the final answer.
    - should_continue: A conditional edge checks the agent's output.
                       If there's a tool call -> route to 'tools'.
                       If there's no tool call -> route to END.
    - tools: Executes the requested tool (Tavily search), gets the result, and goes back to 'agent'.
    """
    
    # 1. Initialize the LLM and tools
    # We use llama-3.3-70b-versatile as requested. Make sure GROQ_API_KEY is in the environment.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = get_tools()
    
    # Bind the tools to the LLM so it knows what it can call
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Define the core agent node function
    def call_model(state: MessagesState):
        """
        Invokes the LLM with the current list of messages.
        Returns a dictionary with a new message to append to the state.
        """
        messages = state["messages"]
        system_message = SystemMessage(
            content="You are a research assistant. Only use the web search tool when you don't know the answer with high confidence, or when the question requires current/real-time information (e.g., weather, news, recent events, prices). For well-established facts you're confident about, answer directly without searching."
        )
        
        # Prepend the system message to the message history
        full_messages = [system_message] + messages
        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
        
    # 3. Define the routing logic
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        """
        Determines the next step based on the last message from the agent.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM made a tool call, we must execute the tools
        if last_message.tool_calls:
            return "tools"
            
        # Otherwise, the agent has finished reasoning and returned the answer
        return END

    # 4. Construct the Graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Conditional routing from agent to either tools or END
    workflow.add_conditional_edges("agent", should_continue)
    
    # After executing tools, always return to the agent to reason about the tool output
    workflow.add_edge("tools", "agent")
    
    # Compile the graph into an executable LangChain Runnable
    app = workflow.compile()
    
    return app
