"""
Tool definitions for the LangGraph agent.
"""

from langchain_community.tools.tavily_search import TavilySearchResults

def get_tools():
    """
    Returns a list of tools available for the agent.
    
    Currently includes:
    - TavilySearchResults: A web search tool to find up-to-date information.
    """
    # Create the search tool. By default, it uses the TAVILY_API_KEY environment variable.
    search_tool = TavilySearchResults(max_results=3)
    
    return [search_tool]
