"""
Tool definitions for the LangGraph agent.
"""
import json
import logging
import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.sandbox import run_code_in_subprocess, SandboxConfig

logger = logging.getLogger(__name__)

@tool
def query_knowledge_base(query: str) -> str:
    """
    Search the AI Knowledge Workspace vector database for information.
    Use this tool for domain-specific questions related to indexed documents.
    """
    try:
        # Initialize connections
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Embed query
        query_vector = embeddings_model.embed_query(query)
        
        # Search Qdrant
        results = client.query_points(
            collection_name="documents",
            query=query_vector,
            limit=5,
            with_payload=True
        )
        
        # Format results
        if not results.points:
            return "No relevant information found in the knowledge base."
            
        formatted_results = []
        MAX_TOTAL_CHARS = 1200 # Keep well within Groq's 6000 TPM limit and JSON formatting limits
        current_chars = 0
        
        for point in results.points:
            payload = point.payload or {}
            raw_text = payload.get("text", "")
            text = raw_text[:500]
            if len(raw_text) > 500:
                text += "... [truncated]"
                
            doc_id = payload.get("document_id", "Unknown")
            page = payload.get("chunk_index", 0)
            score = point.score
            
            chunk_str = f"[Doc: {doc_id} | Page: {page} | Score: {score:.3f}]\n{text}"
            
            if current_chars + len(chunk_str) > MAX_TOTAL_CHARS:
                formatted_results.append("[Remaining results truncated to fit token limits]")
                break
                
            formatted_results.append(chunk_str)
            current_chars += len(chunk_str)
            
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error querying knowledge base: {e}"

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code in a sandboxed environment and return the standard output.
    Use this tool to perform calculations, data analysis, or manipulate retrieved information.
    The code should use `print()` to output results.
    execute_python(code="text = '''retrieved text...'''\nprint(len(text))").
    """
    # Execute in an isolated subprocess via the sandbox module.
    # All validation (blocklist, code length), timeout enforcement,
    # and output capture are handled by run_code_in_subprocess().
    # See src/sandbox.py for configurable limits (SandboxConfig).
    result = run_code_in_subprocess(code)

    # Log safe metadata only — never log the code itself or secrets
    logger.info(
        "execute_python: success=%s duration_ms=%.1f error_type=%s "
        "code_length=%d output_length=%d",
        result.success,
        result.execution_time_ms,
        result.error_type,
        len(code),
        len(result.stdout),
    )

    # Convert structured result back to string for LangGraph compatibility.
    # The agent expects a plain string return — same interface as before.
    if result.success:
        if not result.stdout.strip():
            return "Code executed successfully, but no output was printed."
        return result.stdout
    else:
        # Return the error in a format consistent with the old implementation
        # so the Executor/Critic can interpret it the same way.
        error_msg = result.stderr or f"Execution failed: {result.error_type}"
        return f"Execution error: {error_msg}"

def get_tools():
    """
    Returns a list of tools available for the agent.
    """
    search_tool = TavilySearch(max_results=3)
    return [search_tool, query_knowledge_base, execute_python]
