"""
Tool definitions for the LangGraph agent.
"""
import logging
import os
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.contracts import Evidence, ToolResult
from src.sandbox import run_code_in_subprocess

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
            return ToolResult(
                ok=True,
                tool_name="query_knowledge_base",
                data={"message": "No relevant information found in the knowledge base."},
            ).to_json()
            
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
            chunk_index = payload.get("chunk_index", 0)
            score = point.score
            
            chunk_str = (
                f"[Evidence: KB-{len(formatted_results) + 1} | Doc: {doc_id} | "
                f"Chunk: {chunk_index} | Score: {score:.3f}]\n{text}"
            )
            
            if current_chars + len(chunk_str) > MAX_TOTAL_CHARS:
                formatted_results.append("[Remaining results truncated to fit token limits]")
                break
                
            formatted_results.append(chunk_str)
            current_chars += len(chunk_str)
            
        return ToolResult(
            ok=True,
            tool_name="query_knowledge_base",
            data={"results": "\n\n---\n\n".join(formatted_results)},
            evidence=tuple(
                Evidence(
                    evidence_id=f"KB-{index}",
                    source_type="knowledge_base",
                    source_id=str((point.payload or {}).get("document_id", "Unknown")),
                    excerpt=str((point.payload or {}).get("text", ""))[:500],
                    score=point.score,
                )
                for index, point in enumerate(results.points, start=1)
            ),
        ).to_json()
    except Exception as e:
        logger.exception("Knowledge-base query failed")
        return ToolResult(
            ok=False,
            tool_name="query_knowledge_base",
            error_type=type(e).__name__,
            error_message=str(e),
        ).to_json()

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
        return ToolResult(
            ok=True,
            tool_name="execute_python",
            data={"stdout": result.stdout or "", "printed": bool(result.stdout.strip())},
            evidence=(
                Evidence(
                    evidence_id="CALC-1",
                    source_type="calculation",
                    source_id="execute_python",
                    excerpt=result.stdout[:1000],
                ),
            ),
        ).to_json()
    else:
        return ToolResult(
            ok=False,
            tool_name="execute_python",
            error_type=result.error_type,
            error_message=result.stderr or "Execution failed without an error message",
        ).to_json()

def get_tools():
    """
    Returns a list of tools available for the agent.
    """
    from langchain_tavily import TavilySearch

    search_tool = TavilySearch(max_results=3)
    return [search_tool, query_knowledge_base, execute_python]
