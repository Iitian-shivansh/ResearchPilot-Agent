"""
Tool definitions for the LangGraph agent.
"""
import sys
import io
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

@tool
def query_knowledge_base(query: str) -> str:
    """
    Search the AI Knowledge Workspace vector database for information.
    Use this tool for domain-specific questions related to indexed documents.
    """
    try:
        # Initialize connections
        client = QdrantClient(url="http://localhost:6333")
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
    # Block dangerous operations before execution
    BLOCKED_KEYWORDS = [
        "os.system", "subprocess", "shutil", "open(", "eval(", "exec(",
        "__import__", "importlib", "pathlib", "socket", "requests",
        "urllib", "http.client", "ftplib", "smtplib", "ctypes",
        "multiprocessing", "threading", "signal",
    ]
    code_lower = code.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword.lower() in code_lower:
            return f"Execution blocked: '{keyword}' is not allowed in the sandbox for security reasons."

    # Create a safe execution environment (restricted globals/locals)
    safe_globals = {
        "__builtins__": {
            "print": print, "len": len, "range": range, "int": int, "float": float, 
            "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "bool": bool, "sum": sum, "min": min, "max": max, "abs": abs,
            "round": round, "any": any, "all": all, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
        },
        "math": __import__("math"),
        "json": __import__("json")
    }
    
    # Capture standard output
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    
    MAX_OUTPUT_LENGTH = 5000
    
    try:
        exec(code, safe_globals)
        output = redirected_output.getvalue()
        if not output.strip():
            return "Code executed successfully, but no output was printed."
        if len(output) > MAX_OUTPUT_LENGTH:
            return output[:MAX_OUTPUT_LENGTH] + "\n\n... [output truncated for safety]"
        return output
    except NameError as e:
        if "query_knowledge_base" in str(e) or "tavily_search_results_json" in str(e):
            return "Execution error: NameError - tool functions cannot be called inside the sandbox. Pass retrieved text as a string literal instead."
        return f"Execution error: {e}"
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        sys.stdout = old_stdout

def get_tools():
    """
    Returns a list of tools available for the agent.
    """
    search_tool = TavilySearchResults(max_results=3)
    return [search_tool, query_knowledge_base, execute_python]
