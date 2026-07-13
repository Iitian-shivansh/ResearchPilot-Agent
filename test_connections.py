import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_qdrant():
    print("Testing Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print(f"SUCCESS: Connected to Qdrant. Found collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"FAILED: Could not connect to Qdrant. Error: {e}")

def test_gemini_embedding():
    print("\nTesting Gemini Embeddings...")
    if not os.getenv("GEMINI_API_KEY"):
        print("FAILED: GEMINI_API_KEY is not set.")
        return
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        result = embeddings.embed_query("Hello world")
        print(f"SUCCESS: Successfully generated embedding of length {len(result)}")
    except Exception as e:
        print(f"FAILED: Could not generate embedding. Error: {e}")

if __name__ == "__main__":
    test_qdrant()
    test_gemini_embedding()
