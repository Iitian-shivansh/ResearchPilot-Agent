"""Unit tests for retrieval configuration and structured evidence output."""

import json
import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.tools import query_knowledge_base


class FakeEmbeddings:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]


class FakeClient:
    def __init__(self):
        self.kwargs = None

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "text": "Relevant chunk",
                        "document_id": "notes.md",
                        "chunk_index": 2,
                    },
                    score=0.87,
                )
            ]
        )


class TestKnowledgeBaseTool(TestCase):
    def test_retrieval_passes_limit_and_score_threshold(self):
        client = FakeClient()
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "https://example", "QDRANT_API_KEY": "key",
             "KB_TOP_K": "3", "KB_SCORE_THRESHOLD": "0.72"},
        ), patch("src.tools.QdrantClient", return_value=client), patch(
            "src.tools.GoogleGenerativeAIEmbeddings",
            return_value=FakeEmbeddings(),
        ):
            result = json.loads(query_knowledge_base.invoke({"query": "question"}))

        self.assertTrue(result["ok"])
        self.assertEqual(client.kwargs["limit"], 3)
        self.assertEqual(client.kwargs["score_threshold"], 0.72)
        self.assertEqual(result["evidence"][0]["evidence_id"], "KB-1")
        self.assertEqual(result["evidence"][0]["source_id"], "notes.md")

    def test_invalid_retrieval_settings_fall_back_safely(self):
        client = FakeClient()
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "https://example", "QDRANT_API_KEY": "key",
             "KB_TOP_K": "invalid", "KB_SCORE_THRESHOLD": "invalid"},
        ), patch("src.tools.QdrantClient", return_value=client), patch(
            "src.tools.GoogleGenerativeAIEmbeddings",
            return_value=FakeEmbeddings(),
        ):
            query_knowledge_base.invoke({"query": "question"})

        self.assertEqual(client.kwargs["limit"], 5)
        self.assertEqual(client.kwargs["score_threshold"], 0.0)
