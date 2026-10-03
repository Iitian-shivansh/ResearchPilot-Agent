"""Unit tests for the local-document Qdrant ingestion pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from src.ingestion import (
    IngestionConfig,
    chunk_text,
    discover_documents,
    ingest_directory,
)


class FakeEmbeddings:
    def __init__(self, size: int = 3) -> None:
        self.size = size
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text) % 10), 1.0, 0.5][: self.size] for text in texts]


class FakeQdrant:
    def __init__(self, exists: bool = False, vector_size: int = 3) -> None:
        self.exists = exists
        self.vector_size = vector_size
        self.created: list[dict] = []
        self.upserts: list[list] = []
        self.deletes: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs) -> None:
        self.exists = True
        self.created.append(kwargs)

    def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=self.vector_size))
            )
        )

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs["points"])

    def delete(self, **kwargs) -> None:
        self.deletes.append(kwargs)


class TestIngestion(TestCase):
    def test_chunk_text_normalizes_whitespace_and_overlaps(self):
        chunks = chunk_text("one   two three four", chunk_size=6, chunk_overlap=2)
        self.assertEqual(chunks, ["one tw", "two th", "three ", "e four", "ur"])

    def test_discover_documents_is_recursive_and_sorted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.md").write_text("z", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "a.TXT").write_text("a", encoding="utf-8")
            (root / "ignored.pdf").write_text("ignored", encoding="utf-8")

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in discover_documents(root)],
                ["nested/a.TXT", "z.md"],
            )

    def test_ingest_creates_collection_and_uploads_expected_payloads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.md").write_text("alpha beta gamma", encoding="utf-8")
            client = FakeQdrant()
            embeddings = FakeEmbeddings()

            count = ingest_directory(
                root,
                client,
                embeddings,
                IngestionConfig(chunk_size=100, chunk_overlap=0),
            )

            self.assertEqual(count, 1)
            self.assertEqual(client.created[0]["collection_name"], "documents")
            point = client.upserts[0][0]
            self.assertEqual(point.payload["text"], "alpha beta gamma")
            self.assertEqual(point.payload["document_id"], "notes.md")
            self.assertEqual(point.payload["chunk_index"], 0)
            self.assertEqual(len(point.payload["source_hash"]), 64)
            self.assertTrue(point.payload["ingestion_run_id"])
            self.assertEqual(len(point.vector), 3)
            self.assertEqual(len(client.deletes), 1)

    def test_ingest_validates_existing_collection_vector_size(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("content", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vector size"):
                ingest_directory(
                    root,
                    FakeQdrant(exists=True, vector_size=99),
                    FakeEmbeddings(),
                )

    def test_ingest_is_safe_to_repeat_with_deterministic_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("repeatable content", encoding="utf-8")
            first_client = FakeQdrant()
            second_client = FakeQdrant()
            embeddings = FakeEmbeddings()

            ingest_directory(root, first_client, embeddings)
            ingest_directory(root, second_client, embeddings)

            self.assertEqual(
                first_client.upserts[0][0].id,
                second_client.upserts[0][0].id,
            )

    def test_ingest_deletes_previous_chunks_when_source_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "notes.txt"
            path.write_text("old content", encoding="utf-8")
            client = FakeQdrant()
            embeddings = FakeEmbeddings()

            ingest_directory(root, client, embeddings)
            first_hash = client.upserts[0][0].payload["source_hash"]
            path.write_text("new content", encoding="utf-8")
            ingest_directory(root, client, embeddings)

            second_hash = client.upserts[-1][0].payload["source_hash"]
            self.assertNotEqual(first_hash, second_hash)
            self.assertEqual(len(client.deletes), 2)
