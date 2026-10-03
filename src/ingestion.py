"""Document ingestion helpers for the Qdrant knowledge base."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


class EmbeddingsProtocol(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for one ingestion run."""

    collection_name: str = DEFAULT_COLLECTION_NAME
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")


def discover_documents(input_path: str | Path) -> list[Path]:
    """Return supported documents below a file or directory, in stable order."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document type '{path.suffix}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [path]

    documents = sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda candidate: candidate.as_posix().lower(),
    )
    if not documents:
        raise ValueError(
            f"No supported documents found in {path}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return documents


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character chunks without returning blanks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    return [
        normalized[start : start + chunk_size]
        for start in range(0, len(normalized), step)
        if normalized[start : start + chunk_size]
    ]


def _document_id(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return relative.as_posix()


def _point_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}\0{chunk_index}\0{text}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _delete_source_chunks(
    client: QdrantClient,
    collection_name: str,
    document_id: str,
) -> None:
    """Remove all prior chunks for one source before replacing it."""

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
        wait=True,
    )


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    info = client.get_collection(collection_name)
    configured_size = info.config.params.vectors.size
    if configured_size != vector_size:
        raise ValueError(
            f"Collection '{collection_name}' has vector size {configured_size}, "
            f"but the embedding model returned vectors of size {vector_size}."
        )


def _upsert_batch(
    client: QdrantClient,
    embeddings: EmbeddingsProtocol,
    collection_name: str,
    batch: Sequence[tuple[str, int, str, str]],
    ingestion_run_id: str,
) -> int:
    texts = [item[2] for item in batch]
    vectors = embeddings.embed_documents(texts)
    if len(vectors) != len(batch):
        raise ValueError(
            f"Embedding service returned {len(vectors)} vectors for {len(batch)} chunks"
        )
    if not vectors or not vectors[0]:
        raise ValueError("Embedding service returned an empty vector")

    points = [
        PointStruct(
            id=_point_id(document_id, chunk_index, text),
            vector=vector,
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source_hash": source_hash,
                "ingestion_run_id": ingestion_run_id,
            },
        )
        for (document_id, chunk_index, text, source_hash), vector in zip(batch, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points)


def ingest_directory(
    input_path: str | Path,
    client: QdrantClient,
    embeddings: EmbeddingsProtocol,
    config: IngestionConfig | None = None,
) -> int:
    """Embed supported documents and upsert their chunks into Qdrant."""

    config = config or IngestionConfig()
    documents = discover_documents(input_path)
    root = Path(input_path)
    if root.is_file():
        root = root.parent

    chunks: list[tuple[str, int, str, str]] = []
    for path in documents:
        document_id = _document_id(path, root)
        text = path.read_text(encoding="utf-8")
        source_hash = _source_hash(text)
        chunks.extend(
            (document_id, index, chunk, source_hash)
            for index, chunk in enumerate(
                chunk_text(text, config.chunk_size, config.chunk_overlap)
            )
        )

    if not chunks:
        raise ValueError("The discovered documents contain no non-whitespace text")

    ingestion_run_id = str(uuid.uuid4())
    first_vector = embeddings.embed_documents([chunks[0][2]])
    if len(first_vector) != 1 or not first_vector[0]:
        raise ValueError("Embedding service returned an invalid vector")
    _ensure_collection(client, config.collection_name, len(first_vector[0]))

    for document_id in {chunk[0] for chunk in chunks}:
        _delete_source_chunks(client, config.collection_name, document_id)

    total = _upsert_batch(
        client,
        _EmbeddingWithFirstVector(embeddings, chunks[0][2], first_vector[0]),
        config.collection_name,
        chunks[:1],
        ingestion_run_id,
    )
    for start in range(1, len(chunks), config.batch_size):
        total += _upsert_batch(
            client,
            embeddings,
            config.collection_name,
            chunks[start : start + config.batch_size],
            ingestion_run_id,
        )

    logger.info(
        "Ingested %d chunks from %d documents into '%s'",
        total,
        len(documents),
        config.collection_name,
    )
    return total


class _EmbeddingWithFirstVector:
    """Avoid embedding the first chunk twice while preserving the embed API."""

    def __init__(
        self,
        embeddings: EmbeddingsProtocol,
        first_text: str,
        first_vector: list[float],
    ) -> None:
        self._embeddings = embeddings
        self._first_text = first_text
        self._first_vector = first_vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if texts == [self._first_text]:
            return [self._first_vector]
        return self._embeddings.embed_documents(texts)


def create_clients() -> tuple[QdrantClient, EmbeddingsProtocol]:
    """Create production clients from the documented environment variables."""

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not qdrant_url or not qdrant_api_key or not gemini_api_key:
        raise RuntimeError(
            "QDRANT_URL, QDRANT_API_KEY, and GEMINI_API_KEY must be set"
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return (
        QdrantClient(url=qdrant_url, api_key=qdrant_api_key),
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_api_key,
        ),
    )
