"""Command-line entry point for populating the Qdrant knowledge base."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from src.ingestion import IngestionConfig, create_clients, ingest_directory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest .txt and .md files into the ResearchPilot Qdrant collection."
    )
    parser.add_argument("input_path", help="A document file or directory to ingest")
    parser.add_argument(
        "--collection",
        default="documents",
        help="Qdrant collection name (default: documents)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding/upsert batch size (default: 64)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    try:
        client, embeddings = create_clients()
        count = ingest_directory(
            args.input_path,
            client,
            embeddings,
            IngestionConfig(
                collection_name=args.collection,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Ingested {count} chunks into Qdrant collection '{args.collection}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
