#!/usr/bin/env python3
"""CLI entry point for the RAG pipeline.

Usage:
    python main.py index                  Index all documents in data/raw/
    python main.py query "Your question"  Query the indexed documents
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import PROJECT_ROOT, load_config
from src.pipeline import RAGPipeline


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_index(args: argparse.Namespace) -> None:
    """Run the indexing pipeline."""
    config = load_config()
    pipeline = RAGPipeline(config=config)
    data_dir = PROJECT_ROOT / "data" / "raw"
    count = pipeline.index(data_dir)
    print(f"\n✅ Indexed {count} chunk(s) from {data_dir}")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a query against the indexed documents."""
    config = load_config()
    pipeline = RAGPipeline(config=config)
    result = pipeline.query(args.question)

    print(f"\n📝 Answer:\n{result.answer}\n")
    if result.sources:
        print("📚 Sources:")
        for src in result.sources:
            source_file = src.get("source_file", "unknown")
            chunk_idx = src.get("chunk_index", "?")
            page = src.get("page_number", "")
            location = f"  - {source_file}, chunk {chunk_idx}"
            if page:
                location += f", page {page}"
            print(location)


def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    subparsers.add_parser("index", help="Index documents from data/raw/")

    # query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("question", type=str, help="The question to ask")

    args = parser.parse_args()
    setup_logging(debug=args.debug)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
