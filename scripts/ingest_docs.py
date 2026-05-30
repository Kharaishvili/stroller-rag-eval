#!/usr/bin/env python
"""CLI for ingesting markdown documents into Chroma"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from stroller_rag_eval.config import get_config
from stroller_rag_eval.rag.pipeline import ingest_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest markdown docs into Chroma.")
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        help="Markdown source directory. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection-name",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append to the existing Chroma collection instead of rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.source_dir:
        config = replace(
            config,
            source_dirs=tuple(path.expanduser().resolve() for path in args.source_dir),
        )
    if args.persist_dir:
        config = replace(config, chroma_persist_dir=args.persist_dir.expanduser().resolve())
    if args.collection_name:
        config = replace(config, collection_name=args.collection_name)

    result = ingest_documents(config, reset=not args.no_reset)
    print("Ingestion complete")
    print(f"Documents loaded: {result.document_count}")
    print(f"Chunks indexed: {result.chunk_count}")
    print(f"Collection: {result.collection_name}")
    print(f"Persist directory: {result.persist_dir}")


if __name__ == "__main__":
    main()
