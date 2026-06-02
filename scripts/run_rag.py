#!/usr/bin/env python
"""CLI for running a question through the RAG pipeline"""

from __future__ import annotations

import argparse
from dataclasses import replace

from langchain_core.documents import Document

from stroller_rag_eval.config import get_config
from stroller_rag_eval.rag.manuals import manual_choices, metadata_filter_for_manual
from stroller_rag_eval.rag.pipeline import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question over the RAG store.")
    parser.add_argument(
        "question_parts",
        nargs="*",
        help="Question text. Alternative to --question.",
    )
    parser.add_argument(
        "--question",
        help="Question text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--manual",
        choices=manual_choices(),
        help="Restrict retrieval to a product manual.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = args.question or " ".join(args.question_parts).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")

    config = get_config()
    if args.top_k:
        config = replace(config, top_k=args.top_k)

    response = answer_question(
        question,
        config,
        metadata_filter=metadata_filter_for_manual(args.manual),
    )
    _print_retrieved_chunks(response.retrieved_documents)
    print("\nFinal answer")
    print(response.answer)


def _print_retrieved_chunks(documents: list[Document]) -> None:
    print("Retrieved chunks")
    if not documents:
        print("No chunks retrieved.")
        return

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_path", document.metadata.get("source", "unknown"))
        chunk_index = document.metadata.get("chunk_index", "unknown")
        preview = " ".join(document.page_content.strip().split())
        print(f"\n[{index}] source={source} chunk_index={chunk_index}")
        print(preview[:900])


if __name__ == "__main__":
    main()
