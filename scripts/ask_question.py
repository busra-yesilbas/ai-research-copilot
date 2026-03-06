#!/usr/bin/env python3
"""CLI: Ask a question against a persisted vector index.

Usage::

    # Bash
    python scripts/ask_question.py --index data/index --query "What is the main contribution?"
    python scripts/ask_question.py --index data/index --query "What datasets were used?" --top-k 3

    # PowerShell
    python scripts\\ask_question.py --index data\\index --query "What is the main contribution?"

Prerequisites:
    - Run ``python scripts/build_index.py`` first to build the index.
    - sentence-transformers must be installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.embeddings.embedding_model import EmbeddingModel  # noqa: E402
from app.rag.rag_pipeline import LocalAnswerGenerator, RAGPipeline  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.utils.logger import configure_logging, get_logger  # noqa: E402
from app.vector_store.faiss_store import load_vector_store  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask_question",
        description="Ask a question against a persisted vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", required=True, type=Path, help="Path to the vector index directory.")
    parser.add_argument("--query", required=True, type=str, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name (default: from settings).")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu, cuda, …).")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved source chunks.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(level=args.log_level)
    log = get_logger(__name__)
    settings = get_settings()

    index_path = args.index.resolve()
    if not (index_path / "metadata.json").exists():
        print(
            f"ERROR: No index found at '{index_path}'.\n"
            "Run 'python scripts/build_index.py' first.",
            file=sys.stderr,
        )
        return 1

    model_name = args.model or settings.embedding_model_name
    device = args.device or settings.embedding_device

    try:
        embedding_model = EmbeddingModel(
            model_name=model_name,
            device=device,
            cache_dir=str(settings.models_dir_resolved),
        )
        store = load_vector_store(str(index_path), embedding_model)
    except ImportError as exc:
        print(f"ERROR: Missing dependency — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Failed to load index — {exc}", file=sys.stderr)
        return 1

    retriever = Retriever(embedding_model, store)
    pipeline = RAGPipeline(retriever, LocalAnswerGenerator())

    try:
        result = pipeline.query(args.query, top_k=args.top_k)
    except Exception as exc:
        print(f"ERROR: Query failed — {exc}", file=sys.stderr)
        return 1

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Query: {result.query}")
    print(sep)
    print(f"\n{result.answer}\n")

    if args.show_sources:
        print(f"{sep}")
        print(f"  Sources ({len(result.sources)} chunks):")
        print(sep)
        for i, src in enumerate(result.sources, 1):
            page = src.metadata.get("page_num", "?")
            fname = src.metadata.get("file_name", "?")
            print(f"  [{i}] score={src.score:.4f}  file={fname}  page={page}")
            print(f"      {src.text[:150]}…")
            print()

    print(f"  Latency: {result.latency_ms:.1f} ms")
    print(sep)
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
