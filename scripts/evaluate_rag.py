#!/usr/bin/env python3
"""CLI: Evaluate retrieval quality of a persisted vector index.

Generates synthetic queries from indexed chunks, runs retrieval, and
reports Recall@k and MRR.

Usage::

    # Bash
    python scripts/evaluate_rag.py --index data/index --k 5 --n-queries 20

    # PowerShell
    python scripts\\evaluate_rag.py --index data\\index --k 5 --n-queries 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.embeddings.embedding_model import EmbeddingModel  # noqa: E402
from app.evaluation.rag_eval import RagEvaluator  # noqa: E402
from app.ingestion.models import Chunk  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402
from app.utils.logger import configure_logging, get_logger  # noqa: E402
from app.vector_store.faiss_store import load_vector_store  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_rag",
        description="Evaluate retrieval quality (Recall@k, MRR) for a vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", required=True, type=Path, help="Vector index directory.")
    parser.add_argument("--k", type=int, default=5, help="Retrieval depth for Recall@k.")
    parser.add_argument("--n-queries", type=int, default=20, help="Number of synthetic queries.")
    parser.add_argument("--model", type=str, default=None, help="Embedding model (default: from settings).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-report", type=Path, default=None, help="Save JSON report to this path.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING")
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(level=args.log_level)
    log = get_logger(__name__)
    settings = get_settings()

    index_path = args.index.resolve()
    if not (index_path / "metadata.json").exists():
        print(f"ERROR: No index at '{index_path}'. Run build_index.py first.", file=sys.stderr)
        return 1

    model_name = args.model or settings.embedding_model_name
    device = args.device or settings.embedding_device

    try:
        embedding_model = EmbeddingModel(model_name=model_name, device=device,
                                         cache_dir=str(settings.models_dir_resolved))
        store = load_vector_store(str(index_path), embedding_model)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if store.count() == 0:
        print("ERROR: Index is empty. Upload papers first.", file=sys.stderr)
        return 1

    # Reconstruct Chunk-like objects from the store metadata
    entries = getattr(store, "_entries", [])

    class _MinimalChunk:
        def __init__(self, chunk_id: str, text: str) -> None:
            self.id = chunk_id
            self.text = text
            self.document_id = ""

    chunks = [_MinimalChunk(e["chunk_id"], e.get("text", "")) for e in entries]

    retriever = Retriever(embedding_model, store)
    evaluator = RagEvaluator(retriever)
    report = evaluator.evaluate(chunks, n_queries=args.n_queries, k=args.k)

    print(report.summary())

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w") as fh:
            json.dump(report.model_dump(), fh, indent=2)
        print(f"\nReport saved to '{args.save_report}'")

    return 0


def main() -> None:
    parser = _build_parser()
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
