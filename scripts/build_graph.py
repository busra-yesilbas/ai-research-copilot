#!/usr/bin/env python3
"""CLI: Build a knowledge graph from a persisted vector index.

Extracts entities (Model, Dataset, Task, Metric) from indexed chunks and
saves the co-occurrence graph to JSON (or Neo4j if configured).

Usage::

    # Bash
    python scripts/build_graph.py --index data/index
    python scripts/build_graph.py --index data/index --out data/graph/graph.json

    # PowerShell
    python scripts\\build_graph.py --index data\\index
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
from app.knowledge_graph.graph_builder import GraphBuilder  # noqa: E402
from app.utils.logger import configure_logging, get_logger  # noqa: E402
from app.vector_store.faiss_store import load_vector_store  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_graph",
        description="Build a knowledge graph from an indexed vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", required=True, type=Path, help="Vector index directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for graph.json (default: data/graph).")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(level=args.log_level)
    log = get_logger(__name__)
    settings = get_settings()

    index_path = args.index.resolve()
    if not (index_path / "metadata.json").exists():
        print(f"ERROR: No index at '{index_path}'.", file=sys.stderr)
        return 1

    model_name = args.model or settings.embedding_model_name
    device = args.device or settings.embedding_device

    try:
        embedding_model = EmbeddingModel(model_name=model_name, device=device,
                                         cache_dir=str(settings.models_dir_resolved))
        store = load_vector_store(str(index_path), embedding_model)
    except Exception as exc:
        print(f"ERROR loading index: {exc}", file=sys.stderr)
        return 1

    entries = getattr(store, "_entries", [])

    class _Chunk:
        def __init__(self, text: str, doc_id: str) -> None:
            self.text = text
            self.document_id = doc_id

    chunks = [_Chunk(e.get("text", ""), e.get("metadata", {}).get("document_id", "")) for e in entries]

    graph_dir = str(args.out) if args.out else None
    builder = GraphBuilder(graph_dir=graph_dir)
    graph = builder.build(chunks)
    builder.save(graph)

    sep = "─" * 50
    print(f"\n{sep}")
    print("  Knowledge Graph Summary")
    print(sep)
    print(f"  {graph.summary()}")
    print(sep)
    return 0


def main() -> None:
    sys.exit(run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
