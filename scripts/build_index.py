#!/usr/bin/env python3
"""CLI: Parse a PDF, embed its chunks, and persist a vector index.

This script wires together the M2 ingestion pipeline (PDF parser + chunker)
with the M3 embedding and vector store layers.

Usage examples::

    # Bash / macOS / Linux
    python scripts/build_index.py --pdf data/raw/paper.pdf --out data/index
    python scripts/build_index.py --pdf paper.pdf --out data/index \\
        --chunk-size 512 --chunk-overlap 64 --model all-MiniLM-L6-v2

    # PowerShell / Windows
    python scripts\\build_index.py --pdf data\\raw\\paper.pdf --out data\\index
    python scripts\\build_index.py --pdf paper.pdf --out data\\index `
        --chunk-size 512 --chunk-overlap 64

The script prints a summary like::

    ──────────────────────────────────────────────────────
      Build Index Summary
    ──────────────────────────────────────────────────────
      File       : attention_is_all_you_need.pdf
      Chunks     : 42
      Vector dim : 384
      Backend    : sklearn
      Index path : data/index
    ──────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of invocation path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.embeddings.embedding_model import EmbeddingModel  # noqa: E402
from app.ingestion.chunking import chunk_document  # noqa: E402
from app.ingestion.pdf_parser import parse_pdf  # noqa: E402
from app.utils.logger import configure_logging, get_logger  # noqa: E402
from app.vector_store.faiss_store import get_vector_store  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="build_index",
        description="Parse a PDF, embed its chunks, and save a vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python scripts/build_index.py --pdf paper.pdf --out data/index",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for the persisted vector index.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Sentence-transformers model name or path "
            "(default: from EMBEDDING_MODEL_NAME env var or all-MiniLM-L6-v2)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Torch device string: 'cpu', 'cuda', 'mps', … (default: from settings).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Max characters per chunk (default: from CHUNK_SIZE env var).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Overlapping chars between chunks (default: from CHUNK_OVERLAP env var).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Embedding batch size (default: from EMBEDDING_BATCH_SIZE env var).",
    )
    parser.add_argument(
        "--prefer-faiss",
        action="store_true",
        default=False,
        help="Prefer FAISS backend (requires faiss-cpu to be installed).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        metavar="LEVEL",
        help="Logging verbosity.",
    )
    return parser


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the build-index pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code — ``0`` on success, ``1`` on any error.
    """
    configure_logging(level=args.log_level)
    log = get_logger(__name__)
    settings = get_settings()

    # Resolve parameters with settings as fallback
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap
    model_name = args.model or settings.embedding_model_name
    device = args.device or settings.embedding_device
    batch_size = args.batch_size or settings.embedding_batch_size

    pdf_path = args.pdf.resolve()
    out_path = args.out.resolve()

    # ── Validate input ────────────────────────────────────────────────────────
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        return 1
    if not pdf_path.is_file():
        log.error("Path is not a file: %s", pdf_path)
        return 1

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────────
    log.info("Step 1/4 — Parsing PDF: %s", pdf_path.name)
    try:
        document = parse_pdf(pdf_path)
    except ImportError as exc:
        log.error("Missing dependency: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("PDF parse failed: %s", exc, exc_info=args.log_level == "DEBUG")
        return 1

    log.info(
        "  Parsed %d pages, %d chars",
        document.metadata.total_pages,
        document.metadata.total_chars,
    )

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    log.info(
        "Step 2/4 — Chunking (size=%d, overlap=%d)…", chunk_size, chunk_overlap
    )
    try:
        chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError as exc:
        log.error("Chunking error: %s", exc)
        return 1

    if not chunks:
        log.error("No chunks produced — the PDF may be image-only or empty.")
        return 1

    log.info("  Produced %d chunks", len(chunks))

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    log.info("Step 3/4 — Embedding with model '%s' on device '%s'…", model_name, device)
    try:
        embedding_model = EmbeddingModel(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            cache_dir=str(settings.models_dir_resolved),
        )
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "source_path": c.metadata.get("source_path", ""),
                "file_name": c.metadata.get("file_name", ""),
                "title": c.metadata.get("title", ""),
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]
        embeddings = embedding_model.embed_texts(texts)
    except ImportError as exc:
        log.error("Missing dependency: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Embedding failed: %s", exc, exc_info=args.log_level == "DEBUG")
        return 1

    vec_dim = len(embeddings[0]) if embeddings else 0
    log.info("  Embedded %d texts, dim=%d", len(embeddings), vec_dim)

    # ── Step 4: Build and persist index ──────────────────────────────────────
    log.info("Step 4/4 — Building and saving vector index to '%s'…", out_path)
    try:
        store = get_vector_store(embedding_model, prefer_faiss=args.prefer_faiss)
        store.add_embeddings(embeddings, texts, metadatas)
        store.save(str(out_path))
    except Exception as exc:  # noqa: BLE001
        log.error("Index build failed: %s", exc, exc_info=args.log_level == "DEBUG")
        return 1

    backend = type(store).__name__.replace("VectorStore", "").lower()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(
        file_name=pdf_path.name,
        n_chunks=len(chunks),
        vec_dim=vec_dim,
        backend=backend,
        index_path=str(out_path),
    )
    return 0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_summary(
    file_name: str,
    n_chunks: int,
    vec_dim: int,
    backend: str,
    index_path: str,
) -> None:
    """Print a formatted build-index summary to stdout."""
    sep = "─" * 56
    print(f"\n{sep}")
    print("  Build Index Summary")
    print(sep)
    print(f"  File       : {file_name}")
    print(f"  Chunks     : {n_chunks}")
    print(f"  Vector dim : {vec_dim}")
    print(f"  Backend    : {backend}")
    print(f"  Index path : {index_path}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
