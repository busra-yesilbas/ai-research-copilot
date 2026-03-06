#!/usr/bin/env python3
"""CLI: Ingest a PDF into the AI Research Copilot system.

Parses a PDF file, chunks the extracted text, and optionally saves the
result as JSON so it can later be embedded and indexed.

Usage examples::

    # Bash / macOS / Linux
    python scripts/ingest_pdf.py --pdf data/raw/paper.pdf
    python scripts/ingest_pdf.py --pdf paper.pdf --chunk-size 256 --chunk-overlap 32
    python scripts/ingest_pdf.py --pdf paper.pdf --save-json --output-dir data/processed

    # PowerShell / Windows
    python scripts\\ingest_pdf.py --pdf data\\raw\\paper.pdf
    python scripts\\ingest_pdf.py --pdf paper.pdf --save-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of where the script is invoked.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.ingestion.chunking import chunk_document  # noqa: E402
from app.ingestion.models import IngestionResult  # noqa: E402
from app.ingestion.pdf_parser import parse_pdf  # noqa: E402
from app.utils.logger import configure_logging, get_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ingest_pdf",
        description="Parse a PDF and chunk its text for the AI Research Copilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/ingest_pdf.py --pdf paper.pdf --save-json"
        ),
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Maximum characters per chunk (overrides CHUNK_SIZE env var).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Overlapping characters between chunks (overrides CHUNK_OVERLAP env var).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write the processed JSON output. "
            "Defaults to DATA_DIR/processed when --save-json is set."
        ),
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Persist the ingestion result as <stem>.json in --output-dir.",
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
# Core logic
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the ingestion pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: ``0`` on success, ``1`` on any error.
    """
    configure_logging(level=args.log_level)
    log = get_logger(__name__)
    settings = get_settings()

    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap

    # Validate chunk parameters early for a clear user-facing error
    if chunk_overlap >= chunk_size:
        log.error(
            "chunk_overlap (%d) must be less than chunk_size (%d).",
            chunk_overlap,
            chunk_size,
        )
        return 1

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        log.error("File not found: %s", pdf_path)
        return 1
    if not pdf_path.is_file():
        log.error("Path is not a file: %s", pdf_path)
        return 1

    # ── Parse ────────────────────────────────────────────────────────────────
    log.info("Parsing PDF: %s", pdf_path.name)
    try:
        document = parse_pdf(pdf_path)
    except ImportError as exc:
        log.error("Dependency error: %s", exc)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("Parse error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected parse error: %s", exc, exc_info=True)
        return 1

    # ── Chunk ────────────────────────────────────────────────────────────────
    log.info(
        "Chunking document (chunk_size=%d, chunk_overlap=%d)…",
        chunk_size,
        chunk_overlap,
    )
    try:
        chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ValueError as exc:
        log.error("Chunking parameter error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected chunking error: %s", exc, exc_info=True)
        return 1

    result = IngestionResult(document=document, chunks=chunks)

    # ── Print summary ────────────────────────────────────────────────────────
    _print_summary(result)

    # ── Save JSON (optional) ─────────────────────────────────────────────────
    if args.save_json or args.output_dir:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else settings.processed_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{pdf_path.stem}.json"

        payload = {
            "document": result.document.model_dump(mode="json"),
            "chunks": [c.model_dump(mode="json") for c in result.chunks],
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        log.info("Saved ingestion result → %s", out_path)
        print(f"\n  Output saved: {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_summary(result: IngestionResult) -> None:
    """Print a formatted ingestion summary to stdout."""
    sep = "─" * 54
    print(f"\n{sep}")
    print("  Ingestion Summary")
    print(sep)
    for line in result.summary().splitlines():
        print(f"  {line}")
    if result.chunks:
        avg = sum(c.char_count for c in result.chunks) / len(result.chunks)
        min_c = min(c.char_count for c in result.chunks)
        max_c = max(c.char_count for c in result.chunks)
        print(f"  Chunk chars : avg={avg:.0f}  min={min_c}  max={max_c}")
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
