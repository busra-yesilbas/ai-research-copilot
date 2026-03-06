"""PDF parsing using PyMuPDF (``import fitz``).

The parser is designed to be:

- **Fail-safe**: a per-page ``try/except`` ensures a single corrupt page never
  aborts the whole document; a warning is emitted instead.
- **Deterministic**: the same PDF always produces the same :class:`Document`.
- **Robust**: text normalisation strips form feeds, excessive blank lines, and
  common Unicode typographic characters that break downstream tokenisers.
- **Dependency-light**: the only non-stdlib dependency is ``PyMuPDF``.

Typical usage::

    from app.ingestion.pdf_parser import parse_pdf

    doc = parse_pdf("path/to/paper.pdf")
    print(doc.title, doc.metadata.total_pages)
"""

from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.ingestion.models import Document, DocumentMetadata, Page, Section, SectionType
from app.utils.logger import get_logger

if TYPE_CHECKING:
    pass  # fitz imported lazily to allow graceful ImportError message

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Text inserted between pages in full_text
PAGE_SEPARATOR: str = "\n\n"

# Maximum heading line length (longer lines are treated as body text)
_MAX_HEADING_LEN: int = 80

# Maximum look-around when searching for the end of a heading line
_HEADING_SEARCH_WINDOW: int = 150

# Map of lowercase keyword → SectionType
_SECTION_KEYWORD_MAP: dict[str, SectionType] = {
    "abstract": SectionType.ABSTRACT,
    "introduction": SectionType.INTRODUCTION,
    "related work": SectionType.RELATED_WORK,
    "related works": SectionType.RELATED_WORK,
    "prior work": SectionType.RELATED_WORK,
    "background": SectionType.BACKGROUND,
    "preliminaries": SectionType.BACKGROUND,
    "method": SectionType.METHODS,
    "methods": SectionType.METHODS,
    "methodology": SectionType.METHODS,
    "approach": SectionType.METHODS,
    "proposed method": SectionType.METHODS,
    "model": SectionType.METHODS,
    "architecture": SectionType.METHODS,
    "experiment": SectionType.EXPERIMENTS,
    "experiments": SectionType.EXPERIMENTS,
    "experimental": SectionType.EXPERIMENTS,
    "experimental setup": SectionType.EXPERIMENTS,
    "evaluation": SectionType.EXPERIMENTS,
    "setup": SectionType.EXPERIMENTS,
    "result": SectionType.RESULTS,
    "results": SectionType.RESULTS,
    "findings": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "analysis": SectionType.DISCUSSION,
    "ablation": SectionType.DISCUSSION,
    "conclusion": SectionType.CONCLUSION,
    "conclusions": SectionType.CONCLUSION,
    "concluding remarks": SectionType.CONCLUSION,
    "summary": SectionType.CONCLUSION,
    "references": SectionType.REFERENCES,
    "bibliography": SectionType.REFERENCES,
    "acknowledgment": SectionType.ACKNOWLEDGMENTS,
    "acknowledgments": SectionType.ACKNOWLEDGMENTS,
    "acknowledgement": SectionType.ACKNOWLEDGMENTS,
    "acknowledgements": SectionType.ACKNOWLEDGMENTS,
    "appendix": SectionType.APPENDIX,
    "supplementary": SectionType.APPENDIX,
    "supplementary material": SectionType.APPENDIX,
}

# Numbered-section pattern: "1", "1.", "1.2", "2.3.1", optionally followed by text
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s*(.*)")

# Unicode typographic replacements
_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2013", "-"),   # en-dash
    ("\u2014", "--"),  # em-dash
    ("\u2018", "'"),   # left single quote
    ("\u2019", "'"),   # right single quote
    ("\u201c", '"'),   # left double quote
    ("\u201d", '"'),   # right double quote
    ("\u00a0", " "),   # non-breaking space
    ("\u2022", "*"),   # bullet
    ("\f", "\n"),      # form feed
]

# Collapse 3+ consecutive newlines to exactly 2
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(pdf_path: Path | str) -> Document:
    """Parse a PDF file and return a fully structured :class:`Document`.

    Args:
        pdf_path: Path to a ``.pdf`` file (``str`` or ``pathlib.Path``).

    Returns:
        A :class:`Document` with populated pages, sections, full_text, and
        metadata.

    Raises:
        ImportError:     If ``PyMuPDF`` (``fitz``) is not installed.
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError:      If *pdf_path* is a directory, not a file.
        RuntimeError:    If PyMuPDF fails to open the file (corrupt/encrypted).
    """
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install PyMuPDF"
        ) from exc

    path = Path(pdf_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in {".pdf"}:
        logger.warning("File '%s' does not have a .pdf extension.", path.name)

    logger.info("Parsing PDF: %s (%.1f KB)", path.name, path.stat().st_size / 1024)

    try:
        fitz_doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF could not open '{path}': {exc}") from exc

    with fitz_doc:
        pdf_meta = _extract_pdf_metadata(fitz_doc)
        pages, full_text, page_offsets = _extract_pages(fitz_doc)

    sections = _detect_sections(full_text, page_offsets)
    title = _infer_title(pdf_meta, pages)
    doc_id = Document.make_id(str(path))

    metadata = DocumentMetadata(
        source_path=str(path),
        file_name=path.name,
        file_size_bytes=path.stat().st_size,
        total_pages=len(pages),
        total_chars=len(full_text),
        pdf_metadata=pdf_meta,
    )

    document = Document(
        id=doc_id,
        title=title,
        source_path=str(path),
        pages=pages,
        sections=sections,
        full_text=full_text,
        metadata=metadata,
    )

    logger.info(
        "Parsed '%s': %d pages | %d sections | %d chars",
        path.name,
        len(pages),
        len(sections),
        len(full_text),
    )
    return document


# ---------------------------------------------------------------------------
# Internal helpers — page extraction
# ---------------------------------------------------------------------------


def _extract_pages(
    fitz_doc: Any,
) -> tuple[list[Page], str, list[tuple[int, int]]]:
    """Extract text from every page and build ``full_text``.

    Args:
        fitz_doc: An open ``fitz.Document`` instance.

    Returns:
        A 3-tuple ``(pages, full_text, page_offsets)`` where
        ``page_offsets[i]`` is the ``(start, end)`` character range of
        page ``i+1`` within ``full_text``.
    """
    pages: list[Page] = []
    raw_texts: list[str] = []

    for page_idx in range(len(fitz_doc)):
        try:
            fitz_page = fitz_doc[page_idx]
            raw = fitz_page.get_text("text")
            text = _normalize_text(raw)
        except Exception as exc:
            logger.warning(
                "Could not extract text from page %d: %s", page_idx + 1, exc
            )
            text = ""

        pages.append(
            Page(
                page_num=page_idx + 1,
                text=text,
                char_count=len(text),
                metadata={},
            )
        )
        raw_texts.append(text)

    # Build full_text with PAGE_SEPARATOR between pages and track offsets
    full_text_parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    cursor = 0

    for i, text in enumerate(raw_texts):
        if i > 0:
            full_text_parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        page_offsets.append((start, cursor))

    return pages, "".join(full_text_parts), page_offsets


# ---------------------------------------------------------------------------
# Internal helpers — section detection
# ---------------------------------------------------------------------------


def _detect_sections(
    full_text: str,
    page_offsets: list[tuple[int, int]],
) -> list[Section]:
    """Detect logical sections in *full_text* using heading heuristics.

    The algorithm is a single linear pass that maintains a current heading and
    accumulates body lines until the next heading is found.  Performance is
    O(n) in the number of characters.

    Args:
        full_text:    Concatenated document text.
        page_offsets: List of ``(start, end)`` character ranges per page
                      (as returned by :func:`_extract_pages`).

    Returns:
        Ordered list of :class:`Section` objects.  May be empty if no
        recognisable headings are found.
    """
    if not full_text.strip():
        return []

    # Pre-compute page-end offsets for bisect-based page lookup
    page_ends = [end for _, end in page_offsets]

    def _char_to_page(offset: int) -> int:
        """Map a character offset to its 1-indexed page number."""
        if not page_ends:
            return 1
        idx = bisect_right(page_ends, offset)
        return min(idx + 1, len(page_ends))

    sections: list[Section] = []

    # Parser state
    current_heading: str | None = None
    current_body_lines: list[str] = []
    current_section_start: int = 0
    cursor: int = 0

    def _flush() -> None:
        """Finalise the current in-progress section and append to *sections*."""
        content = "\n".join(current_body_lines).strip()
        if not content:
            return
        heading = current_heading or "Preamble"
        title_clean = _strip_leading_numbers(heading)
        sections.append(
            Section(
                title=title_clean,
                section_type=_classify_section(heading),
                content=content,
                page_start=_char_to_page(current_section_start),
                page_end=_char_to_page(cursor),
                char_count=len(content),
            )
        )

    for line in full_text.split("\n"):
        if _is_heading(line):
            _flush()
            current_heading = line
            current_body_lines = []
            current_section_start = cursor + len(line) + 1  # skip heading + '\n'
        else:
            current_body_lines.append(line)

        cursor += len(line) + 1  # +1 for the '\n' separator

    _flush()
    return sections


def _is_heading(line: str) -> bool:
    """Return ``True`` if *line* looks like a section heading.

    Heuristics applied (in order):

    1. Numbered prefix (``"1."`` / ``"2.3"`` / ``"1.2.3"``) followed by a
       title-cased word that appears in the known section keyword map.
    2. ALL-CAPS line whose lowercase matches a keyword.
    3. Clean (no leading number) line that exactly matches a keyword.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LEN:
        return False

    lower = stripped.lower()
    clean = _strip_leading_numbers(lower)

    # 1. Keyword match (with or without leading numbers)
    for keyword in _SECTION_KEYWORD_MAP:
        if clean == keyword or clean.startswith(keyword + " ") or clean.startswith(keyword + ":"):
            return True

    # 2. ALL-CAPS line matching a keyword
    if stripped.isupper() and 3 <= len(stripped) <= 50:
        for keyword in _SECTION_KEYWORD_MAP:
            if lower == keyword or lower.startswith(keyword + " "):
                return True

    return False


def _classify_section(heading: str) -> SectionType:
    """Map a heading string to the best-matching :class:`SectionType`."""
    lower = heading.strip().lower()
    clean = _strip_leading_numbers(lower)

    # Longest-prefix match to avoid "method" matching "methods" as METHODS
    # when "methods" itself is in the map.
    best_match: SectionType = SectionType.OTHER
    best_len: int = 0
    for keyword, section_type in _SECTION_KEYWORD_MAP.items():
        if (clean == keyword or clean.startswith(keyword + " ") or clean.startswith(keyword + ":")):
            if len(keyword) > best_len:
                best_match = section_type
                best_len = len(keyword)

    return best_match


def _strip_leading_numbers(text: str) -> str:
    """Remove a leading numbering prefix (``"1."`` / ``"2.3 "`` etc.)."""
    m = _NUMBERED_RE.match(text.strip())
    if m:
        return m.group(2).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Internal helpers — text normalisation
# ---------------------------------------------------------------------------


def _normalize_text(raw: str) -> str:
    """Clean raw text extracted from a PDF page.

    Steps:
    1. Replace common Unicode typographic characters with ASCII equivalents.
    2. Collapse runs of 3+ newlines to exactly 2.
    3. Strip trailing whitespace from every line.
    4. Strip leading / trailing whitespace from the whole string.

    Args:
        raw: Raw string from ``fitz_page.get_text("text")``.

    Returns:
        Cleaned string ready for chunking and section detection.
    """
    if not raw:
        return ""

    text = raw
    for src, dst in _UNICODE_REPLACEMENTS:
        text = text.replace(src, dst)

    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# Internal helpers — metadata & title
# ---------------------------------------------------------------------------


def _extract_pdf_metadata(fitz_doc: Any) -> dict[str, Any]:
    """Extract non-empty metadata entries from a ``fitz.Document``.

    Args:
        fitz_doc: An open ``fitz.Document`` instance.

    Returns:
        Dict containing only keys with non-empty string values.
    """
    raw: dict[str, Any] = fitz_doc.metadata or {}
    return {
        k: v.strip()
        for k, v in raw.items()
        if isinstance(v, str) and v.strip()
    }


def _infer_title(pdf_meta: dict[str, Any], pages: list[Page]) -> str:
    """Infer a document title from PDF metadata or first-page text.

    Priority:
    1. ``pdf_meta["title"]`` if present and non-trivial (> 4 chars).
    2. First non-empty line on the first page that is ≥ 5 characters.
    3. Fallback: ``"Untitled Document"``.

    Args:
        pdf_meta: Cleaned PDF metadata dict.
        pages:    Parsed pages (at least one expected).

    Returns:
        Best-guess title string, truncated to 200 characters.
    """
    meta_title = pdf_meta.get("title", "")
    if meta_title and len(meta_title) > 4:
        return meta_title[:200]

    if pages:
        for line in pages[0].text.split("\n"):
            stripped = line.strip()
            if len(stripped) >= 5:
                return stripped[:200]

    return "Untitled Document"
