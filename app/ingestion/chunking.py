"""Deterministic, boundary-aware character chunker.

The chunker splits a :class:`~app.ingestion.models.Document` into overlapping
text windows called :class:`~app.ingestion.models.Chunk` objects.  Each chunk
carries:

- A **deterministic SHA-256-based ID** — identical input always yields
  identical IDs, making re-ingestion safe and idempotent.
- **Character offsets** (``char_start``, ``char_end``) so any chunk can be
  traced back to an exact position in the source document.
- A **page number** derived from the document's page-offset map so retrieval
  results can cite "page 3 of paper.pdf".
- **Inherited metadata** (source path, title, total pages) so each chunk is
  fully self-contained for display and evaluation.

Splitting strategy
------------------
1. Advance a sliding window of ``chunk_size`` characters.
2. Before committing the split point, search backwards up to
   ``_BOUNDARY_SEARCH_WINDOW`` characters for the nearest natural boundary
   (paragraph break → sentence end → line break → word break).  This avoids
   cutting in the middle of words or sentences.
3. The next chunk starts at ``split_point - chunk_overlap``, creating the
   requested overlap.

Typical usage::

    from app.ingestion.chunking import chunk_document

    chunks = chunk_document(document, chunk_size=512, chunk_overlap=64)
"""

from __future__ import annotations

import bisect
from typing import Any

from app.ingestion.models import Chunk, Document
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How far back to search for a natural split boundary
_BOUNDARY_SEARCH_WINDOW: int = 120

# Sentence-ending punctuation characters
_SENTENCE_ENDS: frozenset[str] = frozenset(".!?")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split *document* into overlapping text chunks.

    Args:
        document:      Parsed :class:`Document` to chunk.
        chunk_size:    Maximum number of characters per chunk.  Must be
                       strictly greater than *chunk_overlap*.
        chunk_overlap: Number of characters shared between the end of chunk N
                       and the start of chunk N+1.

    Returns:
        Ordered :class:`list` of :class:`Chunk` objects with deterministic IDs.
        Returns an empty list if ``document.full_text`` contains no non-whitespace
        characters.

    Raises:
        ValueError: If ``chunk_overlap >= chunk_size`` or either value is ≤ 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than "
            f"chunk_size ({chunk_size})."
        )

    text = document.full_text
    if not text.strip():
        logger.warning(
            "Document '%s' has no extractable text; returning zero chunks.",
            document.id,
        )
        return []

    page_ends, page_nums = _build_page_index(document)
    base_metadata = _build_base_metadata(document)

    chunks: list[Chunk] = []
    start: int = 0
    chunk_idx: int = 0

    while start < len(text):
        # Determine raw end of this window
        raw_end = min(start + chunk_size, len(text))

        # If we haven't hit the end of the text, try to snap to a boundary
        end = _find_boundary(text, raw_end) if raw_end < len(text) else raw_end

        chunk_text = text[start:end]

        # Skip chunks that are purely whitespace
        if not chunk_text.strip():
            start = end
            continue

        # Assign page number based on the chunk's midpoint character
        mid = (start + end) // 2
        page_num = _page_for_offset(mid, page_ends, page_nums)

        chunks.append(
            Chunk(
                id=Chunk.make_id(document.id, start, end),
                document_id=document.id,
                text=chunk_text,
                char_start=start,
                char_end=end,
                chunk_index=chunk_idx,
                page_num=page_num,
                metadata={**base_metadata, "chunk_index": chunk_idx},
            )
        )

        chunk_idx += 1

        # If we just consumed the tail of the document, we're done.
        if end >= len(text):
            break

        # Next chunk begins chunk_overlap chars before the current end.
        next_start = end - chunk_overlap
        if next_start <= start:
            # Safety: always advance by at least 1 to prevent infinite loop.
            next_start = start + 1
        start = next_start

    logger.debug(
        "Chunked document '%s': %d chunks | text=%d chars | size=%d | overlap=%d",
        document.id,
        len(chunks),
        len(text),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Boundary finder
# ---------------------------------------------------------------------------


def _find_boundary(text: str, target: int) -> int:
    """Return the best split index at or before *target*.

    Searches backwards within a window of :data:`_BOUNDARY_SEARCH_WINDOW`
    characters for the nearest natural boundary, in priority order:

    1. Paragraph break (``"\\n\\n"``).
    2. Sentence end (``'.'`` / ``'!'`` / ``'?'`` followed by whitespace).
    3. Line break (``"\\n"``).
    4. Word boundary (``" "``).
    5. Hard cut at *target* if none of the above are found.

    Args:
        text:   Full document text.
        target: Preferred (maximum) split index.

    Returns:
        Adjusted split index ≤ *target*.
    """
    if target >= len(text):
        return len(text)

    window_start = max(0, target - _BOUNDARY_SEARCH_WINDOW)
    window = text[window_start:target]

    # 1. Paragraph break
    idx = window.rfind("\n\n")
    if idx != -1:
        return window_start + idx + 2  # include both newlines

    # 2. Sentence boundary
    for i in range(len(window) - 1, -1, -1):
        ch = window[i]
        if ch in _SENTENCE_ENDS:
            # The character after the punctuation must be whitespace or end of window
            next_ch = window[i + 1] if i + 1 < len(window) else " "
            if next_ch in " \n\t":
                return window_start + i + 1

    # 3. Line break
    idx = window.rfind("\n")
    if idx != -1:
        return window_start + idx + 1

    # 4. Word boundary
    idx = window.rfind(" ")
    if idx != -1:
        return window_start + idx + 1

    # 5. Hard cut
    return target


# ---------------------------------------------------------------------------
# Page mapping helpers
# ---------------------------------------------------------------------------


def _build_page_index(document: Document) -> tuple[list[int], list[int]]:
    """Build sorted lists of page-end offsets and page numbers for bisect lookup.

    Args:
        document: Parsed document whose ``pages`` field is populated.

    Returns:
        ``(page_ends, page_nums)`` where ``page_ends[i]`` is the exclusive end
        character offset of page ``page_nums[i]`` within ``document.full_text``.
    """
    from app.ingestion.pdf_parser import PAGE_SEPARATOR

    page_ends: list[int] = []
    page_nums: list[int] = []
    cursor = 0

    for i, page in enumerate(document.pages):
        cursor += len(page.text)
        page_ends.append(cursor)
        page_nums.append(page.page_num)
        if i < len(document.pages) - 1:
            cursor += len(PAGE_SEPARATOR)

    return page_ends, page_nums


def _page_for_offset(
    offset: int,
    page_ends: list[int],
    page_nums: list[int],
) -> int | None:
    """Return the 1-indexed page number for *offset*.

    Args:
        offset:    Character offset within ``full_text``.
        page_ends: Sorted list of page-end offsets (from :func:`_build_page_index`).
        page_nums: Corresponding 1-indexed page numbers.

    Returns:
        Page number, or ``None`` if *page_ends* is empty.
    """
    if not page_ends:
        return None
    idx = bisect.bisect_right(page_ends, offset)
    return page_nums[min(idx, len(page_nums) - 1)]


def _build_base_metadata(document: Document) -> dict[str, Any]:
    """Build the metadata dict inherited by every chunk of *document*.

    Args:
        document: Source document.

    Returns:
        Dict with ``source_path``, ``file_name``, ``title``, ``document_id``,
        and ``total_pages``.
    """
    return {
        "document_id": document.id,
        "source_path": document.source_path,
        "file_name": document.metadata.file_name,
        "title": document.title,
        "total_pages": document.metadata.total_pages,
    }
