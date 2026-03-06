"""Unit tests for app.ingestion.chunking.

No PDF dependency — all tests work on in-memory Document objects built
from synthetic text.

Test groups:
- ``TestChunkDocumentValidation``  — invalid argument rejection
- ``TestChunkCounts``              — expected number of chunks for various inputs
- ``TestChunkOverlap``             — overlap correctness between consecutive chunks
- ``TestDeterministicIds``         — same input → same IDs every time
- ``TestUniqueIds``                — no two chunks share the same ID
- ``TestChunkMetadata``            — metadata propagation into chunks
- ``TestPageMapping``              — page_num assigned to each chunk
- ``TestEdgeCases``                — empty text, single-char, overlap=0
- ``TestFindBoundary``             — boundary-finder unit tests
"""

from __future__ import annotations

import math

import pytest

from app.ingestion.chunking import _find_boundary, chunk_document
from app.ingestion.models import Chunk, Document, DocumentMetadata, Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(text: str, n_pages: int = 1, source: str = "/fake/doc.pdf") -> Document:
    """Build a minimal Document from a raw text string.

    Splits *text* evenly across *n_pages* pages.
    """
    total = len(text)
    page_size = max(1, math.ceil(total / n_pages))
    pages: list[Page] = []

    for i in range(n_pages):
        chunk = text[i * page_size : (i + 1) * page_size]
        pages.append(Page(page_num=i + 1, text=chunk, char_count=len(chunk)))

    full_text = "\n\n".join(p.text for p in pages)

    return Document(
        id=Document.make_id(source),
        title="Test Document",
        source_path=source,
        pages=pages,
        sections=[],
        full_text=full_text,
        metadata=DocumentMetadata(
            source_path=source,
            file_name="doc.pdf",
            file_size_bytes=0,
            total_pages=n_pages,
            total_chars=len(full_text),
        ),
    )


def _make_long_text(n_chars: int = 2000) -> str:
    """Return a synthetic text string of exactly *n_chars* characters."""
    unit = "The quick brown fox jumps over the lazy dog. "
    repeated = (unit * (n_chars // len(unit) + 2))[:n_chars]
    return repeated


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestChunkDocumentValidation:
    """chunk_document raises ValueError for invalid parameter combinations."""

    def test_overlap_equal_to_size_raises(self) -> None:
        doc = _make_document(_make_long_text())
        with pytest.raises(ValueError, match="strictly less than"):
            chunk_document(doc, chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self) -> None:
        doc = _make_document(_make_long_text())
        with pytest.raises(ValueError, match="strictly less than"):
            chunk_document(doc, chunk_size=100, chunk_overlap=150)

    def test_negative_chunk_size_raises(self) -> None:
        doc = _make_document(_make_long_text())
        with pytest.raises(ValueError, match="positive"):
            chunk_document(doc, chunk_size=-1, chunk_overlap=0)

    def test_negative_overlap_raises(self) -> None:
        doc = _make_document(_make_long_text())
        with pytest.raises(ValueError, match="non-negative"):
            chunk_document(doc, chunk_size=100, chunk_overlap=-1)


# ---------------------------------------------------------------------------
# Chunk counts
# ---------------------------------------------------------------------------


class TestChunkCounts:
    """Produced chunk counts match analytical expectations."""

    def test_single_chunk_when_text_shorter_than_chunk_size(self) -> None:
        text = "Short text that fits in one chunk."
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_single_chunk_when_text_equals_chunk_size(self) -> None:
        text = "x" * 512
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_multiple_chunks_for_long_text(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_more_chunks_with_smaller_chunk_size(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks_small = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        chunks_large = chunk_document(doc, chunk_size=400, chunk_overlap=40)
        assert len(chunks_small) > len(chunks_large)

    def test_zero_overlap_produces_fewer_chunks_than_with_overlap(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks_no_overlap = chunk_document(doc, chunk_size=200, chunk_overlap=0)
        chunks_overlap = chunk_document(doc, chunk_size=200, chunk_overlap=50)
        assert len(chunks_no_overlap) <= len(chunks_overlap)

    def test_empty_document_returns_no_chunks(self) -> None:
        doc = _make_document("   \n\n   ")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert chunks == []

    def test_empty_full_text_returns_no_chunks(self) -> None:
        doc = _make_document("")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert chunks == []


# ---------------------------------------------------------------------------
# Overlap correctness
# ---------------------------------------------------------------------------


class TestChunkOverlap:
    """The last N chars of chunk i equal the first N chars of chunk i+1."""

    def _verify_overlap(
        self, chunks: list[Chunk], chunk_size: int, chunk_overlap: int
    ) -> None:
        for i in range(len(chunks) - 1):
            curr = chunks[i]
            nxt = chunks[i + 1]
            # Actual overlap = curr.char_end - nxt.char_start
            actual_overlap = curr.char_end - nxt.char_start
            # Allow ±boundary_window tolerance due to boundary snapping
            assert actual_overlap >= 0, (
                f"Chunk {i} ends at {curr.char_end}, chunk {i+1} starts at {nxt.char_start}: "
                "chunks should not have gaps"
            )
            # Verify text overlap: suffix of chunk i == prefix of chunk i+1
            if actual_overlap > 0:
                overlap_from_curr = curr.text[-actual_overlap:]
                overlap_from_next = nxt.text[:actual_overlap]
                assert overlap_from_curr == overlap_from_next, (
                    f"Text overlap mismatch between chunk {i} and {i+1}"
                )

    def test_overlap_text_matches_between_consecutive_chunks(self) -> None:
        text = _make_long_text(3000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 2
        self._verify_overlap(chunks, 300, 50)

    def test_no_overlap_produces_no_gaps(self) -> None:
        text = _make_long_text(1000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=0)
        for i in range(len(chunks) - 1):
            curr = chunks[i]
            nxt = chunks[i + 1]
            assert curr.char_end >= nxt.char_start


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


class TestDeterministicIds:
    """The same document always produces the same chunk IDs."""

    def test_same_input_same_ids(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks_a = chunk_document(doc, chunk_size=300, chunk_overlap=30)
        chunks_b = chunk_document(doc, chunk_size=300, chunk_overlap=30)
        assert [c.id for c in chunks_a] == [c.id for c in chunks_b]

    def test_different_text_different_ids(self) -> None:
        doc_a = _make_document("A" * 600, source="/fake/doc_a.pdf")
        doc_b = _make_document("B" * 600, source="/fake/doc_b.pdf")
        chunks_a = chunk_document(doc_a, chunk_size=200, chunk_overlap=20)
        chunks_b = chunk_document(doc_b, chunk_size=200, chunk_overlap=20)
        ids_a = {c.id for c in chunks_a}
        ids_b = {c.id for c in chunks_b}
        assert ids_a.isdisjoint(ids_b), "Different texts must not share chunk IDs"

    def test_chunk_make_id_is_16_hex_chars(self) -> None:
        cid = Chunk.make_id("doc123", 0, 100)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)


# ---------------------------------------------------------------------------
# Unique IDs within a document
# ---------------------------------------------------------------------------


class TestUniqueIds:
    """No two chunks within the same document share an ID."""

    def test_all_chunk_ids_unique(self) -> None:
        text = _make_long_text(3000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    """Chunk metadata inherits the correct document-level values."""

    def test_document_id_matches(self) -> None:
        doc = _make_document(_make_long_text(1000))
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.document_id == doc.id

    def test_metadata_contains_source_path(self) -> None:
        source = "/my/special/path/paper.pdf"
        doc = _make_document(_make_long_text(1000), source=source)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source_path"] == source

    def test_metadata_contains_file_name(self) -> None:
        doc = _make_document(_make_long_text(1000))
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "file_name" in chunk.metadata

    def test_metadata_contains_title(self) -> None:
        doc = _make_document(_make_long_text(1000))
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata.get("title") == doc.title

    def test_chunk_index_is_sequential(self) -> None:
        doc = _make_document(_make_long_text(2000))
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for expected_idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected_idx

    def test_char_start_end_are_valid_offsets(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert 0 <= chunk.char_start < chunk.char_end
            assert chunk.char_end <= len(doc.full_text)

    def test_chunk_text_matches_full_text_slice(self) -> None:
        """chunk.text must exactly equal full_text[char_start:char_end]."""
        text = _make_long_text(2000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            expected = doc.full_text[chunk.char_start : chunk.char_end]
            assert chunk.text == expected, (
                f"Chunk {chunk.chunk_index}: text mismatch at [{chunk.char_start}:{chunk.char_end}]"
            )


# ---------------------------------------------------------------------------
# Page number mapping
# ---------------------------------------------------------------------------


class TestPageMapping:
    """Chunks are assigned plausible page numbers."""

    def test_single_page_doc_all_chunks_on_page_1(self) -> None:
        doc = _make_document(_make_long_text(1000), n_pages=1)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.page_num == 1

    def test_multi_page_doc_has_varying_page_nums(self) -> None:
        """With 3 pages, at least two different page numbers should appear."""
        text = _make_long_text(3000)
        doc = _make_document(text, n_pages=3)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        page_nums = {c.page_num for c in chunks if c.page_num is not None}
        assert len(page_nums) >= 2, (
            f"Expected chunks from multiple pages, got: {page_nums}"
        )

    def test_page_num_within_valid_range(self) -> None:
        text = _make_long_text(2000)
        doc = _make_document(text, n_pages=2)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            if chunk.page_num is not None:
                assert 1 <= chunk.page_num <= doc.metadata.total_pages


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_single_character_text(self) -> None:
        doc = _make_document("x")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].text == "x"

    def test_text_with_only_newlines_returns_no_chunks(self) -> None:
        doc = _make_document("\n\n\n\n\n")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert chunks == []

    def test_zero_overlap(self) -> None:
        text = _make_long_text(1000)
        doc = _make_document(text)
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=0)
        assert all(c.char_start >= 0 for c in chunks)

    def test_minimal_fixture(self, minimal_document: Document) -> None:
        """The shared minimal_document fixture produces chunks correctly."""
        chunks = chunk_document(minimal_document, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == minimal_document.id

    def test_chunk_size_larger_than_text(self) -> None:
        doc = _make_document("Hello world.")
        chunks = chunk_document(doc, chunk_size=10_000, chunk_overlap=100)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _find_boundary  (unit-level)
# ---------------------------------------------------------------------------


class TestFindBoundary:
    """The boundary finder returns sensible split points."""

    def test_returns_paragraph_break(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        # target just past the first paragraph break
        idx = _find_boundary(text, 20)
        # The split should be at the paragraph break (char 18 = index after \n\n)
        assert text[idx - 1] == "\n" or text[max(0, idx - 2) : idx] == "\n\n"

    def test_returns_sentence_end(self) -> None:
        text = "One sentence ends here. Another begins."
        idx = _find_boundary(text, 30)
        assert 0 < idx <= 30

    def test_target_at_end_returns_len(self) -> None:
        text = "Short text."
        assert _find_boundary(text, len(text)) == len(text)

    def test_target_beyond_end_returns_len(self) -> None:
        text = "Short text."
        assert _find_boundary(text, 9999) == len(text)

    def test_result_never_exceeds_target(self) -> None:
        text = _make_long_text(500)
        for target in range(10, len(text), 37):
            result = _find_boundary(text, target)
            assert result <= target, f"_find_boundary exceeded target at {target}"
