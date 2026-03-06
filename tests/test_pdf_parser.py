"""Unit tests for app.ingestion.pdf_parser.

Tests are grouped by concern:
- ``TestParseErrors``   — error handling (missing file, not-a-file, etc.)
- ``TestDocumentStructure`` — Document object shape and field correctness
- ``TestPageExtraction``    — per-page text and metadata
- ``TestSectionDetection``  — heading heuristics (_is_heading, _classify_section)
- ``TestNormalizeText``     — text normalisation edge-cases
- ``TestInferTitle``        — title inference priority

The ``sample_pdf`` fixture (defined in conftest.py) creates a real 3-page PDF
using PyMuPDF; tests in ``TestDocumentStructure`` / ``TestPageExtraction``
request it and are automatically skipped if fitz is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.pdf_parser import (
    _classify_section,
    _infer_title,
    _is_heading,
    _normalize_text,
    _strip_leading_numbers,
    parse_pdf,
)
from app.ingestion.models import Document, SectionType


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParseErrors:
    """parse_pdf raises meaningful exceptions for bad inputs."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.pdf"
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_pdf(missing)

    def test_directory_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            parse_pdf(tmp_path)

    def test_missing_fitz_raises_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate fitz being absent and verify the ImportError message."""
        import builtins
        real_import = builtins.__import__

        def _block_fitz(name: str, *args: object, **kwargs: object) -> object:
            if name == "fitz":
                raise ModuleNotFoundError("No module named 'fitz'")
            return real_import(name, *args, **kwargs)

        dummy_pdf = tmp_path / "dummy.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4")  # minimal stub

        monkeypatch.setattr(builtins, "__import__", _block_fitz)
        with pytest.raises(ImportError, match="PyMuPDF"):
            parse_pdf(dummy_pdf)


# ---------------------------------------------------------------------------
# Document structure (requires sample_pdf fixture → PyMuPDF)
# ---------------------------------------------------------------------------


class TestDocumentStructure:
    """The returned Document has all required fields correctly populated."""

    def test_document_id_is_16_hex_chars(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert len(doc.id) == 16
        assert all(c in "0123456789abcdef" for c in doc.id)

    def test_document_id_is_deterministic(self, sample_pdf: Path) -> None:
        doc1 = parse_pdf(sample_pdf)
        doc2 = parse_pdf(sample_pdf)
        assert doc1.id == doc2.id

    def test_source_path_is_absolute(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert Path(doc.source_path).is_absolute()

    def test_title_is_non_empty_string(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert isinstance(doc.title, str)
        assert len(doc.title.strip()) > 0

    def test_full_text_is_non_empty(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert len(doc.full_text.strip()) > 0

    def test_full_text_char_count_matches_metadata(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert doc.metadata.total_chars == len(doc.full_text)

    def test_metadata_file_name_matches(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert doc.metadata.file_name == sample_pdf.name

    def test_metadata_file_size_positive(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert doc.metadata.file_size_bytes > 0

    def test_parsed_at_is_set(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert doc.metadata.parsed_at is not None


# ---------------------------------------------------------------------------
# Page extraction (requires PyMuPDF)
# ---------------------------------------------------------------------------


class TestPageExtraction:
    """Pages are correctly extracted with the right structure."""

    def test_page_count_is_three(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert doc.metadata.total_pages == 3
        assert len(doc.pages) == 3

    def test_pages_are_1_indexed(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        page_nums = [p.page_num for p in doc.pages]
        assert page_nums == [1, 2, 3]

    def test_all_pages_have_text(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        for page in doc.pages:
            assert len(page.text.strip()) > 0, f"Page {page.page_num} is empty"

    def test_char_count_matches_text_length(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        for page in doc.pages:
            assert page.char_count == len(page.text)

    def test_full_text_contains_all_page_texts(self, sample_pdf: Path) -> None:
        """Every page's text should appear somewhere in full_text."""
        doc = parse_pdf(sample_pdf)
        for page in doc.pages:
            # Use the first 30 non-whitespace chars as a probe
            probe = page.text.strip()[:30]
            if probe:
                assert probe in doc.full_text


# ---------------------------------------------------------------------------
# Section detection (requires PyMuPDF)
# ---------------------------------------------------------------------------


class TestSectionDetection:
    """Detected sections have correct types and non-empty content."""

    def test_sections_is_a_list(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        assert isinstance(doc.sections, list)

    def test_section_content_is_non_empty(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        for section in doc.sections:
            assert len(section.content.strip()) > 0, (
                f"Section '{section.title}' has empty content"
            )

    def test_section_char_count_matches_content(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        for section in doc.sections:
            assert section.char_count == len(section.content)

    def test_section_pages_are_valid(self, sample_pdf: Path) -> None:
        doc = parse_pdf(sample_pdf)
        for section in doc.sections:
            assert section.page_start >= 1
            assert section.page_end >= section.page_start


# ---------------------------------------------------------------------------
# _is_heading  (unit-level, no PDF needed)
# ---------------------------------------------------------------------------


class TestIsHeading:
    """_is_heading correctly classifies lines."""

    @pytest.mark.parametrize(
        "line",
        [
            "ABSTRACT",
            "Abstract",
            "abstract",
            "1. Introduction",
            "2. Methods",
            "3. Experiments",
            "References",
            "REFERENCES",
            "Conclusion",
            "2.1 Related Work",
            "Discussion",
        ],
    )
    def test_known_headings_are_detected(self, line: str) -> None:
        assert _is_heading(line), f"Expected '{line}' to be a heading"

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "In this section we show that the proposed method outperforms",
            "Table 1: Comparison of results on the benchmark dataset",
            "This is a very long line that goes on and on without stopping at all and should never be a heading.",
            "a",
        ],
    )
    def test_non_headings_are_rejected(self, line: str) -> None:
        assert not _is_heading(line), f"Expected '{line}' NOT to be a heading"


# ---------------------------------------------------------------------------
# _classify_section  (unit-level)
# ---------------------------------------------------------------------------


class TestClassifySection:
    """_classify_section maps headings to the expected SectionType."""

    @pytest.mark.parametrize(
        "heading, expected",
        [
            ("Abstract", SectionType.ABSTRACT),
            ("ABSTRACT", SectionType.ABSTRACT),
            ("1. Introduction", SectionType.INTRODUCTION),
            ("2. Methods", SectionType.METHODS),
            ("3. Experiments", SectionType.EXPERIMENTS),
            ("4. Results", SectionType.RESULTS),
            ("5. Discussion", SectionType.DISCUSSION),
            ("6. Conclusion", SectionType.CONCLUSION),
            ("References", SectionType.REFERENCES),
            ("Acknowledgments", SectionType.ACKNOWLEDGMENTS),
            ("RandomUnknownSection", SectionType.OTHER),
        ],
    )
    def test_classification(self, heading: str, expected: SectionType) -> None:
        assert _classify_section(heading) == expected


# ---------------------------------------------------------------------------
# _normalize_text  (unit-level)
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """_normalize_text cleans up raw extracted text."""

    def test_empty_string_returns_empty(self) -> None:
        assert _normalize_text("") == ""

    def test_form_feed_replaced_with_newline(self) -> None:
        result = _normalize_text("page1\fpage2")
        assert "\f" not in result
        assert "page1" in result and "page2" in result

    def test_en_dash_replaced(self) -> None:
        result = _normalize_text("state\u2013of\u2013the\u2013art")
        assert "\u2013" not in result
        assert "state-of-the-art" in result

    def test_triple_newlines_collapsed(self) -> None:
        result = _normalize_text("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_trailing_whitespace_stripped_per_line(self) -> None:
        result = _normalize_text("hello   \nworld   ")
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_leading_trailing_stripped(self) -> None:
        result = _normalize_text("  \n  hello  \n  ")
        assert result == result.strip()


# ---------------------------------------------------------------------------
# _strip_leading_numbers  (unit-level)
# ---------------------------------------------------------------------------


class TestStripLeadingNumbers:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1. Introduction", "Introduction"),
            ("2.3 Methods", "Methods"),
            ("1.2.3 Background", "Background"),
            ("Abstract", "Abstract"),
            ("REFERENCES", "REFERENCES"),
        ],
    )
    def test_strips_correctly(self, raw: str, expected: str) -> None:
        assert _strip_leading_numbers(raw) == expected


# ---------------------------------------------------------------------------
# _infer_title  (unit-level)
# ---------------------------------------------------------------------------


class TestInferTitle:
    """_infer_title respects the PDF metadata → first-line → fallback priority."""

    def test_uses_pdf_metadata_title(self) -> None:
        from app.ingestion.models import Page

        pages = [Page(page_num=1, text="Some first line", char_count=15)]
        title = _infer_title({"title": "My Great Paper"}, pages)
        assert title == "My Great Paper"

    def test_ignores_short_metadata_title(self) -> None:
        from app.ingestion.models import Page

        pages = [Page(page_num=1, text="First Line Of Document", char_count=22)]
        title = _infer_title({"title": "ok"}, pages)  # len 2, < threshold 4
        assert "First Line" in title

    def test_falls_back_to_first_page_line(self) -> None:
        from app.ingestion.models import Page

        pages = [Page(page_num=1, text="Deep Neural Networks\nAbstract", char_count=29)]
        title = _infer_title({}, pages)
        assert "Deep Neural Networks" in title

    def test_ultimate_fallback_when_no_pages(self) -> None:
        title = _infer_title({}, [])
        assert title == "Untitled Document"

    def test_title_truncated_to_200_chars(self) -> None:
        long_title = "A" * 300
        from app.ingestion.models import Page

        pages = [Page(page_num=1, text=long_title, char_count=len(long_title))]
        title = _infer_title({"title": long_title}, pages)
        assert len(title) <= 200
