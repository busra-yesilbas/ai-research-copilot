"""Shared pytest fixtures for the AI Research Copilot test suite.

Fixtures defined here are automatically available to every test file under
``tests/`` without explicit imports.

The ``sample_pdf`` fixture creates a 3-page synthetic PDF using PyMuPDF so
that PDF-related tests never depend on external files.  If PyMuPDF is not
installed the fixture calls ``pytest.importorskip``, which causes any test
that requests it to be **skipped** rather than failing with an ImportError.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic PDF fixture
# ---------------------------------------------------------------------------

# Content we embed in the synthetic PDF.  Defined at module level so tests
# can import and assert against these constants directly.

PDF_PAGE_1_TEXT = (
    "Deep Learning for Natural Language Processing\n\n"
    "ABSTRACT\n\n"
    "This paper presents novel approaches to natural language processing "
    "using deep learning techniques. We demonstrate significant improvements "
    "over existing baseline methods across multiple benchmark datasets."
)

PDF_PAGE_2_TEXT = (
    "1. Introduction\n\n"
    "Natural language processing has seen remarkable progress in recent years. "
    "Transformer-based architectures have fundamentally changed how we approach "
    "text understanding tasks.\n\n"
    "In this work we propose a new methodology combining attention mechanisms "
    "with graph neural networks to achieve state-of-the-art performance."
)

PDF_PAGE_3_TEXT = (
    "2. Methods\n\n"
    "Our approach consists of three components: a pre-trained language model, "
    "a graph construction module, and a fusion layer. "
    "Each component is described in detail in the following subsections.\n\n"
    "References\n\n"
    "Vaswani et al. (2017) Attention is all you need. NeurIPS."
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return the path to a minimal 3-page synthetic PDF.

    The PDF contains recognisable section headings (Abstract, Introduction,
    Methods, References) so section-detection tests can make meaningful
    assertions.

    Scope is ``session`` so the file is created only once per test run.

    Yields:
        ``pathlib.Path`` pointing to the generated ``.pdf`` file.

    Skips:
        Any test using this fixture if ``PyMuPDF`` (``fitz``) is not installed.
    """
    fitz = pytest.importorskip("fitz", reason="PyMuPDF is required for PDF tests.")

    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "sample_paper.pdf"

    doc = fitz.open()  # new empty document

    for page_text in [PDF_PAGE_1_TEXT, PDF_PAGE_2_TEXT, PDF_PAGE_3_TEXT]:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text(
            (50, 72),   # top-left insert point
            page_text,
            fontname="helv",
            fontsize=11,
        )

    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


# ---------------------------------------------------------------------------
# Minimal in-memory Document fixture (no PDF dependency)
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_document() -> "Document":  # type: ignore[name-defined]  # noqa: F821
    """Return a minimal :class:`Document` built entirely from in-memory data.

    Safe to use in chunker tests that do not need PyMuPDF.
    """
    from app.ingestion.models import Document, DocumentMetadata, Page

    text_p1 = "The quick brown fox jumps over the lazy dog. " * 20
    text_p2 = "Pack my box with five dozen liquor jugs. " * 20

    full_text = text_p1 + "\n\n" + text_p2
    source = "/fake/path/test_doc.pdf"

    return Document(
        id=Document.make_id(source),
        title="Minimal Test Document",
        source_path=source,
        pages=[
            Page(page_num=1, text=text_p1, char_count=len(text_p1)),
            Page(page_num=2, text=text_p2, char_count=len(text_p2)),
        ],
        sections=[],
        full_text=full_text,
        metadata=DocumentMetadata(
            source_path=source,
            file_name="test_doc.pdf",
            file_size_bytes=0,
            total_pages=2,
            total_chars=len(full_text),
        ),
    )
