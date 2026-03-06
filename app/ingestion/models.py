"""Pydantic data models for the ingestion pipeline.

These are the core domain objects that flow through the entire system:
PDF → Document → Chunk → (embeddings, vector store, RAG).

All IDs are deterministic (SHA-256-based) so the same input always produces
the same output — critical for reproducibility and idempotent re-ingestion.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SectionType(str, Enum):
    """Canonical section types found in academic research papers."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHODS = "methods"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    APPENDIX = "appendix"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class Page(BaseModel):
    """Represents a single parsed page of a PDF document.

    Attributes:
        page_num:   1-indexed page number.
        text:       Normalised extracted text content.
        char_count: Length of *text* (cached for quick access).
        metadata:   Arbitrary per-page metadata (e.g. rotation, dimensions).
    """

    page_num: int = Field(..., ge=1, description="1-indexed page number.")
    text: str = Field(..., description="Extracted and normalised page text.")
    char_count: int = Field(..., ge=0, description="Character count of the page text.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-page metadata (dimensions, rotation, …).",
    )


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------


class Section(BaseModel):
    """A logical section detected within a document via heading heuristics.

    Attributes:
        title:        Heading text with leading numbering stripped.
        section_type: Classified :class:`SectionType`.
        content:      Full text body of the section (heading excluded).
        page_start:   1-indexed page where the section heading appears.
        page_end:     1-indexed page where the section content ends.
        char_count:   Character count of *content*.
    """

    title: str = Field(..., description="Section heading (numbers stripped).")
    section_type: SectionType = Field(
        default=SectionType.OTHER,
        description="Classified section category.",
    )
    content: str = Field(..., description="Body text of the section.")
    page_start: int = Field(..., ge=1, description="1-indexed start page.")
    page_end: int = Field(..., ge=1, description="1-indexed end page.")
    char_count: int = Field(..., ge=0, description="Character count of content.")


# ---------------------------------------------------------------------------
# DocumentMetadata
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """File-level and PDF-level metadata for an ingested document.

    Attributes:
        source_path:    Absolute path of the source PDF on disk.
        file_name:      Basename (e.g. ``"paper.pdf"``).
        file_size_bytes: File size in bytes.
        total_pages:    Total number of pages in the PDF.
        total_chars:    Total extracted characters across all pages.
        parsed_at:      UTC timestamp of when the document was parsed.
        pdf_metadata:   Raw metadata dict from PyMuPDF (title, author, …).
    """

    source_path: str = Field(..., description="Absolute path of the source PDF.")
    file_name: str = Field(..., description="PDF file basename.")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes.")
    total_pages: int = Field(..., ge=0, description="Total number of PDF pages.")
    total_chars: int = Field(..., ge=0, description="Total extracted characters.")
    parsed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of parsing.",
    )
    pdf_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw PyMuPDF metadata dictionary.",
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Complete structured representation of a parsed PDF document.

    This is the primary output of the ingestion pipeline and the input to the
    chunker.  All downstream modules (embeddings, RAG, agents) reference
    documents by their stable ``id``.

    Attributes:
        id:          Deterministic 16-char hex ID derived from ``source_path``.
        title:       Inferred document title.
        source_path: Absolute path of the source PDF.
        pages:       Ordered list of :class:`Page` objects.
        sections:    Detected logical sections (may be empty if heuristics
                     find no clear headings).
        full_text:   Concatenated text of all pages, separated by ``"\\n\\n"``.
        metadata:    :class:`DocumentMetadata` instance.
    """

    id: str = Field(
        ...,
        description="Deterministic 16-char hex document ID (SHA-256 of source_path).",
    )
    title: str = Field(..., description="Inferred document title.")
    source_path: str = Field(..., description="Absolute path of the source PDF.")
    pages: list[Page] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    full_text: str = Field(default="", description="Concatenated text of all pages.")
    metadata: DocumentMetadata = Field(..., description="File and PDF-level metadata.")

    @classmethod
    def make_id(cls, source_path: str) -> str:
        """Return a deterministic 16-char hex ID for *source_path*.

        Args:
            source_path: Absolute path string used as the hash input.

        Returns:
            First 16 hexadecimal characters of SHA-256(source_path).
        """
        return hashlib.sha256(source_path.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A contiguous, overlapping text segment produced by the chunker.

    Chunks are the atomic retrieval unit stored in the vector store.  Every
    field needed for downstream display or evaluation is stored directly on
    the chunk so it is fully self-contained.

    Attributes:
        id:           Deterministic 16-char hex ID.
        document_id:  ID of the parent :class:`Document`.
        text:         Chunk text content.
        char_start:   Inclusive start offset within ``Document.full_text``.
        char_end:     Exclusive end offset within ``Document.full_text``.
        chunk_index:  0-indexed position within the parent document.
        page_num:     1-indexed page where the chunk midpoint falls
                      (``None`` if page mapping is unavailable).
        metadata:     Inherited document metadata plus chunk-specific fields.
    """

    id: str = Field(..., description="Deterministic 16-char hex chunk ID.")
    document_id: str = Field(..., description="Parent document ID.")
    text: str = Field(..., description="Text content of this chunk.")
    char_start: int = Field(..., ge=0, description="Start offset in full_text (inclusive).")
    char_end: int = Field(..., ge=0, description="End offset in full_text (exclusive).")
    chunk_index: int = Field(..., ge=0, description="0-indexed chunk position in document.")
    page_num: int | None = Field(
        default=None,
        description="1-indexed page of the chunk midpoint.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-level metadata (source_path, title, total_pages, …).",
    )

    @classmethod
    def make_id(cls, document_id: str, char_start: int, char_end: int) -> str:
        """Return a deterministic 16-char hex ID for a chunk.

        Args:
            document_id: Parent document identifier.
            char_start:  Inclusive start offset.
            char_end:    Exclusive end offset.

        Returns:
            First 16 hexadecimal characters of SHA-256(document_id:start:end).
        """
        raw = f"{document_id}:{char_start}:{char_end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk's text."""
        return len(self.text)


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


class IngestionResult(BaseModel):
    """Aggregated result of a complete document ingestion run.

    Attributes:
        document: The parsed :class:`Document`.
        chunks:   Ordered list of :class:`Chunk` objects ready for embedding.
    """

    document: Document
    chunks: list[Chunk]

    @property
    def chunk_count(self) -> int:
        """Total number of chunks produced."""
        return len(self.chunks)

    @property
    def total_chars(self) -> int:
        """Total extracted characters from the document."""
        return self.document.metadata.total_chars

    def summary(self) -> str:
        """Return a human-readable ingestion summary string."""
        doc = self.document
        sections_str = (
            ", ".join(s.section_type.value for s in doc.sections[:5])
            if doc.sections
            else "none detected"
        )
        return (
            f"File      : {doc.metadata.file_name}\n"
            f"Doc ID    : {doc.id}\n"
            f"Title     : {doc.title[:80]}\n"
            f"Pages     : {doc.metadata.total_pages}\n"
            f"Chars     : {doc.metadata.total_chars:,}\n"
            f"Sections  : {len(doc.sections)} ({sections_str})\n"
            f"Chunks    : {len(self.chunks)}\n"
        )
