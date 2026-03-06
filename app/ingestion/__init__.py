"""Ingestion package — PDF parsing and text chunking.

Public API::

    from app.ingestion import parse_pdf, chunk_document
    from app.ingestion.models import Document, Chunk, IngestionResult
"""

from app.ingestion.chunking import chunk_document
from app.ingestion.models import (
    Chunk,
    Document,
    DocumentMetadata,
    IngestionResult,
    Page,
    Section,
    SectionType,
)
from app.ingestion.pdf_parser import parse_pdf

__all__ = [
    "parse_pdf",
    "chunk_document",
    "Document",
    "DocumentMetadata",
    "Page",
    "Section",
    "SectionType",
    "Chunk",
    "IngestionResult",
]
