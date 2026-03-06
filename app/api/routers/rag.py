"""RAG API router: /upload-paper and /ask endpoints.

Endpoints
---------
``POST /api/v1/upload-paper``
    Accept a PDF, run the full ingestion pipeline (parse → chunk → embed),
    and persist the updated vector index.

``POST /api/v1/ask``
    Accept a JSON query, retrieve relevant chunks, and return a synthesised
    answer plus source citations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.api.deps import get_rag_pipeline, get_vector_store, get_embedding_model
from app.config.settings import get_settings
from app.rag.rag_pipeline import RAGPipeline, SourceInfo
from app.utils.logger import get_logger
from app.vector_store.faiss_store import VectorStore
from app.embeddings.embedding_model import BaseEmbeddingModel

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["RAG"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for ``POST /ask``."""

    query: str = Field(..., min_length=1, description="The question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")


class QueryResponse(BaseModel):
    """Response body for ``POST /ask``."""

    query: str
    answer: str
    sources: list[SourceInfo]
    latency_ms: float


class UploadResponse(BaseModel):
    """Response body for ``POST /upload-paper``."""

    document_id: str
    file_name: str
    total_pages: int
    total_chars: int
    num_chunks: int
    vector_dim: int
    index_size: int
    message: str


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Retrieve relevant chunks from the indexed papers and return "
        "a synthesised extractive answer with source citations."
    ),
)
async def ask(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Answer *body.query* using the RAG pipeline.

    Returns ``503`` if the pipeline is not ready (no indexed papers or missing
    sentence-transformers).
    """
    try:
        result = pipeline.query(body.query, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        logger.error("RAG query failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed.",
        )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        latency_ms=result.latency_ms,
    )


@router.post(
    "/upload-paper",
    response_model=UploadResponse,
    summary="Upload and index a research paper",
    description=(
        "Parse the uploaded PDF, split it into chunks, embed them, and "
        "add them to the vector index.  Accepts optional chunking overrides."
    ),
)
async def upload_paper(
    file: UploadFile = File(..., description="PDF file to ingest."),
    chunk_size: int | None = Form(default=None, ge=64),
    chunk_overlap: int | None = Form(default=None, ge=0),
    embedding_model: BaseEmbeddingModel = Depends(get_embedding_model),
    store: VectorStore = Depends(get_vector_store),
) -> UploadResponse:
    """Parse, chunk, embed, and index *file*.

    After ingestion the vector store is persisted to ``settings.index_dir``.
    """
    settings = get_settings()

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are supported.",
            )

    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    # ── Save upload to a temp file ─────────────────────────────────────────
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # ── Parse ─────────────────────────────────────────────────────────
        try:
            from app.ingestion.pdf_parser import parse_pdf
            document = parse_pdf(tmp_path)
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"PDF parsing not available: {exc}",
            )
        except Exception as exc:
            logger.error("PDF parse failed for '%s': %s", file.filename, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not parse PDF: {exc}",
            )

        # ── Chunk ──────────────────────────────────────────────────────────
        from app.ingestion.chunking import chunk_document

        chunks = chunk_document(document, chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the PDF (image-only?).",
            )

        # ── Embed + add to store ───────────────────────────────────────────
        texts = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "source_path": c.metadata.get("source_path", ""),
                "file_name": c.metadata.get("file_name", file.filename or ""),
                "title": c.metadata.get("title", ""),
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]

        embeddings = embedding_model.embed_texts(texts)
        store.add_embeddings(embeddings, texts, metadatas)

        # ── Persist index ──────────────────────────────────────────────────
        index_path = str(settings.index_dir_resolved)
        try:
            store.save(index_path)
            logger.info(
                "Index saved to '%s' (%d vectors total)", index_path, store.count()
            )
        except Exception as exc:
            logger.warning("Could not persist index: %s", exc)

        vec_dim = len(embeddings[0]) if embeddings else 0

        logger.info(
            "Indexed '%s': %d pages, %d chunks, dim=%d",
            file.filename,
            document.metadata.total_pages,
            len(chunks),
            vec_dim,
        )
        return UploadResponse(
            document_id=document.id,
            file_name=file.filename or "upload.pdf",
            total_pages=document.metadata.total_pages,
            total_chars=document.metadata.total_chars,
            num_chunks=len(chunks),
            vector_dim=vec_dim,
            index_size=store.count(),
            message=f"Successfully indexed {len(chunks)} chunks from '{file.filename}'.",
        )

    finally:
        # Always remove temp file
        tmp_path.unlink(missing_ok=True)
