"""FastAPI dependency injection helpers.

Each function extracts a shared resource from ``app.state`` and raises a
``503 Service Unavailable`` if it has not been initialised.  In tests, these
functions can be replaced via ``app.dependency_overrides`` to inject fakes.

Usage in route handlers::

    from fastapi import Depends
    from app.api.deps import get_rag_pipeline

    @router.post("/ask")
    async def ask(
        body: QueryRequest,
        pipeline: RAGPipeline = Depends(get_rag_pipeline),
    ):
        ...
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.embeddings.embedding_model import BaseEmbeddingModel
from app.rag.rag_pipeline import RAGPipeline
from app.vector_store.faiss_store import VectorStore


def get_embedding_model(request: Request) -> BaseEmbeddingModel:
    """Return the application-level embedding model.

    Raises:
        HTTPException 503: If ``sentence-transformers`` is not installed or
                           the model has not been initialised.
    """
    model: BaseEmbeddingModel | None = getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Embedding model is not available. "
                "Ensure sentence-transformers is installed: "
                "pip install sentence-transformers"
            ),
        )
    return model


def get_vector_store(request: Request) -> VectorStore:
    """Return the application-level vector store.

    Raises:
        HTTPException 503: If the store has not been initialised.
    """
    store: VectorStore | None = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not initialised.",
        )
    return store


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Return the application-level RAG pipeline.

    Raises:
        HTTPException 503: If the pipeline has not been initialised (typically
                           because sentence-transformers is missing).
    """
    pipeline: RAGPipeline | None = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG pipeline is not ready. "
                "Upload a PDF first or install sentence-transformers."
            ),
        )
    return pipeline
