"""Retriever: embed a query and fetch the top-k matching chunks.

The :class:`Retriever` is a thin coordination layer between the embedding
model and the vector store.  It has one responsibility: turn a free-text query
into an ordered list of :class:`~app.vector_store.faiss_store.SearchResult`
objects.

Typical usage::

    from app.rag.retriever import Retriever

    retriever = Retriever(embedding_model, vector_store)
    results = retriever.retrieve("What datasets were used?", k=5)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from app.utils.logger import get_logger
from app.vector_store.faiss_store import SearchResult

if TYPE_CHECKING:
    from app.embeddings.embedding_model import BaseEmbeddingModel
    from app.vector_store.faiss_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """Embeds queries and searches the vector store.

    Args:
        embedding_model: Model used to encode the query string.
        vector_store:    Indexed chunks to search against.
    """

    def __init__(
        self,
        embedding_model: "BaseEmbeddingModel",
        vector_store: "VectorStore",
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def store(self) -> "VectorStore":
        """Direct access to the underlying vector store."""
        return self._vector_store

    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        """Direct access to the embedding model."""
        return self._embedding_model

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """Embed *query* and return the *k* most relevant chunks.

        Args:
            query: Free-text query string.  Must not be empty.
            k:     Maximum number of results.  Clamped to the store size.

        Returns:
            Ordered list of :class:`SearchResult` (highest score first).

        Raises:
            ValueError: If *query* is empty or *k* < 1.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        t0 = time.perf_counter()
        query_embedding = self._embedding_model.embed_query(query)
        results = self._vector_store.search(query_embedding, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        logger.debug(
            "Retrieved %d/%d results for query %r in %.1f ms",
            len(results),
            k,
            query[:60],
            elapsed_ms,
        )
        return results

    def is_ready(self) -> bool:
        """Return ``True`` if the store contains at least one indexed vector."""
        return self._vector_store.count() > 0
