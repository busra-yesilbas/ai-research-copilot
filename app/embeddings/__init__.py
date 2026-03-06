"""Embeddings package.

Public API::

    from app.embeddings import EmbeddingModel, FakeEmbeddingModel, BaseEmbeddingModel
"""

from app.embeddings.embedding_model import (
    BaseEmbeddingModel,
    EmbeddingModel,
    FakeEmbeddingModel,
)

__all__ = ["BaseEmbeddingModel", "EmbeddingModel", "FakeEmbeddingModel"]
