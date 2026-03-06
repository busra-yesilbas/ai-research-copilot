"""RAG (Retrieval-Augmented Generation) package.

Public API::

    from app.rag import Retriever, RAGPipeline, RAGResult, SourceInfo
    from app.rag import BaseAnswerGenerator, LocalAnswerGenerator
"""

from app.rag.rag_pipeline import (
    BaseAnswerGenerator,
    LocalAnswerGenerator,
    RAGPipeline,
    RAGResult,
    SourceInfo,
)
from app.rag.retriever import Retriever

__all__ = [
    "Retriever",
    "RAGPipeline",
    "RAGResult",
    "SourceInfo",
    "BaseAnswerGenerator",
    "LocalAnswerGenerator",
]
