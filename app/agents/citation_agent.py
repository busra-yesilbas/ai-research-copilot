"""Citation agent: find related chunks and surface paper-level citations.

The agent retrieves the most semantically relevant chunks for a given query
and groups them by source document so the caller can present them as
citations or related-work references.

Typical usage::

    from app.agents.citation_agent import CitationAgent

    agent = CitationAgent(rag_pipeline)
    response = agent.run(query="self-supervised learning for NLP")
    for citation in response.citations:
        print(citation["title"], citation["relevance"])
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.rag.rag_pipeline import SourceInfo
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CitationInfo(BaseModel):
    """Metadata for a single cited document.

    Attributes:
        document_id: Stable document identifier.
        file_name:   Original PDF filename.
        title:       Inferred document title (if available).
        page_nums:   Pages that contributed relevant chunks.
        relevance:   Maximum cosine similarity score among the document's chunks.
        chunk_count: Number of matching chunks from this document.
    """

    document_id: str
    file_name: str = ""
    title: str = ""
    page_nums: list[int] = Field(default_factory=list)
    relevance: float = 0.0
    chunk_count: int = 0


class CitationResponse(BaseModel):
    """Response from the citation agent.

    Attributes:
        query:      The search query that was used.
        citations:  Document-level citation list (best match first).
        chunks:     Raw chunk-level results.
        agent:      Agent identifier.
        latency_ms: Wall-clock time in ms.
    """

    query: str
    citations: list[CitationInfo] = Field(default_factory=list)
    chunks: list[SourceInfo] = Field(default_factory=list)
    agent: str = "citation"
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CitationAgent(BaseAgent):
    """Finds related papers/chunks for a given topic or query.

    Retrieves the top-*top_k* chunks and aggregates them by source document
    to produce document-level citation records.

    Args:
        rag_pipeline: RAG pipeline for retrieval.
        top_k:        Number of chunks to retrieve (default: 10).
    """

    def __init__(self, rag_pipeline: "RAGPipeline", top_k: int = 10) -> None:
        super().__init__(rag_pipeline)
        self._top_k = top_k

    def run(self, query: str = "", **kwargs: Any) -> CitationResponse:  # type: ignore[override]
        """Find related chunks and aggregate by source document.

        Args:
            query: Search query describing the topic of interest.
            **kwargs: Ignored.

        Returns:
            :class:`CitationResponse` with document-level citations and raw chunks.

        Raises:
            ValueError: If *query* is empty.
        """
        query = (query or kwargs.get("topic", "")).strip()
        if not query:
            query = "main topic and contributions of this research paper"

        t0 = time.perf_counter()
        rag_result = self._rag.query(query, top_k=self._top_k)
        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        # ── Aggregate by document ──────────────────────────────────────────
        doc_map: dict[str, dict[str, Any]] = {}
        for src in rag_result.sources:
            doc_id = src.metadata.get("document_id") or src.chunk_id[:16]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "file_name": src.metadata.get("file_name", ""),
                    "title": src.metadata.get("title", ""),
                    "page_nums": [],
                    "relevance": src.score,
                    "chunk_count": 0,
                }
            entry = doc_map[doc_id]
            entry["chunk_count"] += 1
            entry["relevance"] = max(entry["relevance"], src.score)
            page = src.metadata.get("page_num")
            if page and page not in entry["page_nums"]:
                entry["page_nums"].append(page)

        citations = [
            CitationInfo(**v)
            for v in sorted(
                doc_map.values(), key=lambda x: x["relevance"], reverse=True
            )
        ]

        logger.info(
            "CitationAgent: query='%s...', %d chunks -> %d documents, %.1f ms",
            query[:50],
            len(rag_result.sources),
            len(citations),
            latency_ms,
        )
        return CitationResponse(
            query=query,
            citations=citations,
            chunks=rag_result.sources,
            latency_ms=latency_ms,
        )
