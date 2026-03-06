"""Summarizer agent: produce a structured summary of indexed research papers.

The agent issues three targeted RAG queries covering the paper's purpose,
methodology, and results/conclusions, then merges the retrieved content into
a single cohesive summary.

Typical usage::

    from app.agents.summarizer_agent import SummarizerAgent

    agent = SummarizerAgent(rag_pipeline)
    response = agent.run(topic="attention mechanism")
    print(response.summary)
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


class SummaryResponse(BaseModel):
    """Structured summary of a research paper.

    Attributes:
        summary:    Multi-sentence extractive summary.
        topic:      The topic/query used to guide summarisation.
        sources:    Supporting retrieved chunks.
        agent:      Agent identifier.
        latency_ms: Wall-clock time in ms.
    """

    summary: str
    topic: str = ""
    sources: list[SourceInfo] = Field(default_factory=list)
    agent: str = "summarizer"
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


_DEFAULT_QUERIES: list[str] = [
    "What is this paper about? Describe the main problem and objective.",
    "What methods, models, and techniques are proposed in this research?",
    "What are the key results, findings, and conclusions of this paper?",
]


class SummarizerAgent(BaseAgent):
    """Summarises the content of indexed research papers.

    Issues multiple targeted RAG queries to cover different aspects of the
    paper (problem statement, methodology, results), then assembles a
    unified summary from the top retrieved sentences.

    Args:
        rag_pipeline: RAG pipeline used for all internal queries.
        top_k:        Chunks to retrieve per sub-query (default: 3).
    """

    def __init__(self, rag_pipeline: "RAGPipeline", top_k: int = 3) -> None:
        super().__init__(rag_pipeline)
        self._top_k = top_k

    def run(self, topic: str = "", **kwargs: Any) -> SummaryResponse:  # type: ignore[override]
        """Generate a summary.

        Args:
            topic: Optional focus topic.  When provided a single targeted
                   query is used instead of the default three-query set.
            **kwargs: Ignored.

        Returns:
            :class:`SummaryResponse` with the assembled summary and sources.
        """
        t0 = time.perf_counter()

        queries = [topic] if topic.strip() else _DEFAULT_QUERIES
        all_sources: list[SourceInfo] = []
        answer_parts: list[str] = []

        for q in queries:
            result = self._rag.query(q, top_k=self._top_k)
            answer_parts.append(result.answer)
            all_sources.extend(result.sources)

        # Deduplicate sources (preserve insertion order)
        seen_ids: set[str] = set()
        unique_sources: list[SourceInfo] = []
        for src in all_sources:
            if src.chunk_id not in seen_ids:
                unique_sources.append(src)
                seen_ids.add(src.chunk_id)

        summary = " ".join(p for p in answer_parts if p)
        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        logger.info(
            "SummarizerAgent: generated summary (%d chars) in %.1f ms",
            len(summary),
            latency_ms,
        )
        return SummaryResponse(
            summary=summary,
            topic=topic,
            sources=unique_sources[:10],
            latency_ms=latency_ms,
        )
