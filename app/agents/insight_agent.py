"""Insight agent: extract key contributions, limitations, and research gaps.

The agent makes four specialised RAG queries and returns a structured insight
report covering:

- **Contributions**: the novel claims and results of the paper.
- **Methodology**: the technical approach.
- **Limitations**: weaknesses, constraints, and caveats.
- **Research gaps**: open questions and suggested future work.

Typical usage::

    from app.agents.insight_agent import InsightAgent

    agent = InsightAgent(rag_pipeline)
    response = agent.run()
    print(response.contributions)
    print(response.limitations)
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


class InsightResponse(BaseModel):
    """Structured insight report extracted from indexed research papers.

    Attributes:
        contributions:  Novel claims, results, and contributions.
        methodology:    Technical approach and proposed methods.
        limitations:    Weaknesses, constraints, and caveats.
        research_gaps:  Open questions and suggested future work.
        sources:        Supporting retrieved chunks.
        agent:          Agent identifier.
        latency_ms:     Wall-clock time in ms.
    """

    contributions: str = ""
    methodology: str = ""
    limitations: str = ""
    research_gaps: str = ""
    sources: list[SourceInfo] = Field(default_factory=list)
    agent: str = "insight"
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


_QUERIES: dict[str, str] = {
    "contributions": (
        "What are the main contributions, novelty, and key claims of this research? "
        "What does this paper propose or achieve that is new?"
    ),
    "methodology": (
        "What methods, models, architectures, and algorithms does this paper use "
        "or propose? Describe the technical approach."
    ),
    "limitations": (
        "What are the limitations, weaknesses, constraints, or failure cases "
        "of the proposed approach? What does the paper not address?"
    ),
    "research_gaps": (
        "What future work, open problems, and research gaps are identified? "
        "What directions does the paper suggest for follow-up research?"
    ),
}


class InsightAgent(BaseAgent):
    """Extracts structured insights from indexed research papers.

    Issues four specialised RAG queries (contributions, methodology,
    limitations, research gaps) and returns a structured report.

    Args:
        rag_pipeline: RAG pipeline for all internal queries.
        top_k:        Chunks to retrieve per sub-query (default: 4).
    """

    def __init__(self, rag_pipeline: "RAGPipeline", top_k: int = 4) -> None:
        super().__init__(rag_pipeline)
        self._top_k = top_k

    def run(self, **kwargs: Any) -> InsightResponse:  # type: ignore[override]
        """Extract insights from indexed papers.

        Args:
            **kwargs: Ignored (reserved for future topic filtering).

        Returns:
            :class:`InsightResponse` with contributions, limitations, gaps.
        """
        t0 = time.perf_counter()
        all_sources: list[SourceInfo] = []
        results: dict[str, str] = {}

        for key, query in _QUERIES.items():
            rag_result = self._rag.query(query, top_k=self._top_k)
            results[key] = rag_result.answer
            all_sources.extend(rag_result.sources)

        # Deduplicate sources
        seen: set[str] = set()
        unique_sources: list[SourceInfo] = []
        for src in all_sources:
            if src.chunk_id not in seen:
                unique_sources.append(src)
                seen.add(src.chunk_id)

        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        logger.info(
            "InsightAgent: completed in %.1f ms, %d unique sources",
            latency_ms,
            len(unique_sources),
        )
        return InsightResponse(
            contributions=results.get("contributions", ""),
            methodology=results.get("methodology", ""),
            limitations=results.get("limitations", ""),
            research_gaps=results.get("research_gaps", ""),
            sources=unique_sources[:12],
            latency_ms=latency_ms,
        )
