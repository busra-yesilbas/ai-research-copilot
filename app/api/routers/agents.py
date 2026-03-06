"""Agents API router: /summarize, /research-insights, /related-work.

Endpoints
---------
``POST /api/v1/summarize``
    Summarise the indexed research papers (or a specific topic).

``POST /api/v1/research-insights``
    Extract contributions, methodology, limitations, and research gaps.

``POST /api/v1/related-work``
    Find related chunks and surface document-level citations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.citation_agent import CitationInfo, CitationResponse
from app.agents.insight_agent import InsightResponse
from app.agents.summarizer_agent import SummaryResponse
from app.api.deps import get_rag_pipeline
from app.rag.rag_pipeline import RAGPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Agents"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SummarizeRequest(BaseModel):
    """Request body for ``POST /summarize``."""

    topic: str = Field(default="", description="Optional focus topic or query.")
    top_k: int = Field(default=3, ge=1, le=10)


class InsightsRequest(BaseModel):
    """Request body for ``POST /research-insights``."""

    top_k: int = Field(default=4, ge=1, le=10)


class RelatedWorkRequest(BaseModel):
    """Request body for ``POST /related-work``."""

    query: str = Field(..., min_length=1, description="Topic or query to find related work for.")
    top_k: int = Field(default=10, ge=1, le=20)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Summarise indexed research papers",
)
async def summarize(
    body: SummarizeRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> SummaryResponse:
    """Generate an extractive summary of indexed papers.

    Optionally narrow the summary to a specific *topic*.
    Returns ``503`` if no papers have been indexed yet.
    """
    from app.agents.summarizer_agent import SummarizerAgent

    agent = SummarizerAgent(pipeline, top_k=body.top_k)
    try:
        return agent.run(topic=body.topic)
    except Exception as exc:
        logger.error("SummarizerAgent failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summarisation failed.",
        )


@router.post(
    "/research-insights",
    response_model=InsightResponse,
    summary="Extract research insights",
)
async def research_insights(
    body: InsightsRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> InsightResponse:
    """Extract contributions, methodology, limitations, and research gaps.

    Returns ``503`` if no papers have been indexed yet.
    """
    from app.agents.insight_agent import InsightAgent

    agent = InsightAgent(pipeline, top_k=body.top_k)
    try:
        return agent.run()
    except Exception as exc:
        logger.error("InsightAgent failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Insight extraction failed.",
        )


@router.post(
    "/related-work",
    response_model=CitationResponse,
    summary="Find related work / citations",
)
async def related_work(
    body: RelatedWorkRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> CitationResponse:
    """Find chunks and papers related to a given topic or query.

    Returns document-level citations sorted by relevance.
    Returns ``503`` if no papers have been indexed yet.
    """
    from app.agents.citation_agent import CitationAgent

    agent = CitationAgent(pipeline, top_k=body.top_k)
    try:
        return agent.run(query=body.query)
    except Exception as exc:
        logger.error("CitationAgent failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Related-work search failed.",
        )
