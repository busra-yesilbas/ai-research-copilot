"""Base class for all research agents.

Agents are high-level tools that wrap the RAG pipeline and produce
structured, purpose-specific outputs (summaries, insights, citations, etc.).
Each agent is responsible for crafting the right queries and post-processing
the retrieved content into a useful format.

All agents are *read-only* with respect to the vector store — they only query,
never modify.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.rag.rag_pipeline import SourceInfo
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.rag.rag_pipeline import RAGPipeline

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared response model
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Generic agent response used when a single text result is sufficient.

    Attributes:
        result:     The main output text.
        sources:    Supporting chunks retrieved by the RAG pipeline.
        agent:      Name of the agent that produced this response.
        latency_ms: Wall-clock time in milliseconds.
        metadata:   Any extra agent-specific data.
    """

    result: str
    sources: list[SourceInfo] = Field(default_factory=list)
    agent: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all research paper agents.

    Subclasses must inject a :class:`~app.rag.rag_pipeline.RAGPipeline`
    at construction time and implement :meth:`run`.

    Args:
        rag_pipeline: Configured pipeline used for all internal queries.
    """

    def __init__(self, rag_pipeline: "RAGPipeline") -> None:
        self._rag = rag_pipeline

    @property
    def name(self) -> str:
        """Human-readable agent name (defaults to class name)."""
        return type(self).__name__

    @abstractmethod
    def run(self, **kwargs: Any) -> BaseModel:
        """Execute the agent's primary task.

        Args:
            **kwargs: Agent-specific keyword arguments.

        Returns:
            A Pydantic model containing the structured result.
        """
