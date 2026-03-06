"""Research agents package.

Public API::

    from app.agents import SummarizerAgent, InsightAgent, CitationAgent
    from app.agents import SummaryResponse, InsightResponse, CitationResponse
"""

from app.agents.base_agent import AgentResponse, BaseAgent
from app.agents.citation_agent import CitationAgent, CitationInfo, CitationResponse
from app.agents.insight_agent import InsightAgent, InsightResponse
from app.agents.summarizer_agent import SummarizerAgent, SummaryResponse

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "SummarizerAgent",
    "SummaryResponse",
    "InsightAgent",
    "InsightResponse",
    "CitationAgent",
    "CitationInfo",
    "CitationResponse",
]
