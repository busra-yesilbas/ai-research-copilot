"""Tests for the research agents (M5).

All tests use FakeEmbeddingModel + SklearnVectorStore and run fully offline.
"""

from __future__ import annotations

import pytest

from app.agents.citation_agent import CitationAgent, CitationResponse
from app.agents.insight_agent import InsightAgent, InsightResponse
from app.agents.summarizer_agent import SummarizerAgent, SummaryResponse
from app.embeddings.embedding_model import FakeEmbeddingModel
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.vector_store.faiss_store import SklearnVectorStore

CORPUS = [
    "We propose a novel transformer-based model for machine translation.",
    "The attention mechanism allows the model to focus on relevant tokens.",
    "Our experiments on WMT14 English-German show BLEU score of 28.4.",
    "The dataset consists of 4.5 million sentence pairs for training.",
    "Limitations include high computational cost and memory requirements.",
    "Future work includes extending the model to low-resource languages.",
    "We compare with LSTM baselines and show significant improvements.",
    "The model achieves state-of-the-art results on multiple benchmarks.",
]

METADATAS = [
    {"chunk_id": f"c{i}", "document_id": "doc001", "file_name": "paper.pdf",
     "title": "Attention is All You Need", "page_num": (i // 2) + 1}
    for i in range(len(CORPUS))
]


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dim=32)


@pytest.fixture
def pipeline(fake_model: FakeEmbeddingModel) -> RAGPipeline:
    store = SklearnVectorStore(fake_model)
    store.add(CORPUS, METADATAS)
    retriever = Retriever(fake_model, store)
    return RAGPipeline(retriever)


# ---------------------------------------------------------------------------
# SummarizerAgent
# ---------------------------------------------------------------------------


class TestSummarizerAgent:
    def test_run_returns_summary_response(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        assert isinstance(response, SummaryResponse)

    def test_summary_is_non_empty(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        assert len(response.summary) > 0

    def test_sources_populated(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        assert len(response.sources) > 0

    def test_topic_narrows_summary(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline, top_k=2)
        response = agent.run(topic="attention mechanism")
        assert isinstance(response.summary, str)
        assert len(response.summary) > 0

    def test_agent_name_is_summarizer(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        assert response.agent == "summarizer"

    def test_latency_is_non_negative(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        assert response.latency_ms >= 0

    def test_no_duplicate_sources(self, pipeline: RAGPipeline) -> None:
        agent = SummarizerAgent(pipeline)
        response = agent.run()
        chunk_ids = [s.chunk_id for s in response.sources]
        assert len(chunk_ids) == len(set(chunk_ids))


# ---------------------------------------------------------------------------
# InsightAgent
# ---------------------------------------------------------------------------


class TestInsightAgent:
    def test_run_returns_insight_response(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert isinstance(response, InsightResponse)

    def test_contributions_non_empty(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert isinstance(response.contributions, str)
        assert len(response.contributions) > 0

    def test_limitations_field_present(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert hasattr(response, "limitations")
        assert isinstance(response.limitations, str)

    def test_research_gaps_field_present(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert hasattr(response, "research_gaps")

    def test_methodology_field_present(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert hasattr(response, "methodology")

    def test_agent_name_is_insight(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert response.agent == "insight"

    def test_sources_populated(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        assert len(response.sources) > 0

    def test_no_duplicate_sources(self, pipeline: RAGPipeline) -> None:
        agent = InsightAgent(pipeline)
        response = agent.run()
        chunk_ids = [s.chunk_id for s in response.sources]
        assert len(chunk_ids) == len(set(chunk_ids))


# ---------------------------------------------------------------------------
# CitationAgent
# ---------------------------------------------------------------------------


class TestCitationAgent:
    def test_run_returns_citation_response(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="transformer model for NLP")
        assert isinstance(response, CitationResponse)

    def test_citations_populated(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="transformer machine translation")
        assert len(response.citations) > 0

    def test_chunks_populated(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="BLEU score results")
        assert len(response.chunks) > 0

    def test_citations_sorted_by_relevance(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="attention mechanism")
        relevances = [c.relevance for c in response.citations]
        assert relevances == sorted(relevances, reverse=True)

    def test_agent_name_is_citation(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="model")
        assert response.agent == "citation"

    def test_empty_query_uses_default(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="")
        assert isinstance(response, CitationResponse)

    def test_citation_has_required_fields(self, pipeline: RAGPipeline) -> None:
        agent = CitationAgent(pipeline)
        response = agent.run(query="dataset training")
        for citation in response.citations:
            assert hasattr(citation, "document_id")
            assert hasattr(citation, "relevance")
            assert hasattr(citation, "chunk_count")
            assert citation.chunk_count >= 1
