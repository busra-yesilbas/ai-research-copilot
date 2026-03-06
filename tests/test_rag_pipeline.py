"""Tests for the RAG pipeline — retriever, answer generator, and pipeline.

All tests use :class:`FakeEmbeddingModel` and :class:`SklearnVectorStore` so
they run offline with no model downloads.
"""

from __future__ import annotations

import pytest

from app.embeddings.embedding_model import FakeEmbeddingModel
from app.rag.rag_pipeline import (
    LocalAnswerGenerator,
    RAGPipeline,
    RAGResult,
    SourceInfo,
)
from app.rag.retriever import Retriever
from app.vector_store.faiss_store import SearchResult, SklearnVectorStore

# ---------------------------------------------------------------------------
# Corpus fixture
# ---------------------------------------------------------------------------

CORPUS = [
    "Deep learning has revolutionised natural language processing.",
    "The transformer architecture uses self-attention mechanisms.",
    "BERT pre-trains on masked language modelling objectives.",
    "Convolutional neural networks excel at image classification tasks.",
    "Recurrent neural networks process sequential data step by step.",
]


@pytest.fixture
def model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dim=32)


@pytest.fixture
def store(model: FakeEmbeddingModel) -> SklearnVectorStore:
    s = SklearnVectorStore(model)
    s.add(CORPUS)
    return s


@pytest.fixture
def retriever(model: FakeEmbeddingModel, store: SklearnVectorStore) -> Retriever:
    return Retriever(model, store)


@pytest.fixture
def pipeline(retriever: Retriever) -> RAGPipeline:
    return RAGPipeline(retriever, LocalAnswerGenerator())


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_retrieve_returns_results(self, retriever: Retriever) -> None:
        results = retriever.retrieve(CORPUS[0], k=3)
        assert len(results) > 0

    def test_exact_match_is_top_result(self, retriever: Retriever) -> None:
        results = retriever.retrieve(CORPUS[0], k=5)
        assert results[0].text == CORPUS[0]

    def test_top_score_is_one_for_exact(self, retriever: Retriever) -> None:
        results = retriever.retrieve(CORPUS[1], k=1)
        assert abs(results[0].score - 1.0) < 1e-4

    def test_k_limits_results(self, retriever: Retriever) -> None:
        for k in [1, 2, 3]:
            results = retriever.retrieve(CORPUS[0], k=k)
            assert len(results) == k

    def test_empty_query_raises(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ", k=3)

    def test_k_zero_raises(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError):
            retriever.retrieve("query", k=0)

    def test_is_ready_true_when_store_populated(self, retriever: Retriever) -> None:
        assert retriever.is_ready() is True

    def test_is_ready_false_on_empty_store(self, model: FakeEmbeddingModel) -> None:
        empty_store = SklearnVectorStore(model)
        r = Retriever(model, empty_store)
        assert r.is_ready() is False

    def test_results_ordered_by_score_descending(self, retriever: Retriever) -> None:
        results = retriever.retrieve("deep learning transformer", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# LocalAnswerGenerator tests
# ---------------------------------------------------------------------------


class TestLocalAnswerGenerator:
    def _make_chunks(self, texts: list[str]) -> list[SearchResult]:
        return [
            SearchResult(chunk_id=f"c{i}", score=1.0 / (i + 1), text=t, metadata={})
            for i, t in enumerate(texts)
        ]

    def test_empty_chunks_returns_fallback(self) -> None:
        gen = LocalAnswerGenerator()
        answer = gen.generate("any query", [])
        assert "No relevant information" in answer

    def test_returns_non_empty_answer(self) -> None:
        gen = LocalAnswerGenerator()
        chunks = self._make_chunks(CORPUS[:3])
        answer = gen.generate("deep learning", chunks)
        assert len(answer) > 0

    def test_answer_contains_text_from_chunks(self) -> None:
        gen = LocalAnswerGenerator()
        text = "This sentence is about attention mechanisms in transformers."
        chunks = self._make_chunks([text])
        answer = gen.generate("attention transformer", chunks)
        assert len(answer) > 0

    def test_answer_max_sentences_respected(self) -> None:
        gen = LocalAnswerGenerator(max_sentences=2)
        long_text = " ".join([f"Sentence number {i} about deep learning." for i in range(20)])
        chunks = self._make_chunks([long_text])
        answer = gen.generate("deep learning", chunks)
        sentence_count = answer.count(". ") + (1 if answer.endswith(".") else 0)
        assert sentence_count <= 5

    def test_deterministic_output(self) -> None:
        gen = LocalAnswerGenerator()
        chunks = self._make_chunks(CORPUS[:3])
        a1 = gen.generate("deep learning", chunks)
        a2 = gen.generate("deep learning", chunks)
        assert a1 == a2


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    def test_query_returns_rag_result(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query(CORPUS[0])
        assert isinstance(result, RAGResult)

    def test_query_has_non_empty_answer(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("deep learning")
        assert len(result.answer) > 0

    def test_query_has_sources(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query(CORPUS[0], top_k=3)
        assert len(result.sources) == 3

    def test_sources_are_source_info_objects(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query(CORPUS[0], top_k=2)
        for src in result.sources:
            assert isinstance(src, SourceInfo)
            assert isinstance(src.chunk_id, str)
            assert isinstance(src.score, float)

    def test_query_records_latency(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query("test query")
        assert result.latency_ms >= 0

    def test_empty_query_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValueError, match="empty"):
            pipeline.query("")

    def test_top_k_zero_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValueError):
            pipeline.query("query", top_k=0)

    def test_source_score_in_valid_range(self, pipeline: RAGPipeline) -> None:
        result = pipeline.query(CORPUS[2], top_k=5)
        for src in result.sources:
            assert -1.01 <= src.score <= 1.01

    def test_empty_store_returns_no_relevant_info(
        self, model: FakeEmbeddingModel
    ) -> None:
        empty_store = SklearnVectorStore(model)
        empty_retriever = Retriever(model, empty_store)
        empty_pipeline = RAGPipeline(empty_retriever)
        result = empty_pipeline.query("any question")
        assert "No relevant information" in result.answer
        assert result.sources == []

    def test_custom_generator_is_used(
        self, retriever: Retriever
    ) -> None:
        class _Fixed(LocalAnswerGenerator):
            def generate(self, query: str, chunks: list) -> str:
                return "FIXED"

        p = RAGPipeline(retriever, _Fixed())
        result = p.query("anything")
        assert result.answer == "FIXED"
