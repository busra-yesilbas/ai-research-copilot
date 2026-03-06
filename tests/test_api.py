"""Integration tests for all FastAPI endpoints (M1 + M4–M7).

M1 tests (health, version, docs) use a plain test client with no overrides.

M4–M7 tests inject a ``FakeEmbeddingModel`` + ``SklearnVectorStore`` through
``app.state`` so the tests run offline without sentence-transformers, FAISS,
or any model downloads.
"""

from __future__ import annotations

import io
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config.settings import get_settings


# ---------------------------------------------------------------------------
# Plain client (M1 — no shared state needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a TestClient for a fresh application instance."""
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Client with pre-populated fake state (M4–M7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rich_client() -> Generator[TestClient, None, None]:
    """TestClient with FakeEmbeddingModel + populated SklearnVectorStore injected."""
    from app.embeddings.embedding_model import FakeEmbeddingModel
    from app.rag.rag_pipeline import LocalAnswerGenerator, RAGPipeline
    from app.rag.retriever import Retriever
    from app.vector_store.faiss_store import SklearnVectorStore

    model = FakeEmbeddingModel(dim=32)
    store = SklearnVectorStore(model)
    texts = [
        "Deep learning has revolutionised natural language processing tasks.",
        "The transformer uses multi-head self-attention mechanisms.",
        "BERT pre-trains on masked language modeling and next sentence prediction.",
        "We evaluate on SQuAD 2.0 and achieve an exact-match score of 83.1.",
        "Limitations include high memory requirements during fine-tuning.",
    ]
    metadatas = [
        {"chunk_id": f"ck{i}", "document_id": "doc-test", "file_name": "test.pdf",
         "title": "Test Paper", "page_num": i + 1}
        for i in range(len(texts))
    ]
    store.add(texts, metadatas)
    retriever = Retriever(model, store)
    pipeline = RAGPipeline(retriever, LocalAnswerGenerator())

    app = create_app()
    app.state.embedding_model = model
    app.state.vector_store = store
    app.state.rag_pipeline = pipeline

    with TestClient(app) as c:
        yield c


# ===========================================================================
# M1 — Meta endpoints
# ===========================================================================


class TestHealthEndpoint:
    def test_status_code_is_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_response_has_ok_status(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_response_contains_environment(self, client: TestClient) -> None:
        assert "environment" in client.get("/health").json()

    def test_environment_is_valid_value(self, client: TestClient) -> None:
        assert client.get("/health").json()["environment"] in (
            "development", "staging", "production"
        )

    def test_content_type_is_json(self, client: TestClient) -> None:
        assert "application/json" in client.get("/health").headers["content-type"]


class TestVersionEndpoint:
    def test_status_code_is_200(self, client: TestClient) -> None:
        assert client.get("/version").status_code == 200

    def test_response_contains_app_name(self, client: TestClient) -> None:
        data = client.get("/version").json()
        assert data["app_name"] == get_settings().app_name

    def test_response_contains_version(self, client: TestClient) -> None:
        data = client.get("/version").json()
        assert isinstance(data["version"], str) and len(data["version"]) > 0

    def test_version_is_semver_like(self, client: TestClient) -> None:
        assert "." in client.get("/version").json()["version"]


class TestDocsEndpoint:
    def test_swagger_ui_accessible(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_redoc_accessible(self, client: TestClient) -> None:
        assert client.get("/redoc").status_code == 200

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200

    def test_openapi_json_has_meta_paths(self, client: TestClient) -> None:
        data = client.get("/openapi.json").json()
        assert "/health" in data["paths"]
        assert "/version" in data["paths"]


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/this-does-not-exist").status_code == 404


# ===========================================================================
# M4 — RAG endpoints
# ===========================================================================


class TestAskEndpoint:
    """POST /api/v1/ask"""

    def test_status_200_with_valid_query(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "deep learning"})
        assert resp.status_code == 200

    def test_response_has_answer(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "deep learning"})
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_has_sources(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "transformer"})
        assert "sources" in resp.json()

    def test_response_echoes_query(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "BERT model"})
        assert resp.json()["query"] == "BERT model"

    def test_top_k_respected(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "deep learning", "top_k": 2})
        assert len(resp.json()["sources"]) <= 2

    def test_returns_latency(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": "attention"})
        assert "latency_ms" in resp.json()

    def test_empty_query_returns_422(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/ask", json={"query": ""})
        assert resp.status_code == 422

    def test_without_index_returns_503(self, client: TestClient) -> None:
        """Default client has no state — should return 503."""
        resp = client.post("/api/v1/ask", json={"query": "test"})
        assert resp.status_code == 503


class TestOpenAPIHasNewPaths:
    """Verify new endpoints are registered in the OpenAPI schema."""

    def test_ask_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/ask" in paths

    def test_upload_paper_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/upload-paper" in paths

    def test_summarize_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/summarize" in paths

    def test_research_insights_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/research-insights" in paths

    def test_related_work_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/related-work" in paths

    def test_graph_in_openapi(self, rich_client: TestClient) -> None:
        paths = rich_client.get("/openapi.json").json()["paths"]
        assert "/api/v1/graph" in paths


# ===========================================================================
# M5 — Agent endpoints
# ===========================================================================


class TestSummarizeEndpoint:
    def test_status_200(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/summarize", json={})
        assert resp.status_code == 200

    def test_response_has_summary(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/summarize", json={})
        assert "summary" in resp.json()
        assert len(resp.json()["summary"]) > 0

    def test_topic_parameter_accepted(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/summarize", json={"topic": "attention mechanism"})
        assert resp.status_code == 200

    def test_without_index_returns_503(self, client: TestClient) -> None:
        resp = client.post("/api/v1/summarize", json={})
        assert resp.status_code == 503


class TestResearchInsightsEndpoint:
    def test_status_200(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/research-insights", json={})
        assert resp.status_code == 200

    def test_response_has_contributions(self, rich_client: TestClient) -> None:
        data = rich_client.post("/api/v1/research-insights", json={}).json()
        assert "contributions" in data

    def test_response_has_limitations(self, rich_client: TestClient) -> None:
        data = rich_client.post("/api/v1/research-insights", json={}).json()
        assert "limitations" in data

    def test_response_has_research_gaps(self, rich_client: TestClient) -> None:
        data = rich_client.post("/api/v1/research-insights", json={}).json()
        assert "research_gaps" in data


class TestRelatedWorkEndpoint:
    def test_status_200(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/related-work", json={"query": "deep learning"})
        assert resp.status_code == 200

    def test_response_has_citations(self, rich_client: TestClient) -> None:
        data = rich_client.post("/api/v1/related-work", json={"query": "transformer"}).json()
        assert "citations" in data

    def test_response_has_query(self, rich_client: TestClient) -> None:
        data = rich_client.post("/api/v1/related-work", json={"query": "BERT"}).json()
        assert data["query"] == "BERT"

    def test_missing_query_returns_422(self, rich_client: TestClient) -> None:
        resp = rich_client.post("/api/v1/related-work", json={})
        assert resp.status_code == 422


# ===========================================================================
# M7 — Knowledge graph endpoint
# ===========================================================================


class TestGraphEndpoint:
    def test_get_graph_returns_200(self, rich_client: TestClient) -> None:
        resp = rich_client.get("/api/v1/graph")
        assert resp.status_code == 200

    def test_graph_response_has_nodes_key(self, rich_client: TestClient) -> None:
        data = rich_client.get("/api/v1/graph").json()
        assert "nodes" in data

    def test_graph_response_has_edges_key(self, rich_client: TestClient) -> None:
        data = rich_client.get("/api/v1/graph").json()
        assert "edges" in data
