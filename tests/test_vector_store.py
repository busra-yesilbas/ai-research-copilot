"""Unit and integration tests for app.vector_store.faiss_store.

All tests use :class:`FakeEmbeddingModel` so they run fully offline.

Test groups
-----------
``TestSklearnVectorStoreBasic``   — add / count / search basics
``TestSklearnVectorStoreSearch``  — correctness of top-k ordering
``TestSklearnVectorStorePersistence`` — save / load roundtrip
``TestFaissVectorStore``          — same tests for FAISS (skipped if unavailable)
``TestGetVectorStore``            — factory helper
``TestLoadVectorStore``           — load_vector_store factory helper
``TestSearchResult``              — Pydantic model validation
``TestDeterministicChunkIds``     — chunk_id derivation rules
``TestAddEmbeddings``             — add_embeddings direct path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.embeddings.embedding_model import FakeEmbeddingModel
from app.vector_store.faiss_store import (
    SearchResult,
    SklearnVectorStore,
    VectorStore,
    get_vector_store,
    load_vector_store,
)


# ---------------------------------------------------------------------------
# Shared fixture — small corpus
# ---------------------------------------------------------------------------

# These exact strings are used as query targets so tests are deterministic.
CORPUS = [
    "Deep learning for natural language processing",
    "Attention is all you need transformer architecture",
    "Convolutional neural networks for image classification",
    "Recurrent networks and long short-term memory",
    "Graph neural networks for molecular property prediction",
]


@pytest.fixture
def fake_model() -> FakeEmbeddingModel:
    """Return a FakeEmbeddingModel with dim=32."""
    return FakeEmbeddingModel(dim=32)


@pytest.fixture
def populated_store(fake_model: FakeEmbeddingModel) -> SklearnVectorStore:
    """Return a SklearnVectorStore pre-loaded with CORPUS texts."""
    store = SklearnVectorStore(fake_model)
    store.add(CORPUS)
    return store


# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------


class TestSearchResult:
    """SearchResult Pydantic model is well-formed."""

    def test_construction(self) -> None:
        sr = SearchResult(chunk_id="abc", score=0.95, text="hello", metadata={})
        assert sr.chunk_id == "abc"
        assert sr.score == 0.95
        assert sr.text == "hello"
        assert sr.metadata == {}

    def test_metadata_default_factory(self) -> None:
        sr = SearchResult(chunk_id="x", score=0.1, text="y")
        assert isinstance(sr.metadata, dict)

    def test_score_can_be_negative(self) -> None:
        sr = SearchResult(chunk_id="x", score=-0.3, text="y")
        assert sr.score == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# SklearnVectorStore — basic operations
# ---------------------------------------------------------------------------


class TestSklearnVectorStoreBasic:
    """Core add / count / search operations."""

    def test_empty_store_count_zero(self, fake_model: FakeEmbeddingModel) -> None:
        store = SklearnVectorStore(fake_model)
        assert store.count() == 0

    def test_add_returns_correct_number_of_ids(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(CORPUS)
        assert len(ids) == len(CORPUS)

    def test_count_matches_added_texts(
        self, populated_store: SklearnVectorStore
    ) -> None:
        assert populated_store.count() == len(CORPUS)

    def test_add_empty_list_returns_empty(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add([])
        assert ids == []
        assert store.count() == 0

    def test_ids_are_strings(self, fake_model: FakeEmbeddingModel) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(["some text"])
        assert all(isinstance(cid, str) for cid in ids)

    def test_ids_are_16_hex_chars_when_no_metadata(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(["test content"])
        assert len(ids[0]) == 16
        assert all(c in "0123456789abcdef" for c in ids[0])

    def test_custom_chunk_id_from_metadata(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        custom_id = "my-custom-chunk-id"
        ids = store.add(["text"], metadatas=[{"chunk_id": custom_id}])
        assert ids[0] == custom_id

    def test_returned_ids_match_stored_ids(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        """IDs returned by add() match chunk_id in search results."""
        store = SklearnVectorStore(fake_model)
        ids = store.add(["unique text here"])
        q = fake_model.embed_query("unique text here")
        results = store.search(q, k=1)
        assert results[0].chunk_id == ids[0]

    def test_incremental_add(self, fake_model: FakeEmbeddingModel) -> None:
        store = SklearnVectorStore(fake_model)
        store.add(["first batch"])
        store.add(["second batch"])
        assert store.count() == 2

    def test_search_on_empty_store_returns_empty(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        q = fake_model.embed_query("anything")
        results = store.search(q, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# SklearnVectorStore — search correctness
# ---------------------------------------------------------------------------


class TestSklearnVectorStoreSearch:
    """Top-k search returns the exact-match text as the best result."""

    def test_exact_match_is_top_result(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        """Querying with text identical to an indexed entry should score 1.0."""
        query_text = CORPUS[0]
        q = fake_model.embed_query(query_text)
        results = populated_store.search(q, k=5)
        assert len(results) > 0
        assert results[0].text == query_text

    def test_top_score_is_one_for_exact_match(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query(CORPUS[2])
        results = populated_store.search(q, k=1)
        assert abs(results[0].score - 1.0) < 1e-4

    def test_results_ordered_by_score_descending(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query(CORPUS[1])
        results = populated_store.search(q, k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_result_count(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query(CORPUS[0])
        for k in [1, 2, 3]:
            results = populated_store.search(q, k=k)
            assert len(results) == k

    def test_k_larger_than_store_returns_all(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query(CORPUS[0])
        results = populated_store.search(q, k=999)
        assert len(results) == populated_store.count()

    def test_result_contains_correct_text(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query(CORPUS[3])
        results = populated_store.search(q, k=1)
        assert results[0].text == CORPUS[3]

    def test_metadata_round_trips_through_search(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        meta = {"source": "paper.pdf", "page_num": 3, "title": "Test Paper"}
        store.add(["some content"], metadatas=[meta | {"chunk_id": "cid-1"}])
        q = fake_model.embed_query("some content")
        results = store.search(q, k=1)
        assert results[0].metadata["source"] == "paper.pdf"
        assert results[0].metadata["page_num"] == 3

    def test_all_result_scores_in_valid_range(
        self, populated_store: SklearnVectorStore, fake_model: FakeEmbeddingModel
    ) -> None:
        q = fake_model.embed_query("random query text")
        results = populated_store.search(q, k=5)
        for r in results:
            assert -1.01 <= r.score <= 1.01, f"Score out of range: {r.score}"


# ---------------------------------------------------------------------------
# SklearnVectorStore — persistence
# ---------------------------------------------------------------------------


class TestSklearnVectorStorePersistence:
    """Save / load roundtrip preserves the ability to search."""

    def test_save_creates_expected_files(
        self,
        populated_store: SklearnVectorStore,
        tmp_path: Path,
    ) -> None:
        populated_store.save(str(tmp_path / "idx"))
        assert (tmp_path / "idx" / "vectors.npy").exists()
        assert (tmp_path / "idx" / "metadata.json").exists()

    def test_load_restores_count(
        self,
        populated_store: SklearnVectorStore,
        fake_model: FakeEmbeddingModel,
        tmp_path: Path,
    ) -> None:
        index_path = str(tmp_path / "idx")
        populated_store.save(index_path)
        loaded = SklearnVectorStore.load(index_path, fake_model)
        assert loaded.count() == populated_store.count()

    def test_load_restores_search_results(
        self,
        populated_store: SklearnVectorStore,
        fake_model: FakeEmbeddingModel,
        tmp_path: Path,
    ) -> None:
        index_path = str(tmp_path / "idx")
        populated_store.save(index_path)
        loaded = SklearnVectorStore.load(index_path, fake_model)

        q = fake_model.embed_query(CORPUS[0])
        original_results = populated_store.search(q, k=3)
        loaded_results = loaded.search(q, k=3)

        assert [r.chunk_id for r in loaded_results] == [
            r.chunk_id for r in original_results
        ]

    def test_load_top_result_matches_original(
        self,
        populated_store: SklearnVectorStore,
        fake_model: FakeEmbeddingModel,
        tmp_path: Path,
    ) -> None:
        index_path = str(tmp_path / "idx")
        populated_store.save(index_path)
        loaded = SklearnVectorStore.load(index_path, fake_model)

        q = fake_model.embed_query(CORPUS[2])
        original_top = populated_store.search(q, k=1)[0]
        loaded_top = loaded.search(q, k=1)[0]

        assert loaded_top.chunk_id == original_top.chunk_id
        assert abs(loaded_top.score - original_top.score) < 1e-4

    def test_save_empty_store_raises(
        self, fake_model: FakeEmbeddingModel, tmp_path: Path
    ) -> None:
        store = SklearnVectorStore(fake_model)
        with pytest.raises(RuntimeError, match="empty"):
            store.save(str(tmp_path / "idx"))

    def test_load_missing_path_raises(
        self, fake_model: FakeEmbeddingModel, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            SklearnVectorStore.load(str(tmp_path / "nonexistent"), fake_model)

    def test_load_vector_store_helper(
        self,
        populated_store: SklearnVectorStore,
        fake_model: FakeEmbeddingModel,
        tmp_path: Path,
    ) -> None:
        index_path = str(tmp_path / "idx")
        populated_store.save(index_path)
        loaded = load_vector_store(index_path, fake_model)
        assert isinstance(loaded, SklearnVectorStore)
        assert loaded.count() == populated_store.count()


# ---------------------------------------------------------------------------
# get_vector_store factory
# ---------------------------------------------------------------------------


class TestGetVectorStore:
    """get_vector_store returns the correct backend."""

    def test_returns_vector_store_instance(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = get_vector_store(fake_model, prefer_faiss=False)
        assert isinstance(store, VectorStore)

    def test_returns_sklearn_when_faiss_not_preferred(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = get_vector_store(fake_model, prefer_faiss=False)
        assert isinstance(store, SklearnVectorStore)

    def test_returns_sklearn_when_faiss_unavailable(
        self, fake_model: FakeEmbeddingModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force faiss import to fail → should return SklearnVectorStore."""
        import builtins
        real_import = builtins.__import__

        def _block_faiss(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "faiss":
                raise ImportError("faiss blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_faiss)
        store = get_vector_store(fake_model, prefer_faiss=True)
        assert isinstance(store, SklearnVectorStore)

    def test_new_store_is_empty(self, fake_model: FakeEmbeddingModel) -> None:
        store = get_vector_store(fake_model, prefer_faiss=False)
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Deterministic chunk IDs
# ---------------------------------------------------------------------------


class TestDeterministicChunkIds:
    """Chunk IDs are stable and derived from content or supplied metadata."""

    def test_same_text_same_id(self, fake_model: FakeEmbeddingModel) -> None:
        store_a = SklearnVectorStore(fake_model)
        store_b = SklearnVectorStore(fake_model)
        ids_a = store_a.add(["stable content"])
        ids_b = store_b.add(["stable content"])
        assert ids_a[0] == ids_b[0]

    def test_different_texts_different_ids(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(["content alpha", "content beta"])
        assert ids[0] != ids[1]

    def test_custom_id_overrides_default(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(["text"], metadatas=[{"chunk_id": "override-id"}])
        assert ids[0] == "override-id"

    def test_auto_id_is_16_hex_chars(self, fake_model: FakeEmbeddingModel) -> None:
        store = SklearnVectorStore(fake_model)
        ids = store.add(["any text without chunk_id"])
        assert len(ids[0]) == 16
        assert all(c in "0123456789abcdef" for c in ids[0])


# ---------------------------------------------------------------------------
# add_embeddings direct path
# ---------------------------------------------------------------------------


class TestAddEmbeddings:
    """add_embeddings bypasses internal embedding computation."""

    def test_add_embeddings_count_correct(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        embeddings = fake_model.embed_texts(CORPUS[:3])
        store.add_embeddings(embeddings, CORPUS[:3])
        assert store.count() == 3

    def test_add_embeddings_search_works(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        embeddings = fake_model.embed_texts(CORPUS)
        store.add_embeddings(embeddings, CORPUS)
        q = fake_model.embed_query(CORPUS[0])
        results = store.search(q, k=1)
        assert results[0].text == CORPUS[0]

    def test_add_embeddings_respects_chunk_id_metadata(
        self, fake_model: FakeEmbeddingModel
    ) -> None:
        store = SklearnVectorStore(fake_model)
        embeddings = fake_model.embed_texts(["hello"])
        ids = store.add_embeddings(
            embeddings, ["hello"], metadatas=[{"chunk_id": "pre-computed-id"}]
        )
        assert ids[0] == "pre-computed-id"


# ---------------------------------------------------------------------------
# FAISS store (skipped when faiss is not installed)
# ---------------------------------------------------------------------------


class TestFaissVectorStore:
    """Mirror tests for FaissVectorStore — skipped when faiss is unavailable."""

    @pytest.fixture(autouse=True)
    def _require_faiss(self) -> None:
        pytest.importorskip("faiss", reason="faiss-cpu not installed; skipping FAISS tests.")

    def test_add_and_count(self, fake_model: FakeEmbeddingModel) -> None:
        from app.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(fake_model)
        store.add(CORPUS)
        assert store.count() == len(CORPUS)

    def test_search_exact_match(self, fake_model: FakeEmbeddingModel) -> None:
        from app.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(fake_model)
        store.add(CORPUS)
        q = fake_model.embed_query(CORPUS[0])
        results = store.search(q, k=1)
        assert results[0].text == CORPUS[0]
        assert abs(results[0].score - 1.0) < 1e-3

    def test_save_load_roundtrip(
        self, fake_model: FakeEmbeddingModel, tmp_path: Path
    ) -> None:
        from app.vector_store.faiss_store import FaissVectorStore

        store = FaissVectorStore(fake_model)
        store.add(CORPUS)
        store.save(str(tmp_path / "faiss_idx"))

        loaded = FaissVectorStore.load(str(tmp_path / "faiss_idx"), fake_model)
        assert loaded.count() == len(CORPUS)

        q = fake_model.embed_query(CORPUS[1])
        r_orig = store.search(q, k=1)[0]
        r_load = loaded.search(q, k=1)[0]
        assert r_orig.chunk_id == r_load.chunk_id
