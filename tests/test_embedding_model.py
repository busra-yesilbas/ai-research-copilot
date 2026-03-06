"""Unit tests for app.embeddings.embedding_model.

All tests in this module use :class:`FakeEmbeddingModel` — a deterministic,
zero-dependency implementation — so the suite runs fully offline with no model
downloads, no torch, and no sentence-transformers required.

The ``EmbeddingModel`` (sentence-transformers) tests are guarded by
``pytest.importorskip("sentence_transformers")`` and will be skipped
automatically in environments where the package is not installed.
"""

from __future__ import annotations

import math

import pytest

from app.embeddings.embedding_model import BaseEmbeddingModel, FakeEmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(vec: list[float]) -> float:
    """Return the L2 norm of *vec*."""
    return math.sqrt(sum(x * x for x in vec))


def _dot(a: list[float], b: list[float]) -> float:
    """Return the dot product of *a* and *b*."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# BaseEmbeddingModel contract (tested via FakeEmbeddingModel)
# ---------------------------------------------------------------------------


class TestBaseEmbeddingModelContract:
    """FakeEmbeddingModel satisfies the BaseEmbeddingModel ABC contract."""

    def test_is_subclass_of_base(self) -> None:
        assert issubclass(FakeEmbeddingModel, BaseEmbeddingModel)

    def test_instance_is_base(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        assert isinstance(model, BaseEmbeddingModel)


# ---------------------------------------------------------------------------
# FakeEmbeddingModel — dimensionality and shape
# ---------------------------------------------------------------------------


class TestFakeEmbeddingModelShape:
    """Output shapes are correct for various dim values."""

    @pytest.mark.parametrize("dim", [4, 8, 16, 32, 64, 128, 256])
    def test_dim_property(self, dim: int) -> None:
        model = FakeEmbeddingModel(dim=dim)
        assert model.dim == dim

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FakeEmbeddingModel(dim=0)

    def test_embed_texts_returns_correct_count(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        texts = ["a", "b", "c", "d", "e"]
        result = model.embed_texts(texts)
        assert len(result) == len(texts)

    def test_embed_texts_returns_correct_dim(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        result = model.embed_texts(["hello world"])
        assert len(result[0]) == 32

    def test_embed_query_returns_single_vector(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        result = model.embed_query("a query")
        assert isinstance(result, list)
        assert len(result) == 16

    def test_embed_empty_list_returns_empty(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        assert model.embed_texts([]) == []

    def test_single_text_list(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        result = model.embed_texts(["only one"])
        assert len(result) == 1
        assert len(result[0]) == 16


# ---------------------------------------------------------------------------
# FakeEmbeddingModel — vector properties
# ---------------------------------------------------------------------------


class TestFakeEmbeddingModelVectorProperties:
    """Vectors are unit-norm and all finite."""

    @pytest.mark.parametrize("text", ["hello", "world", "deep learning", "x", ""])
    def test_vector_is_unit_norm(self, text: str) -> None:
        model = FakeEmbeddingModel(dim=32)
        vec = model.embed_texts([text])[0]
        norm = _norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"norm={norm} for text={text!r}"

    def test_all_floats_are_finite(self) -> None:
        model = FakeEmbeddingModel(dim=64)
        texts = ["text one", "text two", "text three"]
        for vec in model.embed_texts(texts):
            for v in vec:
                assert math.isfinite(v), f"Non-finite value {v}"

    def test_cosine_of_same_text_is_one(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        v = model.embed_texts(["identical text"])[0]
        cosine = _dot(v, v)
        assert abs(cosine - 1.0) < 1e-5

    def test_embed_query_matches_embed_texts(self) -> None:
        """embed_query and embed_texts([text])[0] must return the same vector."""
        model = FakeEmbeddingModel(dim=32)
        text = "some query text"
        via_texts = model.embed_texts([text])[0]
        via_query = model.embed_query(text)
        assert via_texts == via_query


# ---------------------------------------------------------------------------
# FakeEmbeddingModel — determinism
# ---------------------------------------------------------------------------


class TestFakeEmbeddingModelDeterminism:
    """Identical inputs always produce identical outputs."""

    def test_same_text_same_vector(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        v1 = model.embed_texts(["reproducible text"])[0]
        v2 = model.embed_texts(["reproducible text"])[0]
        assert v1 == v2

    def test_different_texts_different_vectors(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        texts = [
            "natural language processing",
            "computer vision research",
            "reinforcement learning agents",
            "transformer architecture",
        ]
        vecs = model.embed_texts(texts)
        # All pairs must differ
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                assert vecs[i] != vecs[j], f"Texts {i} and {j} produced identical vectors"

    def test_new_instance_same_result(self) -> None:
        """Two separate FakeEmbeddingModel instances must produce the same output."""
        text = "test determinism across instances"
        v1 = FakeEmbeddingModel(dim=32).embed_texts([text])[0]
        v2 = FakeEmbeddingModel(dim=32).embed_texts([text])[0]
        assert v1 == v2

    def test_order_preserved(self) -> None:
        model = FakeEmbeddingModel(dim=16)
        texts = ["alpha", "beta", "gamma"]
        result = model.embed_texts(texts)
        for text, vec in zip(texts, result):
            expected = model.embed_texts([text])[0]
            assert vec == expected


# ---------------------------------------------------------------------------
# FakeEmbeddingModel — cosine similarity sanity
# ---------------------------------------------------------------------------


class TestFakeEmbeddingModelSimilarity:
    """Cosine similarity of unit vectors equals their dot product."""

    def test_exact_match_has_cosine_one(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        text = "this exact sentence"
        v_a = model.embed_texts([text])[0]
        v_b = model.embed_query(text)
        cosine = _dot(v_a, v_b)
        assert abs(cosine - 1.0) < 1e-5

    def test_cosine_in_valid_range(self) -> None:
        model = FakeEmbeddingModel(dim=32)
        texts = ["paper alpha", "paper beta", "completely different"]
        vecs = model.embed_texts(texts)
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                cosine = _dot(vecs[i], vecs[j])
                assert -1.0 - 1e-5 <= cosine <= 1.0 + 1e-5, (
                    f"Cosine out of range: {cosine}"
                )


# ---------------------------------------------------------------------------
# EmbeddingModel (sentence-transformers) — skipped if not installed
# ---------------------------------------------------------------------------


class TestSentenceTransformersModel:
    """Tests for the real EmbeddingModel — skipped when sentence-transformers
    is unavailable (offline environments, CI without model cache, etc.)."""

    @pytest.fixture(autouse=True)
    def _require_sentence_transformers(self) -> None:
        pytest.importorskip(
            "sentence_transformers",
            reason="sentence-transformers not installed; skipping real model tests.",
        )

    def test_lazy_load_does_not_raise_on_construction(self) -> None:
        from app.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Should not raise — model is loaded lazily
        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embed_texts_returns_correct_shape(self) -> None:
        from app.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texts = ["hello world", "test sentence"]
        result = model.embed_texts(texts)
        assert len(result) == 2
        assert len(result[0]) == model.dim

    def test_dim_is_384_for_minilm(self) -> None:
        from app.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert model.dim == 384

    def test_embed_texts_unit_norm(self) -> None:
        from app.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vecs = model.embed_texts(["a short text"])
        norm = _norm(vecs[0])
        assert abs(norm - 1.0) < 1e-4

    def test_embed_empty_returns_empty(self) -> None:
        from app.embeddings.embedding_model import EmbeddingModel

        model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert model.embed_texts([]) == []
