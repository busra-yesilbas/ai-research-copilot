"""Embedding model abstractions and implementations.

Design principles
-----------------
- **Lazy loading**: ``EmbeddingModel`` defers importing and loading
  ``sentence-transformers`` until the first call to ``embed_texts`` or
  ``embed_query``.  This keeps import time near-zero for code paths that do
  not need embeddings.
- **Consistent interface**: All implementations satisfy ``BaseEmbeddingModel``,
  making the rest of the system backend-agnostic.
- **L2-normalised output**: Every implementation returns unit-norm vectors so
  that the dot product between any two vectors equals their cosine similarity.
  This is the contract relied upon by the vector store layer.
- ``FakeEmbeddingModel``: Deterministic, zero-dependency embeddings for tests.
  No network access, no torch, no sentence-transformers required.

Typical usage::

    from app.embeddings.embedding_model import EmbeddingModel

    model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.embed_texts(["deep learning", "natural language processing"])
    query_vec = model.embed_query("attention mechanism")
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEmbeddingModel(ABC):
    """Protocol / ABC for all embedding backends.

    Every implementation must:
    - Return ``list[list[float]]`` from ``embed_texts`` with shape
      ``(len(texts), dim)``.
    - Return **L2-normalised** (unit-norm) vectors so cosine similarity equals
      the inner product.
    - Be deterministic: identical input → identical output.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension (number of floats per vector)."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of input strings.  Empty strings are handled
                   gracefully (a placeholder embedding is returned).

        Returns:
            List of float lists, one per input text, each of length
            :attr:`dim`.
        """

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        This is a convenience wrapper around :meth:`embed_texts`.  Override
        in subclasses when the model applies different preprocessing for
        queries vs. passages (e.g. INSTRUCTOR-style models).

        Args:
            text: Query string.

        Returns:
            Float list of length :attr:`dim`.
        """
        if not text.strip():
            logger.warning("Empty query passed to embed_query; returning zero vector.")
            return [0.0] * self.dim
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Sentence-Transformers implementation
# ---------------------------------------------------------------------------


class EmbeddingModel(BaseEmbeddingModel):
    """Sentence-Transformers embedding model with lazy loading.

    The underlying ``SentenceTransformer`` object is created on the first call
    to :meth:`embed_texts`; construction of ``EmbeddingModel`` itself is
    essentially free.

    Args:
        model_name:  HuggingFace model id (default: ``all-MiniLM-L6-v2``).
        device:      Torch device string — ``"cpu"``, ``"cuda"``, ``"mps"``,
                     etc.
        batch_size:  Number of texts encoded per forward pass.
        cache_dir:   Local directory for model weights.  If ``None``, the
                     default HuggingFace / sentence-transformers cache is used.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._model: Any = None  # loaded lazily
        self._dim: int | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Embedding dimension (inferred from model on first access)."""
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier."""
        return self._model_name

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* using the loaded sentence-transformers model.

        Args:
            texts: Strings to embed.  Empty strings are replaced with a
                   single space before encoding.

        Returns:
            List of L2-normalised float lists, shape ``(len(texts), dim)``.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if not texts:
            return []

        self._load()

        # Protect against empty strings which some models reject
        safe = [t if t.strip() else " " for t in texts]

        logger.debug(
            "Encoding %d texts with '%s' (batch_size=%d)…",
            len(safe),
            self._model_name,
            self._batch_size,
        )

        import numpy as np

        raw: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            safe,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # returns unit vectors
        )
        return raw.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Handles empty input gracefully by returning a zero vector of the
        correct dimension.

        Args:
            text: Query string.

        Returns:
            L2-normalised float list of length :attr:`dim`.
        """
        if not text.strip():
            logger.warning("Empty query passed to embed_query; returning zero vector.")
            return [0.0] * self.dim
        return self.embed_texts([text])[0]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the sentence-transformers model (idempotent)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EmbeddingModel. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info(
            "Loading sentence-transformers model '%s' on device='%s'…",
            self._model_name,
            self._device,
        )

        kwargs: dict[str, Any] = {"device": self._device}
        if self._cache_dir:
            kwargs["cache_folder"] = self._cache_dir

        self._model = SentenceTransformer(self._model_name, **kwargs)

        # Infer dimension from a tiny probe (avoids an extra property on the class)
        import numpy as np

        probe: np.ndarray = self._model.encode(["_"], convert_to_numpy=True)
        self._dim = int(probe.shape[1])

        logger.info(
            "Model '%s' loaded: dim=%d device=%s",
            self._model_name,
            self._dim,
            self._device,
        )


# ---------------------------------------------------------------------------
# Deterministic fake model for testing
# ---------------------------------------------------------------------------


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic pseudo-embedding model — for testing only.

    Generates embeddings by hashing the input text with SHA-256 and
    converting the digest bytes into a float vector that is then L2-normalised.
    This is **not** semantic, but it is:

    - **Deterministic**: same text → same vector, always.
    - **Unique (with high probability)**: different texts almost always
      produce different vectors.
    - **Zero-dependency**: uses only Python stdlib (``hashlib``, ``struct``,
      ``math``).  No torch, no sentence-transformers, no network.

    Args:
        dim: Number of dimensions in each returned vector (default: 32).
    """

    def __init__(self, dim: int = 32) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic unit vectors for *texts*.

        Args:
            texts: Input strings (empty list returns empty list).

        Returns:
            List of L2-normalised float lists, shape ``(len(texts), dim)``.
        """
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (identical to ``embed_texts([text])[0]``).

        Args:
            text: Query string.

        Returns:
            L2-normalised float list of length :attr:`dim`.
        """
        return self._hash_vector(text)

    # ── Private ────────────────────────────────────────────────────────────────

    def _hash_vector(self, text: str) -> list[float]:
        """Convert *text* to a deterministic unit-norm float vector.

        Steps:
        1. SHA-256 hash of the UTF-8 encoded text → 32 bytes.
        2. Repeat digest to fill ``dim * 4`` bytes.
        3. Unpack as **signed 32-bit integers** (avoids IEEE 754 NaN/Inf
           that would appear when treating arbitrary bytes as float32).
        4. Cast integers to float and L2-normalise.

        Args:
            text: Input string (empty string handled).

        Returns:
            Normalised float list of length :attr:`dim`.
        """
        # Normalise empty input so different "empty" representations unify.
        key = text.strip() if text.strip() else "__empty__"
        digest = sha256(key.encode("utf-8")).digest()  # 32 bytes

        n_bytes = self._dim * 4
        # Repeat digest until we have enough bytes
        repeats = n_bytes // 32 + 1
        raw = (digest * repeats)[:n_bytes]

        # Signed int32 values are always finite — unlike float32 which can be NaN/Inf
        raw_ints: list[int] = list(struct.unpack(f"{self._dim}i", raw))
        floats: list[float] = [float(v) for v in raw_ints]

        # L2 normalise
        norm = math.sqrt(sum(f * f for f in floats))
        if norm < 1e-9:
            # Extremely unlikely (all-zero int vector) — return uniform unit vector
            val = 1.0 / math.sqrt(self._dim)
            return [val] * self._dim

        return [f / norm for f in floats]
