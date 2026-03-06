"""Vector store implementations — FAISS and pure-numpy/sklearn fallback.

Architecture
------------
``VectorStore``  (abstract base)
    ├── ``FaissVectorStore``   — FAISS ``IndexFlatIP`` (cosine via L2-norm)
    └── ``SklearnVectorStore`` — NumPy matrix multiply fallback

Both stores share:
- The **same public interface**: ``add``, ``search``, ``save``, ``load``,
  ``count``.
- **Cosine similarity** scoring (higher = better match): both normalise
  vectors to unit length before indexing/querying.
- **Deterministic chunk IDs**: derived from ``metadata["chunk_id"]`` when
  supplied, or from ``SHA-256(text)[:16]`` otherwise.
- **JSON + binary persistence**: metadata (chunk_id, text, dict) is saved as
  ``metadata.json``; vectors go to ``index.faiss`` or ``vectors.npy``.

Factory helpers
---------------
``get_vector_store(embedding_model, prefer_faiss)``
    Returns ``FaissVectorStore`` if ``faiss`` is importable, else
    ``SklearnVectorStore``.

``load_vector_store(path, embedding_model)``
    Reads ``metadata.json`` from *path* to detect the store type, then
    delegates to the correct ``load`` classmethod.

Typical usage::

    from app.embeddings.embedding_model import EmbeddingModel
    from app.vector_store.faiss_store import get_vector_store, load_vector_store

    model = EmbeddingModel()
    store = get_vector_store(model)
    store.add(["text one", "text two"], [{"source": "paper.pdf"}, {}])
    store.save("data/index")

    # Later…
    store2 = load_vector_store("data/index", model)
    results = store2.search(model.embed_query("text one"), k=3)
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.embeddings.embedding_model import BaseEmbeddingModel

logger = get_logger(__name__)

# File names used by both store types
_META_FILE = "metadata.json"
_FAISS_FILE = "index.faiss"
_NUMPY_FILE = "vectors.npy"


# ---------------------------------------------------------------------------
# SearchResult — the atomic retrieval result
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single result returned by :meth:`VectorStore.search`.

    Attributes:
        chunk_id: Stable identifier for the retrieved chunk.
        score:    Cosine similarity score in ``[-1, 1]``; higher is better.
        text:     Original text of the chunk.
        metadata: Arbitrary metadata stored alongside the chunk at index time.
    """

    chunk_id: str = Field(..., description="Stable chunk identifier.")
    score: float = Field(..., description="Cosine similarity (higher = more relevant).")
    text: str = Field(..., description="Original chunk text.")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Concrete implementations must satisfy this interface so the rest of the
    system (RAG pipeline, evaluation) is backend-agnostic.
    """

    def __init__(self, embedding_model: "BaseEmbeddingModel") -> None:
        """
        Args:
            embedding_model: The model used to embed texts in :meth:`add`.
        """
        self._embedding_model = embedding_model

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Embed *texts* and add them to the store.

        Args:
            texts:     Strings to embed and index.
            metadatas: Optional per-text metadata dicts.  If a dict contains
                       the key ``"chunk_id"``, that value is used as the
                       identifier; otherwise a deterministic hash of the text
                       is generated.

        Returns:
            List of chunk IDs in the same order as *texts*.
        """

    def add_embeddings(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add pre-computed embeddings directly (avoids re-embedding).

        Useful when embeddings have already been computed in a batch outside
        the store (e.g. for progress tracking or distributed indexing).

        Args:
            embeddings: Pre-computed L2-normalised vectors.
            texts:      Original strings (stored for later retrieval).
            metadatas:  Optional per-text metadata.

        Returns:
            List of chunk IDs.
        """
        return self._add_vectors(embeddings, texts, metadatas)

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Return the *k* most similar chunks to *query_embedding*.

        Args:
            query_embedding: Pre-computed query vector (will be L2-normalised
                             internally before comparison).
            k:               Maximum number of results to return.

        Returns:
            Ordered list of :class:`SearchResult` (best match first).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to *path* (a directory).

        Args:
            path: Directory path.  Created if it does not exist.
        """

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        embedding_model: "BaseEmbeddingModel",
    ) -> "VectorStore":
        """Load a previously saved store from *path*.

        Args:
            path:            Directory written by :meth:`save`.
            embedding_model: Embedding model to attach to the loaded store.

        Returns:
            Fully initialised store ready to call :meth:`search` on.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed vectors."""

    # ── Abstract helper (implemented per-store) ───────────────────────────────

    @abstractmethod
    def _add_vectors(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None,
    ) -> list[str]:
        """Internal: index pre-computed *embeddings*."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_chunk_id(text: str, metadata: dict[str, Any]) -> str:
        """Return chunk_id from metadata, or derive from text content."""
        if "chunk_id" in metadata:
            return str(metadata["chunk_id"])
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _write_metadata(
        path: Path,
        store_type: str,
        dim: int,
        entries: list[dict[str, Any]],
    ) -> None:
        """Write metadata.json to *path*."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / _META_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "store_type": store_type,
                    "vector_dim": dim,
                    "count": len(entries),
                    "entries": entries,
                },
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        """Read and return the metadata.json from *path*."""
        meta_file = path / _META_FILE
        if not meta_file.exists():
            raise FileNotFoundError(f"No {_META_FILE} found in '{path}'.")
        with open(meta_file, encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# FAISS-backed store
# ---------------------------------------------------------------------------


class FaissVectorStore(VectorStore):
    """Vector store backed by FAISS ``IndexFlatIP`` (exact inner-product search).

    Cosine similarity is achieved by L2-normalising all vectors before
    indexing; the inner product of two unit vectors equals their cosine.

    Availability
    ------------
    ``faiss-cpu`` must be installed (``pip install faiss-cpu``).  If it is not
    available, use :class:`SklearnVectorStore` instead or let
    :func:`get_vector_store` choose automatically.
    """

    def __init__(self, embedding_model: "BaseEmbeddingModel") -> None:
        super().__init__(embedding_model)
        self._index: Any = None   # faiss.IndexFlatIP, built lazily
        self._entries: list[dict[str, Any]] = []
        self._dim: int | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Embed *texts* and add them to the FAISS index.

        Args:
            texts:     Strings to embed and index.
            metadatas: Optional per-text metadata dicts.

        Returns:
            List of chunk IDs.
        """
        if not texts:
            return []
        embeddings = self._embedding_model.embed_texts(texts)
        return self._add_vectors(embeddings, texts, metadatas)

    def _add_vectors(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None,
    ) -> list[str]:
        try:
            import faiss  # type: ignore[import]
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FaissVectorStore. "
                "Install with: pip install faiss-cpu"
            ) from exc

        if not embeddings:
            return []

        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)

        if self._index is None:
            self._dim = vecs.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)
            logger.debug("Created FAISS IndexFlatIP with dim=%d", self._dim)

        self._index.add(vecs)

        chunk_ids: list[str] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            cid = self._resolve_chunk_id(text, meta)
            self._entries.append({"chunk_id": cid, "text": text, "metadata": meta})
            chunk_ids.append(cid)

        logger.debug(
            "FAISS store: added %d vectors → total=%d", len(texts), self.count()
        )
        return chunk_ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for the *k* nearest vectors to *query_embedding*.

        Args:
            query_embedding: Query vector (normalised internally).
            k:               Number of results.

        Returns:
            Ordered list of :class:`SearchResult` (best score first).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        try:
            import faiss  # type: ignore[import]
            import numpy as np
        except ImportError:
            return []

        actual_k = min(k, self._index.ntotal)
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)

        scores, indices = self._index.search(q, actual_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self._entries[idx]
            results.append(
                SearchResult(
                    chunk_id=entry["chunk_id"],
                    score=float(score),
                    text=entry["text"],
                    metadata=entry["metadata"],
                )
            )
        return results

    def save(self, path: str) -> None:
        """Persist index to *path*.

        Writes:
        - ``index.faiss`` — the FAISS binary index.
        - ``metadata.json`` — chunk_id, text, metadata for every entry.

        Args:
            path: Directory path; created if absent.
        """
        if self._index is None:
            raise RuntimeError("Cannot save an empty FaissVectorStore.")

        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("faiss-cpu required to save FaissVectorStore.") from exc

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(dir_path / _FAISS_FILE))
        self._write_metadata(dir_path, "faiss", self._dim or 0, self._entries)

        logger.info(
            "FaissVectorStore saved to '%s' (%d vectors, dim=%d)",
            path,
            self.count(),
            self._dim,
        )

    @classmethod
    def load(
        cls,
        path: str,
        embedding_model: "BaseEmbeddingModel",
    ) -> "FaissVectorStore":
        """Load a FaissVectorStore from *path*.

        Args:
            path:            Directory previously written by :meth:`save`.
            embedding_model: Embedding model to attach.

        Returns:
            Loaded :class:`FaissVectorStore`.

        Raises:
            FileNotFoundError: If *path* or required files are missing.
            ImportError:       If ``faiss-cpu`` is not installed.
        """
        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required to load FaissVectorStore."
            ) from exc

        dir_path = Path(path)
        faiss_file = dir_path / _FAISS_FILE
        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {faiss_file}")

        store = cls(embedding_model)
        store._index = faiss.read_index(str(faiss_file))

        meta = cls._read_metadata(dir_path)
        store._dim = meta["vector_dim"]
        store._entries = meta["entries"]

        logger.info(
            "FaissVectorStore loaded from '%s': %d vectors, dim=%d",
            path,
            store.count(),
            store._dim,
        )
        return store

    def count(self) -> int:
        """Return the number of indexed vectors."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# NumPy / sklearn fallback store
# ---------------------------------------------------------------------------


class SklearnVectorStore(VectorStore):
    """Pure-NumPy vector store — always available (no C extensions needed).

    Stores vectors as a 2-D float32 NumPy array and performs cosine search
    via a matrix-vector multiply.  Correct for datasets up to ~100 k vectors;
    for larger scale, switch to FAISS.

    Persistence
    -----------
    - ``vectors.npy`` — all normalised vectors (shape ``(n, dim)``).
    - ``metadata.json`` — chunk_id, text, metadata per entry.
    """

    def __init__(self, embedding_model: "BaseEmbeddingModel") -> None:
        super().__init__(embedding_model)
        self._vectors: "Any" = None   # np.ndarray[float32], shape (n, dim)
        self._entries: list[dict[str, Any]] = []
        self._dim: int | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Embed *texts* and append them to the numpy vector matrix.

        Args:
            texts:     Strings to embed and index.
            metadatas: Optional per-text metadata dicts.

        Returns:
            List of chunk IDs.
        """
        if not texts:
            return []
        embeddings = self._embedding_model.embed_texts(texts)
        return self._add_vectors(embeddings, texts, metadatas)

    def _add_vectors(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None,
    ) -> list[str]:
        if not embeddings:
            return []

        import numpy as np

        vecs = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms  # L2 normalise

        if self._vectors is None:
            self._dim = int(vecs.shape[1])
            self._vectors = vecs
        else:
            self._vectors = np.vstack([self._vectors, vecs])

        chunk_ids: list[str] = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            cid = self._resolve_chunk_id(text, meta)
            self._entries.append({"chunk_id": cid, "text": text, "metadata": meta})
            chunk_ids.append(cid)

        logger.debug(
            "SklearnStore: added %d vectors → total=%d", len(texts), self.count()
        )
        return chunk_ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[SearchResult]:
        """Return top-*k* chunks by cosine similarity.

        Args:
            query_embedding: Query vector (normalised internally).
            k:               Number of results.

        Returns:
            List of :class:`SearchResult` sorted by score descending.
        """
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        import numpy as np

        actual_k = min(k, self._vectors.shape[0])

        q = np.array(query_embedding, dtype="float32")
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            q = q / q_norm

        # Cosine similarity = dot product of unit vectors
        scores: np.ndarray = self._vectors @ q  # shape (n,)
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            entry = self._entries[int(idx)]
            results.append(
                SearchResult(
                    chunk_id=entry["chunk_id"],
                    score=float(scores[idx]),
                    text=entry["text"],
                    metadata=entry["metadata"],
                )
            )
        return results

    def save(self, path: str) -> None:
        """Persist vectors and metadata to *path*.

        Writes:
        - ``vectors.npy``   — float32 NumPy array, shape ``(n, dim)``.
        - ``metadata.json`` — chunk_id, text, metadata.

        Args:
            path: Directory path; created if absent.
        """
        if self._vectors is None:
            raise RuntimeError("Cannot save an empty SklearnVectorStore.")

        import numpy as np

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        np.save(str(dir_path / _NUMPY_FILE), self._vectors)
        self._write_metadata(dir_path, "sklearn", self._dim or 0, self._entries)

        logger.info(
            "SklearnVectorStore saved to '%s' (%d vectors, dim=%d)",
            path,
            self.count(),
            self._dim,
        )

    @classmethod
    def load(
        cls,
        path: str,
        embedding_model: "BaseEmbeddingModel",
    ) -> "SklearnVectorStore":
        """Load a SklearnVectorStore from *path*.

        Args:
            path:            Directory previously written by :meth:`save`.
            embedding_model: Embedding model to attach.

        Returns:
            Loaded :class:`SklearnVectorStore`.

        Raises:
            FileNotFoundError: If *path* or required files are missing.
        """
        import numpy as np

        dir_path = Path(path)
        npy_file = dir_path / _NUMPY_FILE
        if not npy_file.exists():
            raise FileNotFoundError(f"Vectors file not found: {npy_file}")

        store = cls(embedding_model)
        store._vectors = np.load(str(npy_file))
        store._dim = int(store._vectors.shape[1])

        meta = cls._read_metadata(dir_path)
        store._entries = meta["entries"]

        logger.info(
            "SklearnVectorStore loaded from '%s': %d vectors, dim=%d",
            path,
            store.count(),
            store._dim,
        )
        return store

    def count(self) -> int:
        """Return the number of indexed vectors."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def get_vector_store(
    embedding_model: "BaseEmbeddingModel",
    prefer_faiss: bool = True,
) -> VectorStore:
    """Return the best available vector store for *embedding_model*.

    Tries FAISS first (if *prefer_faiss* is True); falls back to
    :class:`SklearnVectorStore` if ``faiss`` cannot be imported.

    Args:
        embedding_model: Embedding model injected into the store.
        prefer_faiss:    If ``True``, try FAISS before sklearn.

    Returns:
        An empty :class:`VectorStore` ready to accept ``add`` calls.
    """
    if prefer_faiss:
        try:
            import faiss  # noqa: F401  # type: ignore[import]
            logger.info("FAISS available — using FaissVectorStore.")
            return FaissVectorStore(embedding_model)
        except ImportError:
            logger.info("faiss not available — falling back to SklearnVectorStore.")

    return SklearnVectorStore(embedding_model)


def load_vector_store(
    path: str,
    embedding_model: "BaseEmbeddingModel",
) -> VectorStore:
    """Load a persisted vector store from *path*.

    Reads ``metadata.json`` to detect the store type (``"faiss"`` or
    ``"sklearn"``), then delegates to the appropriate ``load`` classmethod.

    Args:
        path:            Directory written by :meth:`VectorStore.save`.
        embedding_model: Embedding model to attach to the loaded store.

    Returns:
        Loaded :class:`VectorStore`.

    Raises:
        FileNotFoundError: If *path* or ``metadata.json`` is missing.
        ValueError:        If the stored ``store_type`` is unrecognised.
    """
    dir_path = Path(path)
    meta_file = dir_path / _META_FILE
    if not meta_file.exists():
        raise FileNotFoundError(f"No {_META_FILE} found in '{path}'.")

    with open(meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)

    store_type: str = meta.get("store_type", "sklearn")

    if store_type == "faiss":
        return FaissVectorStore.load(path, embedding_model)
    elif store_type == "sklearn":
        return SklearnVectorStore.load(path, embedding_model)
    else:
        raise ValueError(
            f"Unrecognised store_type '{store_type}' in {meta_file}. "
            "Expected 'faiss' or 'sklearn'."
        )
