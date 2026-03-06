"""Application configuration.

All settings are read from environment variables or a ``.env`` file located
in the working directory.  Pydantic-settings handles coercion, validation, and
provides rich error messages when a value is out of range.

Usage::

    from app.config.settings import get_settings

    settings = get_settings()
    print(settings.app_name)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment / ``.env`` file.

    Every field maps 1-to-1 to an environment variable of the same name
    (case-insensitive).  Defaults are chosen so the app works out-of-the-box
    without any ``.env`` file present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(
        default="AI Research Copilot",
        description="Human-readable application name.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic version string.",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment tag.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (extra verbosity, auto-reload hints).",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root logger level.",
    )
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log output format. Use 'json' in production / containers.",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for all runtime data files.",
    )
    index_dir: Path = Field(
        default=Path("data/index"),
        description="Directory where the vector index is persisted.",
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="Local directory used as the sentence-transformers model cache.",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model id passed to SentenceTransformer().",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        description="Number of texts encoded per forward pass.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Torch device string ('cpu', 'cuda', 'mps', …).",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        ge=64,
        description="Maximum number of characters in a single chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Number of overlapping characters between consecutive chunks.",
    )

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_backend: Literal["faiss", "sklearn"] = Field(
        default="faiss",
        description=(
            "Preferred vector store backend. "
            "Automatically falls back to 'sklearn' if faiss-cpu is not installed."
        ),
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        description="Default number of nearest neighbours returned by the retriever.",
    )
    vector_dim: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Embedding vector dimension.  If None the dimension is inferred from "
            "the model at runtime and stored in metadata.json."
        ),
    )

    # ── LLM / RAG ─────────────────────────────────────────────────────────────
    llm_provider: Literal["local", "openai", "anthropic"] = Field(
        default="local",
        description=(
            "'local' uses extractive / template-based synthesis (no API key needed). "
            "Set to 'openai' or 'anthropic' and supply the matching API key env var "
            "to use a real LLM."
        ),
    )
    llm_model_name: str = Field(
        default="local-extractive",
        description="Model identifier forwarded to the configured LLM provider.",
    )

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    neo4j_uri: str | None = Field(
        default=None,
        description="Neo4j bolt URI (e.g. 'bolt://localhost:7687'). None → JSON backend.",
    )
    neo4j_user: str | None = Field(
        default=None,
        description="Neo4j username.",
    )
    neo4j_password: str | None = Field(
        default=None,
        description="Neo4j password.",
    )

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = Field(
        default="0.0.0.0",
        description="Uvicorn bind host.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Uvicorn bind port.",
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        description="Number of uvicorn worker processes.",
    )

    # ── Derived helpers (not env-configurable) ───────────────────────────────
    @property
    def raw_pdf_dir(self) -> Path:
        """Absolute path to the raw PDF input directory; created on access."""
        path = (self.data_dir / "raw").expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_dir(self) -> Path:
        """Absolute path to the processed JSON output directory; created on access."""
        path = (self.data_dir / "processed").expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_dir_resolved(self) -> Path:
        """Return ``index_dir`` as an absolute path, creating it if absent."""
        path = self.index_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir_resolved(self) -> Path:
        """Return ``models_dir`` as an absolute path, creating it if absent."""
        path = self.models_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton.

    The result is cached after the first call, so subsequent calls are free.
    In tests you can clear the cache with ``get_settings.cache_clear()`` before
    constructing a ``Settings`` with overridden env vars.
    """
    return Settings()
