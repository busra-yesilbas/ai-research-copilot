"""FastAPI application factory and entry point.

Run locally::

    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

Lifespan
--------
On startup:
  1. Initialise the embedding model (lazy — weights not loaded until first
     call to ``embed_texts``).  Falls back gracefully if sentence-transformers
     is not installed; embedding endpoints will return ``503``.
  2. Load the vector store from disk if an index exists, otherwise create an
     empty in-memory store.
  3. Wire up the RAG pipeline (Retriever + LocalAnswerGenerator).

On shutdown:
  - Persist the vector store to disk if it contains any vectors.

State
-----
All shared resources are stored on ``app.state`` and injected into route
handlers via the dependency functions in :mod:`app.api.deps`.

  ``app.state.embedding_model``  — :class:`~app.embeddings.embedding_model.EmbeddingModel`
  ``app.state.vector_store``     — :class:`~app.vector_store.faiss_store.VectorStore`
  ``app.state.rag_pipeline``     — :class:`~app.rag.rag_pipeline.RAGPipeline`
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config.settings import get_settings
from app.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(level=settings.log_level, fmt=settings.log_format)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_version() -> str:
    """Return the installed package version, falling back to settings."""
    try:
        return _pkg_version("ai-research-copilot")
    except PackageNotFoundError:
        return settings.app_version


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down shared resources."""
    logger.info(
        "Starting %s v%s [env=%s]",
        settings.app_name,
        _resolve_version(),
        settings.environment,
    )

    # Each block checks for pre-existing state first so that tests can inject
    # fake models before entering the TestClient context manager without having
    # them overwritten by the lifespan.

    # ── 1. Embedding model (lazy — no weights loaded yet) ──────────────────
    embedding_model = getattr(app.state, "embedding_model", None)
    if embedding_model is None:
        try:
            from app.embeddings.embedding_model import EmbeddingModel

            embedding_model = EmbeddingModel(
                model_name=settings.embedding_model_name,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                cache_dir=str(settings.models_dir_resolved),
            )
            logger.info(
                "Embedding model initialised (lazy): '%s'", settings.embedding_model_name
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise embedding model: %s", exc)
        app.state.embedding_model = embedding_model
    else:
        logger.info("Using pre-configured embedding model: %s", type(embedding_model).__name__)

    # ── 2. Vector store ────────────────────────────────────────────────────
    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None and embedding_model is not None:
        try:
            from app.vector_store.faiss_store import get_vector_store, load_vector_store

            index_path = settings.index_dir_resolved
            meta_file = index_path / "metadata.json"
            if meta_file.exists():
                vector_store = load_vector_store(str(index_path), embedding_model)
                logger.info(
                    "Loaded vector store from '%s' (%d vectors)", index_path, vector_store.count()
                )
            else:
                vector_store = get_vector_store(
                    embedding_model,
                    prefer_faiss=(settings.vector_store_backend == "faiss"),
                )
                logger.info(
                    "Created new empty vector store (backend=%s)", type(vector_store).__name__
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise vector store: %s", exc)
        app.state.vector_store = vector_store
    elif vector_store is not None:
        logger.info("Using pre-configured vector store (%d vectors)", vector_store.count())

    # ── 3. RAG pipeline ────────────────────────────────────────────────────
    rag_pipeline = getattr(app.state, "rag_pipeline", None)
    if rag_pipeline is None and embedding_model is not None and vector_store is not None:
        try:
            from app.rag.rag_pipeline import LocalAnswerGenerator, RAGPipeline
            from app.rag.retriever import Retriever

            retriever = Retriever(embedding_model, vector_store)
            rag_pipeline = RAGPipeline(retriever, LocalAnswerGenerator())
            logger.info("RAG pipeline ready (top_k=%d)", settings.retrieval_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise RAG pipeline: %s", exc)
        app.state.rag_pipeline = rag_pipeline
    elif rag_pipeline is not None:
        logger.info("Using pre-configured RAG pipeline.")

    logger.info("%s started successfully.", settings.app_name)
    yield

    # ── Shutdown: persist index ────────────────────────────────────────────
    vs = getattr(app.state, "vector_store", None)
    if vs is not None and vs.count() > 0:
        try:
            index_path = str(settings.index_dir_resolved)
            vs.save(index_path)
            logger.info("Vector store persisted to '%s'", index_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist vector store on shutdown: %s", exc)

    logger.info("Shutting down %s", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application.

    Using a factory function keeps each test run isolated — tests call
    ``create_app()`` directly instead of importing the module-level singleton.
    """
    _app = FastAPI(
        title=settings.app_name,
        version=_resolve_version(),
        description=(
            "A production-grade system for ingesting, querying, and reasoning "
            "over academic research papers — fully offline, no paid APIs required."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Meta routes ───────────────────────────────────────────────────────
    _register_meta_routes(_app)

    # ── Feature routers ───────────────────────────────────────────────────
    from app.api.routers.agents import router as agents_router
    from app.api.routers.graph import router as graph_router
    from app.api.routers.rag import router as rag_router

    _app.include_router(rag_router)
    _app.include_router(agents_router)
    _app.include_router(graph_router)

    return _app


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str
    environment: str


class VersionResponse(BaseModel):
    """Version information response."""

    app_name: str
    version: str


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


def _register_meta_routes(app: FastAPI) -> None:
    """Attach /health and /version to *app*."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness probe",
        tags=["Meta"],
    )
    async def health() -> HealthResponse:
        """Return ``{"status": "ok"}`` when the service is running."""
        return HealthResponse(status="ok", environment=settings.environment)

    @app.get(
        "/version",
        response_model=VersionResponse,
        summary="Application version",
        tags=["Meta"],
    )
    async def version() -> VersionResponse:
        """Return the application name and current semantic version."""
        return VersionResponse(
            app_name=settings.app_name,
            version=_resolve_version(),
        )


# ---------------------------------------------------------------------------
# Module-level app singleton (used by uvicorn and tests)
# ---------------------------------------------------------------------------

app: FastAPI = create_app()
