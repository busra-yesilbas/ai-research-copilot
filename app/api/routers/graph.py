"""Knowledge graph API router: GET /graph.

Endpoints
---------
``GET /api/v1/graph``
    Return the persisted knowledge graph (nodes + edges).  If no graph has
    been built yet, returns an empty graph (not an error).

``POST /api/v1/graph/build``
    Build and persist the knowledge graph from the current vector store contents.
    Returns the new graph.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_vector_store
from app.knowledge_graph.graph_builder import GraphBuilder, GraphData
from app.utils.logger import get_logger
from app.vector_store.faiss_store import VectorStore

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Knowledge Graph"])


@router.get(
    "/graph",
    response_model=GraphData,
    summary="Get knowledge graph",
    description=(
        "Return the persisted knowledge graph extracted from indexed papers. "
        "Call ``POST /graph/build`` first to populate the graph."
    ),
)
async def get_graph() -> GraphData:
    """Load and return the persisted knowledge graph.

    Returns an empty graph (not a 404) if no graph has been built yet,
    so clients can always expect a valid ``GraphData`` response.
    """
    builder = GraphBuilder()
    try:
        return builder.load()
    except FileNotFoundError:
        logger.info("No graph found; returning empty graph.")
        return GraphData()
    except Exception as exc:
        logger.error("Failed to load graph: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load graph: {exc}",
        )


@router.post(
    "/graph/build",
    response_model=GraphData,
    summary="Build knowledge graph from indexed papers",
    description=(
        "Extract entities from all indexed chunks and build a co-occurrence "
        "knowledge graph.  Persists the graph for future ``GET /graph`` calls."
    ),
)
async def build_graph(
    store: VectorStore = Depends(get_vector_store),
) -> GraphData:
    """Build the knowledge graph from the current vector store contents."""
    if store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No documents indexed. Upload papers with POST /upload-paper first.",
        )

    # Reconstruct chunk-like objects from the store's metadata
    class _FakeChunk:
        def __init__(self, text: str, document_id: str) -> None:
            self.text = text
            self.document_id = document_id

    # Access internal entries (implementation detail of VectorStore)
    entries = getattr(store, "_entries", [])
    fake_chunks = [
        _FakeChunk(
            text=e.get("text", ""),
            document_id=e.get("metadata", {}).get("document_id", ""),
        )
        for e in entries
    ]

    builder = GraphBuilder()
    graph = builder.build(fake_chunks)

    try:
        builder.save(graph)
    except Exception as exc:
        logger.warning("Could not persist graph: %s", exc)

    return graph
