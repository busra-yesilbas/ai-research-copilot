"""Vector store package.

Public API::

    from app.vector_store import (
        SearchResult,
        VectorStore,
        FaissVectorStore,
        SklearnVectorStore,
        get_vector_store,
        load_vector_store,
    )
"""

from app.vector_store.faiss_store import (
    FaissVectorStore,
    SearchResult,
    SklearnVectorStore,
    VectorStore,
    get_vector_store,
    load_vector_store,
)

__all__ = [
    "SearchResult",
    "VectorStore",
    "FaissVectorStore",
    "SklearnVectorStore",
    "get_vector_store",
    "load_vector_store",
]
