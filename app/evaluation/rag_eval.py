"""RAG evaluation harness — Recall@k and MRR.

Overview
--------
The harness generates a synthetic evaluation dataset from indexed chunks,
then measures how well the retriever finds the correct chunk for each
synthetic query.

Synthetic query generation
--------------------------
For each sampled chunk, the **first meaningful sentence** is extracted as a
synthetic query.  Because the sentence came verbatim from the chunk, a
perfect retriever should rank that chunk first.

Metrics
-------
``Recall@k``
    Fraction of queries for which the correct chunk appears in the top-*k*
    results.  Range: ``[0, 1]``.

``MRR (Mean Reciprocal Rank)``
    Average of ``1/rank`` for each query, where *rank* is the position of the
    correct chunk in the result list (0 if not found).  Range: ``[0, 1]``.

Typical usage::

    from app.evaluation.rag_eval import RagEvaluator

    evaluator = RagEvaluator(retriever)
    report = evaluator.evaluate(chunks, n_queries=20, k=5)
    print(report.summary())
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.ingestion.models import Chunk
    from app.rag.retriever import Retriever

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SyntheticQuery(BaseModel):
    """A single synthetic (query, correct_chunk_id) pair.

    Attributes:
        query:             The synthetic query string (first sentence of chunk).
        relevant_chunk_id: The chunk ID that should appear as the top result.
        source_text:       The full original chunk text (for debugging).
    """

    query: str
    relevant_chunk_id: str
    source_text: str = ""


class EvalReport(BaseModel):
    """Evaluation report with computed metrics.

    Attributes:
        recall_at_k:  Recall@k value.
        mrr:          Mean Reciprocal Rank.
        k:            The retrieval depth used.
        n_queries:    Number of synthetic queries evaluated.
        n_hits:       Queries where the correct chunk appeared in top-k.
        latency_ms:   Total evaluation wall-clock time.
        per_query:    Per-query results (optional, for debugging).
    """

    recall_at_k: float
    mrr: float
    k: int
    n_queries: int
    n_hits: int
    latency_ms: float = 0.0
    per_query: list[dict] = Field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable one-block summary string."""
        sep = "-" * 48
        return (
            f"\n{sep}\n"
            f"  RAG Evaluation Report\n"
            f"{sep}\n"
            f"  Queries evaluated : {self.n_queries}\n"
            f"  Retrieval depth k : {self.k}\n"
            f"  Recall@{self.k:<2}          : {self.recall_at_k:.4f}\n"
            f"  MRR               : {self.mrr:.4f}\n"
            f"  Hits              : {self.n_hits}/{self.n_queries}\n"
            f"  Latency           : {self.latency_ms:.1f} ms\n"
            f"{sep}"
        )


# ---------------------------------------------------------------------------
# Synthetic query generation
# ---------------------------------------------------------------------------


def generate_synthetic_queries(
    chunks: "list[Chunk]",
    n: int = 20,
    min_query_len: int = 20,
) -> list[SyntheticQuery]:
    """Sample chunks and extract their first meaningful sentence as a query.

    Args:
        chunks:        List of indexed :class:`~app.ingestion.models.Chunk` objects.
        n:             Maximum number of synthetic queries to generate.
        min_query_len: Minimum character length for a valid query sentence.

    Returns:
        List of :class:`SyntheticQuery` objects (may be shorter than *n* if
        chunks are too short).
    """
    if not chunks:
        return []

    queries: list[SyntheticQuery] = []
    # Evenly sample across the chunk list so we cover the full document
    step = max(1, len(chunks) // n)
    indices = list(range(0, len(chunks), step))[:n]

    for idx in indices:
        chunk = chunks[idx]
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk.text)
            if len(s.strip()) >= min_query_len
        ]
        if not sentences:
            continue
        query_text = sentences[0][:200]  # cap length
        queries.append(
            SyntheticQuery(
                query=query_text,
                relevant_chunk_id=chunk.id,
                source_text=chunk.text[:200],
            )
        )

    logger.debug("Generated %d synthetic queries from %d chunks", len(queries), len(chunks))
    return queries


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def recall_at_k(result_ids_list: list[list[str]], relevant_ids: list[str], k: int) -> float:
    """Compute Recall@k.

    Args:
        result_ids_list: For each query, an ordered list of retrieved chunk IDs.
        relevant_ids:    For each query, the single correct chunk ID.
        k:               Retrieval depth.

    Returns:
        Fraction of queries where the correct chunk appears in the top-*k* results.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(
        1
        for retrieved, correct in zip(result_ids_list, relevant_ids)
        if correct in retrieved[:k]
    )
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    result_ids_list: list[list[str]], relevant_ids: list[str]
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        result_ids_list: For each query, an ordered list of retrieved chunk IDs.
        relevant_ids:    For each query, the single correct chunk ID.

    Returns:
        Mean of ``1/rank`` (0 if the correct chunk is not retrieved).
    """
    if not relevant_ids:
        return 0.0
    reciprocals: list[float] = []
    for retrieved, correct in zip(result_ids_list, relevant_ids):
        try:
            rank = retrieved.index(correct) + 1  # 1-indexed
            reciprocals.append(1.0 / rank)
        except ValueError:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RagEvaluator:
    """Runs the full RAG evaluation loop.

    Args:
        retriever: The :class:`~app.rag.retriever.Retriever` to evaluate.
    """

    def __init__(self, retriever: "Retriever") -> None:
        self._retriever = retriever

    def evaluate(
        self,
        chunks: "list[Chunk]",
        n_queries: int = 20,
        k: int = 5,
    ) -> EvalReport:
        """Run evaluation on synthetic queries derived from *chunks*.

        Args:
            chunks:    Indexed chunks (used both to generate queries and as the
                       ground-truth pool the retriever searches through).
            n_queries: Maximum number of synthetic queries.
            k:         Retrieval depth for Recall@k.

        Returns:
            :class:`EvalReport` with Recall@k, MRR, and per-query details.
        """
        if not chunks:
            logger.warning("No chunks provided to RagEvaluator.evaluate(); returning zeros.")
            return EvalReport(recall_at_k=0.0, mrr=0.0, k=k, n_queries=0, n_hits=0)

        synthetic = generate_synthetic_queries(chunks, n=n_queries)
        if not synthetic:
            logger.warning("Could not generate synthetic queries (chunks too short?).")
            return EvalReport(recall_at_k=0.0, mrr=0.0, k=k, n_queries=0, n_hits=0)

        t0 = time.perf_counter()
        result_ids_list: list[list[str]] = []
        relevant_ids: list[str] = []
        per_query: list[dict] = []

        for sq in synthetic:
            try:
                results = self._retriever.retrieve(sq.query, k=k)
                retrieved_ids = [r.chunk_id for r in results]
            except Exception as exc:
                logger.warning("Retrieval failed for query '%s…': %s", sq.query[:40], exc)
                retrieved_ids = []

            result_ids_list.append(retrieved_ids)
            relevant_ids.append(sq.relevant_chunk_id)

            hit = sq.relevant_chunk_id in retrieved_ids[:k]
            try:
                rank = retrieved_ids.index(sq.relevant_chunk_id) + 1
            except ValueError:
                rank = 0

            per_query.append(
                {
                    "query": sq.query[:80],
                    "relevant_chunk_id": sq.relevant_chunk_id,
                    "hit": hit,
                    "rank": rank,
                    "top1_chunk_id": retrieved_ids[0] if retrieved_ids else "",
                }
            )

        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        rk = recall_at_k(result_ids_list, relevant_ids, k)
        mrr = mean_reciprocal_rank(result_ids_list, relevant_ids)
        n_hits = sum(1 for pq in per_query if pq["hit"])

        logger.info(
            "Eval complete: Recall@%d=%.4f  MRR=%.4f  (%d/%d hits)  %.1f ms",
            k,
            rk,
            mrr,
            n_hits,
            len(synthetic),
            latency_ms,
        )
        return EvalReport(
            recall_at_k=rk,
            mrr=mrr,
            k=k,
            n_queries=len(synthetic),
            n_hits=n_hits,
            latency_ms=latency_ms,
            per_query=per_query,
        )
