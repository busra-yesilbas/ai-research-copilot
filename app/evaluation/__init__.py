"""Evaluation package.

Public API::

    from app.evaluation import RagEvaluator, EvalReport, SyntheticQuery
    from app.evaluation import recall_at_k, mean_reciprocal_rank
    from app.evaluation import generate_synthetic_queries
"""

from app.evaluation.rag_eval import (
    EvalReport,
    RagEvaluator,
    SyntheticQuery,
    generate_synthetic_queries,
    mean_reciprocal_rank,
    recall_at_k,
)

__all__ = [
    "RagEvaluator",
    "EvalReport",
    "SyntheticQuery",
    "generate_synthetic_queries",
    "recall_at_k",
    "mean_reciprocal_rank",
]
