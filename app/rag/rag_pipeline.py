"""RAG pipeline: retrieval + local answer synthesis.

Architecture
------------
``RAGPipeline`` coordinates two pluggable components:

1. :class:`~app.rag.retriever.Retriever` — fetches the *k* most relevant
   chunks for a query from the vector store.
2. :class:`BaseAnswerGenerator` — synthesises a natural-language answer from
   the retrieved chunks.

The default generator (:class:`LocalAnswerGenerator`) is **extractive and
offline** — it ranks sentences from retrieved chunks by keyword overlap with
the query and stitches together the best ones.  No LLM, no API key, no
network connection is required.

Extending with a real LLM::

    class OpenAIAnswerGenerator(BaseAnswerGenerator):
        def generate(self, query, chunks):
            context = "\\n\\n".join(c.text for c in chunks)
            return openai.chat(query, context)   # pseudo-code

    pipeline = RAGPipeline(retriever, OpenAIAnswerGenerator())

Typical usage::

    from app.rag.rag_pipeline import RAGPipeline, LocalAnswerGenerator

    pipeline = RAGPipeline(retriever)
    result = pipeline.query("What is the attention mechanism?")
    print(result.answer)
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.utils.logger import get_logger
from app.vector_store.faiss_store import SearchResult

if TYPE_CHECKING:
    from app.rag.retriever import Retriever

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A single retrieved source chunk surfaced in an API response.

    Attributes:
        chunk_id:  Stable identifier of the chunk.
        score:     Cosine similarity score (higher = more relevant).
        text:      Text excerpt (may be truncated for display).
        metadata:  Arbitrary metadata from the vector store entry.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Complete output of a single RAG query.

    Attributes:
        query:      The original query string.
        answer:     Synthesised answer text.
        sources:    Retrieved chunks used to build the answer.
        latency_ms: End-to-end wall-clock time in milliseconds.
    """

    query: str
    answer: str
    sources: list[SourceInfo] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, description="Wall-clock time in ms.")


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


class BaseAnswerGenerator(ABC):
    """Abstract base class for answer generators.

    All implementations receive the original query and a list of
    :class:`~app.vector_store.faiss_store.SearchResult` objects and must
    return a single string answer.
    """

    @abstractmethod
    def generate(self, query: str, chunks: list[SearchResult]) -> str:
        """Synthesise an answer from *query* and *chunks*.

        Args:
            query:  The user's question.
            chunks: Retrieved chunks, ordered by relevance (best first).

        Returns:
            Natural-language answer string.
        """


class LocalAnswerGenerator(BaseAnswerGenerator):
    """Extractive answer generator — zero dependencies, works fully offline.

    Strategy
    --------
    1. Collect all sentences from the retrieved chunks (up to the first
       *top_chunks* chunks).
    2. Score each sentence by:
       - keyword overlap with the query (filtered stop-words), and
       - a rank-decay weight (chunks ranked higher get a higher weight).
    3. Return the *max_sentences* highest-scoring unique sentences joined by
       a space.

    This is not a true NLG system; it is **extractive** — every sentence in
    the answer comes verbatim from the source text.

    Args:
        max_sentences: Maximum number of sentences in the returned answer.
        min_sentence_len: Discard sentences shorter than this (in chars).
        top_chunks: Consider sentences from only the first *top_chunks* results.
    """

    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "this", "that", "these", "those",
        "it", "its", "and", "or", "but", "not", "so", "as", "if", "then",
        "than", "when", "where", "who", "which", "what", "how", "we", "our",
        "their", "they", "he", "she", "his", "her", "i", "you", "your",
    })

    def __init__(
        self,
        max_sentences: int = 5,
        min_sentence_len: int = 30,
        top_chunks: int = 5,
    ) -> None:
        self._max_sentences = max_sentences
        self._min_len = min_sentence_len
        self._top_chunks = top_chunks

    def generate(self, query: str, chunks: list[SearchResult]) -> str:
        """Build an extractive answer from *chunks*.

        Args:
            query:  The user's question.
            chunks: Retrieved results (sorted best-first).

        Returns:
            Concatenated best-matching sentences, or a fallback excerpt.
        """
        if not chunks:
            return "No relevant information found for the given query."

        # ── Extract query keywords ─────────────────────────────────────────
        query_words: set[str] = {
            w.lower().rstrip(".,;:?!\"'")
            for w in query.split()
            if w.lower().rstrip(".,;:?!\"'") not in self._STOP_WORDS
            and len(w) > 2
        }

        # ── Score sentences from top chunks ───────────────────────────────
        candidates: list[tuple[float, str]] = []
        for rank, chunk in enumerate(chunks[: self._top_chunks]):
            rank_weight = 1.0 / (rank + 1)
            raw_sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sentence in raw_sentences:
                sentence = sentence.strip()
                if len(sentence) < self._min_len:
                    continue
                sent_words = {w.lower().rstrip(".,;:?!\"'") for w in sentence.split()}
                overlap = len(query_words & sent_words) if query_words else 0
                score = rank_weight * (1.0 + overlap)
                candidates.append((score, sentence))

        if not candidates:
            return chunks[0].text[:500]

        # ── Select top unique sentences ────────────────────────────────────
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        selected: list[str] = []
        for _, sentence in candidates:
            dedup_key = sentence.lower()[:80]
            if dedup_key not in seen:
                selected.append(sentence)
                seen.add(dedup_key)
            if len(selected) >= self._max_sentences:
                break

        return " ".join(selected) if selected else chunks[0].text[:500]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Orchestrates retrieval + answer generation.

    Args:
        retriever:         Configured :class:`~app.rag.retriever.Retriever`.
        answer_generator:  Generator to synthesise the answer.  Defaults to
                           :class:`LocalAnswerGenerator` (offline, extractive).
    """

    def __init__(
        self,
        retriever: "Retriever",
        answer_generator: BaseAnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = answer_generator or LocalAnswerGenerator()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def retriever(self) -> "Retriever":
        """Underlying retriever instance."""
        return self._retriever

    @property
    def answer_generator(self) -> BaseAnswerGenerator:
        """Underlying answer generator."""
        return self._generator

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> RAGResult:
        """Run a full RAG query.

        Args:
            query: The question or search string.
            top_k: Number of chunks to retrieve.

        Returns:
            :class:`RAGResult` containing the answer and sources.

        Raises:
            ValueError: If *query* is empty or *top_k* < 1.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        t0 = time.perf_counter()

        chunks = self._retriever.retrieve(query, k=top_k)
        answer = self._generator.generate(query, chunks)

        latency_ms = round((time.perf_counter() - t0) * 1_000, 2)

        sources = [
            SourceInfo(
                chunk_id=c.chunk_id,
                score=round(c.score, 4),
                text=c.text[:400],
                metadata=c.metadata,
            )
            for c in chunks
        ]

        logger.info(
            "RAG query '%s...' -> %d sources, %.1f ms",
            query[:60],
            len(sources),
            latency_ms,
        )
        return RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )
