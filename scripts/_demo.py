"""Live pipeline demo — all milestones, no model downloads needed."""
from __future__ import annotations

from app.embeddings.embedding_model import FakeEmbeddingModel
from app.vector_store.faiss_store import SklearnVectorStore
from app.rag.retriever import Retriever
from app.rag.rag_pipeline import RAGPipeline, LocalAnswerGenerator
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.insight_agent import InsightAgent
from app.agents.citation_agent import CitationAgent
from app.evaluation.rag_eval import RagEvaluator
from app.knowledge_graph.graph_builder import GraphBuilder

SEP = "=" * 64

print(SEP)
print("  AI Research Copilot — Live Pipeline Demo")
print("  (FakeEmbeddingModel: no downloads, fully offline)")
print(SEP)

# ── 1. Build in-memory corpus ────────────────────────────────────────────
corpus = [
    "We propose a novel transformer-based architecture called Attention-Net for machine translation tasks.",
    "The attention mechanism allows the encoder-decoder model to focus on relevant tokens during decoding.",
    "Our experiments on WMT14 English-German benchmark achieve a BLEU score of 28.4, surpassing all baselines.",
    "The training dataset consists of 4.5 million sentence pairs; we also evaluate on CIFAR-10 for image tasks.",
    "Key limitations include high computational cost, large memory requirements, and slow inference on CPU.",
    "Future work will explore low-resource settings and extend BERT and GPT-style pre-training to our architecture.",
    "We compare against LSTM and BiLSTM baselines; Attention-Net outperforms them by 3.2 BLEU points on SQuAD.",
    "Evaluation uses precision, recall, F1 score, and exact-match (EM) metrics across all benchmark datasets.",
]
metadatas = [
    {
        "chunk_id": f"c{i}",
        "document_id": "doc-demo",
        "file_name": "demo_paper.pdf",
        "title": "Attention-Net: A Novel Approach",
        "page_num": (i // 2) + 1,
    }
    for i in range(len(corpus))
]

print("\n[1/6]  Building in-memory vector store...")
model = FakeEmbeddingModel(dim=32)
store = SklearnVectorStore(model)
store.add(corpus, metadatas)
print(f"       Indexed {store.count()} chunks  |  dim={model.dim}  |  backend=SklearnVectorStore")

# ── 2. RAG Query ─────────────────────────────────────────────────────────
print("\n[2/6]  RAG Query — main contribution")
retriever = Retriever(model, store)
pipeline = RAGPipeline(retriever, LocalAnswerGenerator())
result = pipeline.query("What is the main contribution of this paper?", top_k=3)
print(f"       Answer  : {result.answer[:220]}")
print(f"       Sources : {len(result.sources)} chunks  (top score={result.sources[0].score:.4f})")
print(f"       Latency : {result.latency_ms:.2f} ms")

print("\n[2b/6] RAG Query — limitations")
result2 = pipeline.query("What are the limitations of this approach?", top_k=3)
print(f"       Answer  : {result2.answer[:220]}")
print(f"       Latency : {result2.latency_ms:.2f} ms")

# ── 3. Summarizer Agent ──────────────────────────────────────────────────
print("\n[3/6]  SummarizerAgent — attention mechanism")
summarizer = SummarizerAgent(pipeline, top_k=2)
summary = summarizer.run(topic="attention mechanism")
print(f"       Summary : {summary.summary[:220]}")
print(f"       Sources : {len(summary.sources)}  |  Latency: {summary.latency_ms:.2f} ms")

# ── 4. InsightAgent ──────────────────────────────────────────────────────
print("\n[4/6]  InsightAgent — structured insights")
insight = InsightAgent(pipeline, top_k=2)
insights = insight.run()
print(f"       Contributions : {insights.contributions[:130]}")
print(f"       Limitations   : {insights.limitations[:130]}")
print(f"       Research gaps : {insights.research_gaps[:130]}")
print(f"       Latency       : {insights.latency_ms:.2f} ms")

# ── 4b. CitationAgent ────────────────────────────────────────────────────
print("\n[4b/6] CitationAgent — BLEU score datasets")
citation_agent = CitationAgent(pipeline, top_k=5)
citations = citation_agent.run(query="BLEU score benchmark dataset")
print(f"       Query     : {citations.query}")
print(f"       Citations : {len(citations.citations)} documents")
for c in citations.citations[:3]:
    print(f"         - {c.title or c.file_name}  (chunks={c.chunk_count}, score={c.relevance:.4f})")

# ── 5. Evaluation ────────────────────────────────────────────────────────
print("\n[5/6]  RagEvaluator — Recall@5 and MRR")

class _MinChunk:
    def __init__(self, cid: str, text: str) -> None:
        self.id = cid
        self.text = text
        self.document_id = "doc-demo"

fake_chunks = [_MinChunk(f"c{i}", t) for i, t in enumerate(corpus)]
evaluator = RagEvaluator(retriever)
report = evaluator.evaluate(fake_chunks, n_queries=8, k=5)
print(report.summary())

# ── 6. Knowledge Graph ───────────────────────────────────────────────────
print("\n[6/6]  GraphBuilder — entity extraction")

class _Chunk:
    def __init__(self, text: str, doc_id: str = "doc-demo") -> None:
        self.text = text
        self.document_id = doc_id

builder = GraphBuilder()
graph = builder.build([_Chunk(t) for t in corpus])
print(f"       {graph.summary()}")
if graph.nodes:
    print("       Top 8 entities:")
    for n in graph.nodes[:8]:
        print(f"         [{n.entity_type:8s}]  {n.name:<30s}  mentions={n.count}")
if graph.edges:
    print(f"       Top 3 co-occurrences:")
    for e in graph.edges[:3]:
        src = next((n.name for n in graph.nodes if n.id == e.source_id), e.source_id[:8])
        tgt = next((n.name for n in graph.nodes if n.id == e.target_id), e.target_id[:8])
        print(f"         {src}  <--co_occurs({e.weight}x)-->  {tgt}")

print(f"\n{SEP}")
print("  Demo complete. All systems operational.")
print(SEP)
