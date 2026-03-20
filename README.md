# AI Research Copilot

> A **production-grade, fully open-source** system for ingesting academic research papers (PDF), building a searchable knowledge base, and performing intelligent Q&A, summarisation, insight extraction, and knowledge graph construction — **entirely offline, no paid APIs required.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

| Milestone | Feature |
|-----------|---------|
| M1 | FastAPI service · `/health` & `/version` · Pydantic settings · Structured JSON/text logging | 
| M2 | PDF ingestion (PyMuPDF) · Section detection · Character chunker with overlap · deterministic IDs | 
| M3 | Sentence-Transformers embeddings (lazy load) · FAISS + sklearn fallback · persist/load index | 
| M4 | RAG pipeline · extractive answer synthesis · `POST /ask` · `POST /upload-paper` | 
| M5 | Summariser, Insight, Citation agents · `/summarize` · `/research-insights` · `/related-work` | 
| M6 | Evaluation harness · Recall@k · MRR · synthetic query generation · CLI | 
| M7 | Knowledge graph (regex entity extraction) · Neo4j or JSON backend · `GET /graph` | 
| M8 | Docker + docker-compose (API + optional Neo4j) · polished README · full test suite |

**Key design principles:**
- **Fully offline** — uses `sentence-transformers/all-MiniLM-L6-v2` + sklearn; zero paid API calls.
- **Pluggable LLM** — `BaseAnswerGenerator` ABC lets you swap in OpenAI/Anthropic by subclassing one method.
- **Deterministic** — SHA-256 based IDs; seeded randomness where applicable.
- **Clean architecture** — strict separation: ingestion → embeddings → vector store → RAG → agents.
- **Windows-compatible** — tested on PowerShell / Python 3.11+.
- **Testable offline** — `FakeEmbeddingModel` enables the full test suite without any downloads.

## Project Structure

```
ai-research-copilot/
├── app/
│   ├── api/
│   │   ├── main.py            # FastAPI factory + lifespan
│   │   ├── deps.py            # Dependency injection
│   │   └── routers/
│   │       ├── rag.py         # /upload-paper, /ask
│   │       ├── agents.py      # /summarize, /research-insights, /related-work
│   │       └── graph.py       # /graph, /graph/build
│   ├── config/
│   │   └── settings.py        # Pydantic settings (reads .env)
│   ├── utils/
│   │   └── logger.py          # JSON / text logging
│   ├── ingestion/
│   │   ├── models.py          # Document, Page, Section, Chunk Pydantic models
│   │   ├── pdf_parser.py      # PyMuPDF parser
│   │   └── chunking.py        # Character-based chunker with overlap
│   ├── embeddings/
│   │   └── embedding_model.py # BaseEmbeddingModel, EmbeddingModel, FakeEmbeddingModel
│   ├── vector_store/
│   │   └── faiss_store.py     # VectorStore ABC, FaissVectorStore, SklearnVectorStore
│   ├── rag/
│   │   ├── retriever.py       # Retriever (embed query + search store)
│   │   └── rag_pipeline.py    # RAGPipeline, BaseAnswerGenerator, LocalAnswerGenerator
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── summarizer_agent.py
│   │   ├── insight_agent.py
│   │   └── citation_agent.py
│   ├── evaluation/
│   │   └── rag_eval.py        # RagEvaluator, Recall@k, MRR
│   └── knowledge_graph/
│       └── graph_builder.py   # Entity extraction + Neo4j / JSON persistence
├── scripts/
│   ├── ingest_pdf.py          # Parse + chunk a PDF
│   ├── build_index.py         # Build + persist vector index
│   ├── ask_question.py        # Query the index from CLI
│   ├── evaluate_rag.py        # Run evaluation metrics
│   └── build_graph.py         # Build + save knowledge graph
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pdf_parser.py
│   ├── test_chunking.py
│   ├── test_embedding_model.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   └── test_agents.py
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── conftest.py
├── pytest.ini
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11+
- (Optional) CUDA-capable GPU for faster embeddings

### 1 — Clone and create a virtual environment

```bash
# Bash
git clone https://github.com/your-org/ai-research-copilot.git
cd ai-research-copilot
python -m venv .venv
source .venv/bin/activate

# PowerShell
git clone https://github.com/your-org/ai-research-copilot.git
cd ai-research-copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **First run:** The embedding model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded
> automatically on the first call to `build_index.py` or the `/upload-paper` endpoint.
> Subsequent runs use the cached copy in `models/`.

### 3 — Copy and edit the environment file

```bash
# Bash
cp .env.example .env

# PowerShell
Copy-Item .env.example .env
```

---

## Configuration

All settings are read from environment variables or the `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `AI Research Copilot` | Application name |
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | `text` | `text` (dev) or `json` (production / containers) |
| `DATA_DIR` | `data` | Root directory for all data files |
| `INDEX_DIR` | `data/index` | Vector store persistence directory |
| `MODELS_DIR` | `models` | Sentence-transformers model cache |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model id |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlapping characters between chunks |
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss` (requires `faiss-cpu`) or `sklearn` |
| `RETRIEVAL_TOP_K` | `5` | Default number of retrieved chunks |
| `LLM_PROVIDER` | `local` | `local` / `openai` / `anthropic` |
| `NEO4J_URI` | *(None)* | Neo4j bolt URI — omit to use JSON graph backend |

---

## Running the API

```bash
# Bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# PowerShell
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

---

## CLI Usage

### Ingest a PDF (M2)

```bash
# Bash
python scripts/ingest_pdf.py --pdf data/raw/paper.pdf
python scripts/ingest_pdf.py --pdf paper.pdf --chunk-size 256 --save-json

# PowerShell
python scripts\ingest_pdf.py --pdf data\raw\paper.pdf
```

Expected output:
```
──────────────────────────────────────────────
  Ingestion Summary
──────────────────────────────────────────────
  File      : paper.pdf
  Pages     : 15      Chars  : 62,441
  Sections  : 8       Chunks : 134
  Chunk chars : avg=476  min=128  max=512
──────────────────────────────────────────────
```

---

### Build a Vector Index (M3)

```bash
# Bash
python scripts/build_index.py --pdf data/raw/paper.pdf --out data/index

# PowerShell
python scripts\build_index.py --pdf data\raw\paper.pdf --out data\index
```

Expected output:
```
────────────────────────────────────────────────────────
  Build Index Summary
────────────────────────────────────────────────────────
  File       : paper.pdf
  Chunks     : 134
  Vector dim : 384
  Backend    : sklearn
  Index path : data/index
────────────────────────────────────────────────────────
```

Index files:

| File | Description |
|------|-------------|
| `data/index/vectors.npy` | Float32 embedding matrix `(n_chunks, 384)` |
| `data/index/metadata.json` | chunk_id, text, page_num, source per chunk |

---

### Ask a Question (M4)

```bash
# Bash
python scripts/ask_question.py \
    --index data/index \
    --query "What is the main contribution of this paper?" \
    --show-sources

# PowerShell
python scripts\ask_question.py `
    --index data\index `
    --query "What is the main contribution of this paper?" `
    --show-sources
```

Expected output:
```
────────────────────────────────────────────────────────────────
  Query: What is the main contribution of this paper?
────────────────────────────────────────────────────────────────

We propose a novel transformer architecture based on multi-head
self-attention. Our model achieves state-of-the-art results on
WMT14 English-German with a BLEU score of 28.4.

  [1] score=0.9821  file=paper.pdf  page=1
      We propose a novel transformer architecture…

  Latency: 42.3 ms
────────────────────────────────────────────────────────────────
```

---

### Run Evaluation (M6)

```bash
# Bash
python scripts/evaluate_rag.py --index data/index --k 5 --n-queries 20

# PowerShell
python scripts\evaluate_rag.py --index data\index --k 5 --n-queries 20
```

Expected output:
```
────────────────────────────────────────────────
  RAG Evaluation Report
────────────────────────────────────────────────
  Queries evaluated : 20
  Retrieval depth k : 5
  Recall@5          : 0.8500
  MRR               : 0.7200
  Hits              : 17/20
  Latency           : 823.4 ms
────────────────────────────────────────────────
```

---

### Build Knowledge Graph (M7)

```bash
# Bash
python scripts/build_graph.py --index data/index

# PowerShell
python scripts\build_graph.py --index data\index
```

Expected output:
```
──────────────────────────────────────────────────
  Knowledge Graph Summary
──────────────────────────────────────────────────
  Nodes: 24 (Dataset: 4, Metric: 6, Model: 8, Task: 6) | Edges: 31 | Docs: 2
──────────────────────────────────────────────────
```

The graph is saved to `data/graph/graph.json` by default.

---

## API Reference

### Meta

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/version` | App version info |

### RAG (M4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload-paper` | Upload + index a PDF |
| `POST` | `/api/v1/ask` | Ask a question |

**Upload a paper:**
```bash
curl -X POST http://localhost:8000/api/v1/upload-paper \
  -F "file=@paper.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What datasets were used?", "top_k": 5}'
```

Response:
```json
{
  "query": "What datasets were used?",
  "answer": "The experiments used WMT14 English-German dataset...",
  "sources": [
    {"chunk_id": "abc123", "score": 0.94, "text": "...", "metadata": {...}}
  ],
  "latency_ms": 38.2
}
```

### Agents (M5)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/summarize` | Summarise indexed papers |
| `POST` | `/api/v1/research-insights` | Extract contributions / limitations / gaps |
| `POST` | `/api/v1/related-work` | Find related work |

```bash
# Summarise
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"topic": "attention mechanism"}'

# Research insights
curl -X POST http://localhost:8000/api/v1/research-insights \
  -H "Content-Type: application/json" \
  -d '{}'

# Related work
curl -X POST http://localhost:8000/api/v1/related-work \
  -H "Content-Type: application/json" \
  -d '{"query": "self-supervised learning for NLP"}'
```

### Knowledge Graph (M7)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/graph` | Get the persisted knowledge graph |
| `POST` | `/api/v1/graph/build` | Build graph from current index |

```bash
# Get graph
curl http://localhost:8000/api/v1/graph

# Build graph
curl -X POST http://localhost:8000/api/v1/graph/build
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific milestone tests
pytest tests/test_api.py -v
pytest tests/test_rag_pipeline.py -v
pytest tests/test_agents.py -v
pytest tests/test_vector_store.py tests/test_embedding_model.py -v

# Run with coverage (requires pytest-cov)
pytest --cov=app --cov-report=term-missing
```

Expected summary:
```
tests/test_api.py             PASSED (13 M1 + 20 M4-M7 tests)
tests/test_pdf_parser.py      PASSED (M2 — requires PyMuPDF)
tests/test_chunking.py        PASSED (M2)
tests/test_embedding_model.py PASSED (M3 — offline via FakeEmbeddingModel)
tests/test_vector_store.py    PASSED (M3)
tests/test_rag_pipeline.py    PASSED (M4)
tests/test_agents.py          PASSED (M5)
========================= 100+ passed =========================
```

> Tests tagged `sentence_transformers` and `faiss` are automatically skipped
> when those packages are not installed.

---

## Docker Setup

### Quick start (API only)

```bash
# Build and start
docker compose up --build

# Or in detached mode
docker compose up --build -d

# View logs
docker compose logs -f api
```

The API will be available at http://localhost:8000.

### With Neo4j knowledge graph backend

```bash
docker compose --profile graph up --build
```

Services:
- API: http://localhost:8000
- Neo4j browser: http://localhost:7474 (user: `neo4j`, password: `password`)
- Neo4j bolt: `bolt://localhost:7687`

### Environment customisation

Pass environment variables via a `.env` file or directly:

```bash
LOG_FORMAT=json CHUNK_SIZE=256 docker compose up
```

---

## Evaluation

The evaluation harness generates synthetic queries by extracting the first
sentence of each sampled chunk, then checks whether retrieval can find the
originating chunk.

**Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Recall@k | hits / total | Fraction of queries where correct chunk is in top-k |
| MRR | mean(1/rank) | Higher = correct chunk tends to be ranked first |

Run via CLI (see above) or import directly:

```python
from app.evaluation import RagEvaluator
from app.rag.retriever import Retriever

evaluator = RagEvaluator(retriever)
report = evaluator.evaluate(chunks, n_queries=50, k=5)
print(report.summary())
# Recall@5: 0.82   MRR: 0.67
```
