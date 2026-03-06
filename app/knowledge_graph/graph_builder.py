"""Knowledge graph builder: entity extraction and persistence.

Overview
--------
The builder scans text chunks with regex patterns to extract four categories
of research entities:

- **Model**:   Neural network architectures and systems (BERT, GPT, etc.)
- **Dataset**: Benchmark datasets used for evaluation (ImageNet, SQuAD, etc.)
- **Task**:    Research tasks (machine translation, image classification, etc.)
- **Metric**:  Evaluation metrics (BLEU, F1, accuracy, etc.)

Graph representation
--------------------
Nodes: ``{id, name, entity_type, count}``
Edges: ``{source_id, target_id, relation, weight}`` — created when two
       entities co-occur in the same chunk.

Persistence
-----------
- If ``neo4j_uri`` is configured in settings, the graph is stored in Neo4j.
- Otherwise it is written as ``data/graph/graph.json``.

Typical usage::

    from app.knowledge_graph.graph_builder import GraphBuilder

    builder = GraphBuilder()
    graph = builder.build(chunks)
    builder.save(graph)
    data = builder.load()
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Entity patterns
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, list[str]] = {
    "Model": [
        r"\b(BERT|RoBERTa|XLNet|GPT[-\s]?\d*|T5|ALBERT|DistilBERT|ELECTRA|DeBERTa|"
        r"LLaMA|LLaMA[-\s]?\d*|PaLM|Gemini|Claude|Mistral|Phi[-\s]?\d*)\b",
        r"\b(ResNet[-\s]?\d*|VGG[-\s]?\d*|EfficientNet|DenseNet|Inception|"
        r"MobileNet|YOLO[-\s]?\w*|CLIP|DALL-E|Stable Diffusion|ViT|Swin)\b",
        r"\b(Transformer|LSTM|BiLSTM|GRU|CNN|RNN|GAN|VAE|Diffusion Model|"
        r"U-Net|Attention Network|Graph Neural Network|GNN|GCN|GAT)\b",
    ],
    "Dataset": [
        r"\b(ImageNet|COCO|MS[-\s]?COCO|SQuAD[-\s]?\d*\.?\d*|GLUE|SuperGLUE|"
        r"MNIST|CIFAR[-\s]?\d+|WMT\d+|WikiText[-\s]?\d*|Penn Treebank)\b",
        r"\b(VQA|GQA|NLVR|MultiNLI|SNLI|CoLA|SST[-\s]?\d*|MRPC|QQP|STS[-\s]?B)\b",
        r"\b(\w[\w\s-]{2,30}(?:dataset|corpus|benchmark|collection|testbed|split))\b",
    ],
    "Task": [
        r"\b(image classification|object detection|image segmentation|"
        r"semantic segmentation|instance segmentation|depth estimation)\b",
        r"\b(machine translation|text classification|named entity recognition|"
        r"relation extraction|sentiment analysis|question answering|"
        r"reading comprehension|natural language inference)\b",
        r"\b(language modeling|text generation|text summarization|"
        r"dialogue generation|code generation|visual question answering)\b",
        r"\b(speech recognition|speaker identification|audio classification)\b",
    ],
    "Metric": [
        r"\b(BLEU[-\s]?\d*|ROUGE[-\s]?[LN12]*|METEOR|CIDEr|SPICE|BERTScore)\b",
        r"\b(accuracy|F1[-\s]?score|precision|recall|mAP|AP@\d+|AUC|AUROC|"
        r"perplexity|exact match|EM score)\b",
        r"\b(\d+(?:\.\d+)?%?\s+(?:accuracy|F1|precision|recall|BLEU|mAP))\b",
    ],
}

# Compile all patterns once
_COMPILED: dict[str, list[re.Pattern[str]]] = {
    entity_type: [re.compile(p, re.IGNORECASE) for p in patterns]
    for entity_type, patterns in _PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Graph data models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A single entity node in the knowledge graph.

    Attributes:
        id:          Deterministic SHA-256-based 16-char hex ID.
        name:        Canonical entity name (normalised).
        entity_type: One of Model, Dataset, Task, Metric.
        count:       Number of times this entity was mentioned.
    """

    id: str
    name: str
    entity_type: str
    count: int = 1


class GraphEdge(BaseModel):
    """A co-occurrence edge between two entities.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        relation:  Edge label (always ``"co_occurs_with"``).
        weight:    Number of chunks where both entities co-occur.
    """

    source_id: str
    target_id: str
    relation: str = "co_occurs_with"
    weight: int = 1


class GraphData(BaseModel):
    """Complete knowledge graph.

    Attributes:
        nodes:        All extracted entity nodes.
        edges:        Co-occurrence edges.
        source_docs:  Document IDs that contributed to this graph.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    source_docs: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """Return a short human-readable summary."""
        type_counts = {}
        for node in self.nodes:
            type_counts[node.entity_type] = type_counts.get(node.entity_type, 0) + 1
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        return (
            f"Nodes: {len(self.nodes)} ({parts}) | "
            f"Edges: {len(self.edges)} | "
            f"Docs: {len(self.source_docs)}"
        )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Extracts entities from text chunks and builds a knowledge graph.

    Args:
        graph_dir: Directory for JSON persistence.  Defaults to
                   ``data/graph`` (resolved relative to cwd).
    """

    def __init__(self, graph_dir: str | None = None) -> None:
        settings = get_settings()
        self._graph_dir = Path(graph_dir) if graph_dir else (
            settings.data_dir / "graph"
        )
        self._graph_file = self._graph_dir / "graph.json"

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, chunks: list[Any]) -> GraphData:
        """Extract entities from *chunks* and build a graph.

        Args:
            chunks: List of objects with a ``.text`` and optional ``.document_id``
                    attribute (e.g. :class:`~app.ingestion.models.Chunk`).

        Returns:
            :class:`GraphData` with nodes and co-occurrence edges.
        """
        # node_id → GraphNode
        node_map: dict[str, GraphNode] = {}
        # (node_id_a, node_id_b) → edge weight
        edge_map: dict[tuple[str, str], int] = {}
        # Collect unique document IDs
        doc_ids: set[str] = set()

        for chunk in chunks:
            text = getattr(chunk, "text", "") or str(chunk)
            doc_id = getattr(chunk, "document_id", "")
            if doc_id:
                doc_ids.add(doc_id)

            chunk_entities = self._extract_entities(text)

            # Update node counts
            chunk_node_ids: list[str] = []
            for entity_type, names in chunk_entities.items():
                for name in names:
                    node_id = _make_node_id(name, entity_type)
                    if node_id not in node_map:
                        node_map[node_id] = GraphNode(
                            id=node_id,
                            name=_normalise(name),
                            entity_type=entity_type,
                        )
                    else:
                        node_map[node_id].count += 1
                    if node_id not in chunk_node_ids:
                        chunk_node_ids.append(node_id)

            # Add co-occurrence edges for every pair in this chunk
            for i in range(len(chunk_node_ids)):
                for j in range(i + 1, len(chunk_node_ids)):
                    a, b = sorted([chunk_node_ids[i], chunk_node_ids[j]])
                    edge_map[(a, b)] = edge_map.get((a, b), 0) + 1

        nodes = sorted(node_map.values(), key=lambda n: -n.count)
        edges = [
            GraphEdge(source_id=a, target_id=b, weight=w)
            for (a, b), w in sorted(edge_map.items(), key=lambda x: -x[1])
        ]

        graph = GraphData(
            nodes=nodes,
            edges=edges,
            source_docs=sorted(doc_ids),
        )
        logger.info("Graph built: %s", graph.summary())
        return graph

    def save(self, graph: GraphData) -> None:
        """Persist *graph* to the configured backend.

        If Neo4j URI is configured, stores to Neo4j; otherwise writes JSON.

        Args:
            graph: The :class:`GraphData` to persist.
        """
        settings = get_settings()
        if settings.neo4j_uri:
            self._save_neo4j(graph, settings)
        else:
            self._save_json(graph)

    def load(self) -> GraphData:
        """Load the persisted graph from JSON.

        Returns:
            :class:`GraphData` loaded from disk.

        Raises:
            FileNotFoundError: If no graph has been saved yet.
        """
        if not self._graph_file.exists():
            raise FileNotFoundError(
                f"No graph found at '{self._graph_file}'. "
                "Run scripts/build_graph.py first."
            )
        with open(self._graph_file, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Graph loaded from '%s'", self._graph_file)
        return GraphData(**data)

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Run regex patterns over *text* and return matched entities.

        Args:
            text: Input text to scan.

        Returns:
            Dict mapping entity_type → list of unique matched strings.
        """
        found: dict[str, list[str]] = {}
        for entity_type, patterns in _COMPILED.items():
            seen: set[str] = set()
            for pattern in patterns:
                for match in pattern.finditer(text):
                    name = match.group(0).strip()
                    key = name.lower()
                    if key not in seen and len(name) > 1:
                        seen.add(key)
                        found.setdefault(entity_type, []).append(name)
        return found

    def _save_json(self, graph: GraphData) -> None:
        """Write *graph* to ``graph.json``."""
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        with open(self._graph_file, "w", encoding="utf-8") as fh:
            json.dump(graph.model_dump(), fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Graph saved to '%s' (%s)", self._graph_file, graph.summary())

    def _save_neo4j(self, graph: GraphData, settings: Any) -> None:
        """Write *graph* to a Neo4j instance.

        Args:
            graph:    Graph data to write.
            settings: Application settings with ``neo4j_*`` fields.
        """
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "neo4j Python driver required. Install with: pip install neo4j"
            ) from exc

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user or "neo4j", settings.neo4j_password or ""),
        )
        try:
            with driver.session() as session:
                for node in graph.nodes:
                    session.run(
                        "MERGE (n:Entity {id: $id}) "
                        "SET n.name = $name, n.type = $type, n.count = $count",
                        id=node.id,
                        name=node.name,
                        type=node.entity_type,
                        count=node.count,
                    )
                for edge in graph.edges:
                    session.run(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[r:CO_OCCURS_WITH]->(b) "
                        "SET r.weight = $weight",
                        src=edge.source_id,
                        tgt=edge.target_id,
                        weight=edge.weight,
                    )
            logger.info("Graph written to Neo4j (%s)", settings.neo4j_uri)
        finally:
            driver.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_id(name: str, entity_type: str) -> str:
    """Return a deterministic 16-char hex node ID."""
    raw = f"{entity_type}:{name.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalise(name: str) -> str:
    """Normalise whitespace and casing in an entity name."""
    return " ".join(name.split())
