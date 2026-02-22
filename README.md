# cog-rag-cognee

[![CI](https://github.com/vpakspace/cog-rag-cognee/actions/workflows/ci.yml/badge.svg)](https://github.com/vpakspace/cog-rag-cognee/actions/workflows/ci.yml)

Semantic memory layer with Cognee SDK — 100% local stack.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (:8506)                    │
│  [Upload] [Search & QA] [Graph Explorer] [Settings]      │
└──────────────────────┬──────────────────────────────────┘
                       │ httpx
┌──────────────────────▼──────────────────────────────────┐
│                FastAPI REST API (:8508)                   │
│  /health  /ingest  /query  /search                       │
│  /graph/stats  /graph/entities                           │
└───────┬──────────────────────┬──────────────────────────┘
        │                      │
┌───────▼───────┐     ┌───────▼────────┐
│PipelineService│     │  GraphClient   │
│(Cognee wrapper│     │(Neo4j driver)  │
│ add/cognify/  │     │ get_entities   │
│ search/reset) │     │ get_relations  │
└───┬───────────┘     │ get_stats      │
    │                 └───────┬────────┘
┌───▼─────────────────────────▼──────────────────────────┐
│              Cognee SDK (pip install cognee)             │
│  Ollama LLM + Ollama Embeddings                         │
│  Neo4j (graph) + LanceDB (vector)                       │
└───┬──────────┬───────────┬─────────────────────────────┘
    │          │           │
┌───▼───┐ ┌───▼────┐ ┌────▼─────┐
│Ollama │ │ Neo4j  │ │ LanceDB  │
│:11434 │ │:7474   │ │ (files)  │
│local  │ │Docker  │ │ embedded │
└───────┘ └────────┘ └──────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama llama3.1:8b (local) |
| Embeddings | Ollama nomic-embed-text (768d, ~100 languages) |
| Graph DB | Neo4j 5 (Docker) |
| Vector DB | LanceDB (embedded, pip install) |
| Core SDK | Cognee (ECL pipeline, dedup, ontology) |
| API | FastAPI |
| UI | Streamlit (4 tabs, EN/RU) |
| Graph Viz | PyVis (interactive, entity type filter) |

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Ollama (installed locally or via Docker)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/vpakspace/cog-rag-cognee.git
cd cog-rag-cognee

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start services
docker compose up -d

# 4. Pull Ollama models
bash scripts/pull_models.sh

# 5. Configure
cp .env.example .env
# Edit .env if needed

# 6. Run API
python run_api.py

# 7. Run UI (separate terminal)
streamlit run ui/streamlit_app.py --server.port 8506
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/ingest` | Upload + Extract + Cognify |
| POST | `/api/v1/query` | RAG: search + generate answer |
| POST | `/api/v1/search` | Search only (no generation) |
| GET | `/api/v1/graph/stats` | Live knowledge graph statistics |
| GET | `/api/v1/graph/entities` | Graph nodes + edges for visualization |

### Example: Query

```bash
curl -X POST http://localhost:8508/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Cognee?", "mode": "GRAPH_COMPLETION"}'
```

### Example: Ingest

```bash
curl -X POST http://localhost:8508/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Cognee transforms documents into AI memory."}'
```

### Example: Graph Entities

```bash
# All entities (default limit 200)
curl http://localhost:8508/api/v1/graph/entities

# Filter by type
curl "http://localhost:8508/api/v1/graph/entities?entity_types=Person,Organization&limit=50"
```

## CLI Ingestion

```bash
python scripts/ingest.py data/sample_en.txt data/sample_ru.txt
```

## Project Structure

```
cog-rag-cognee/
├── cog_rag_cognee/           # Core package
│   ├── config.py             # Pydantic Settings
│   ├── models.py             # Domain models
│   ├── service.py            # PipelineService (Cognee wrapper)
│   ├── graph_client.py       # Neo4j driver wrapper (direct Cypher)
│   ├── cognee_setup.py       # Cognee SDK configuration
│   ├── ontology.py           # OWL/RDF ontology loader
│   └── exceptions.py         # Custom exceptions
├── api/
│   ├── app.py                # FastAPI factory + lifespan
│   ├── routes.py             # REST endpoints (6)
│   └── deps.py               # Dependency injection (service + graph_client)
├── ui/
│   ├── streamlit_app.py      # 4-tab UI
│   ├── i18n.py               # EN/RU translations (~80 keys)
│   └── components/
│       └── graph_viz.py      # PyVis visualization (entity type colors)
├── scripts/
│   ├── ingest.py             # CLI ingestion
│   └── pull_models.sh        # Ollama model download
├── ontologies/
│   └── example.owl           # Example domain ontology
├── data/                     # Sample documents (EN/RU)
├── benchmark/                # Evaluation questions
├── tests/                    # 43 pytest tests, 92% coverage
├── docker-compose.yml        # Neo4j + Ollama
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Core Features

1. **ECL Pipeline + Persistent Memory** — Extract-Cognify-Load via Cognee SDK
2. **Semantic Deduplication** — exact hash + LLM fuzzy matching for entities
3. **Ontology Integration** — OWL/RDF domain grounding
4. **Graph Explorer** — interactive PyVis visualization with live Neo4j queries, entity type filter, stats dashboard

## Graph Explorer

The Graph Explorer tab provides interactive visualization of the knowledge graph built by Cognee:

- **Live data** from Neo4j via direct Cypher queries (GraphClient)
- **Entity type filter** — multiselect to show/hide Person, Organization, Location, etc.
- **Stats dashboard** — node count, edge count, entity type breakdown
- **PyVis rendering** — interactive drag-and-drop, zoom, hover tooltips
- **Color coding** — Person (red), Organization (blue), Location (green), Date (yellow), Document (purple), Chunk (gray)
- **Graceful fallback** — shows placeholder when Neo4j is unavailable

## Benchmark

10 questions (5 EN + 5 RU) in `benchmark/questions.json`. Categories: simple, relation.

## Tests

```bash
pytest tests/ -v --cov=cog_rag_cognee --cov=api   # 43 tests, 92% coverage
ruff check .                                        # Lint
```

## Configuration

All settings via environment variables or `.env` file. See `.env.example` for the full list.

Key settings:
- `LLM_MODEL` — Ollama model (default: `llama3.1:8b`)
- `EMBEDDING_MODEL` — embedding model (default: `nomic-embed-text:latest`)
- `GRAPH_DATABASE_URL` — Neo4j connection (default: `neo4j://localhost:7687`)
- `GRAPH_DATABASE_USERNAME` / `GRAPH_DATABASE_PASSWORD` — Neo4j credentials
- `VECTOR_DB_PROVIDER` — vector store (default: `lancedb`)

## Deferred Features

- BM25 keyword search (tantivy / SQLite FTS5)
- Memify graph optimization
- Iterative probing
- Docling document parser (GPU)
- Semantic cache

## License

MIT
