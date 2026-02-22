# Design: cog-rag-cognee

**Date**: 2026-02-22
**Status**: Approved
**Author**: vladspace_ubuntu24 + Claude Opus 4.6

## Goal

Новый проект на базе Cognee SDK — semantic memory layer с knowledge graph и модульными ECL pipelines. 100% локальный стек (Ollama + Neo4j + LanceDB), без cloud API keys.

## Decisions

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| Storage: vector | LanceDB (embedded) | Простой деплой, pip install, без Docker для вектора |
| Storage: graph | Neo4j 5 (Docker) | Persistent graph, Cypher, проверен в Cog-RAG |
| Storage: BM25 | Позже (tantivy/SQLite FTS5) | MVP без keyword search, Cognee graph компенсирует |
| LLM | Ollama llama3.1:8b | 100% локальный, проверен в Cog-RAG |
| Embeddings | Ollama nomic-embed-text-v2-moe (768d) | ~100 языков вкл. русский |
| Architecture | Cognee SDK как ядро | Максимум из SDK: ECL, dedup, ontology, search |
| Interface | FastAPI + Streamlit | Проверенный паттерн из Cog-RAG |
| Project name | cog-rag-cognee | ~/cog-rag-cognee/ |

## Core Features (MVP)

1. **ECL Pipeline + Persistent Memory** — Extract-Cognify-Load, ядро Cognee
2. **Semantic Deduplication** — exact hash + LLM fuzzy matching для entities
3. **Ontology Integration** — OWL/RDF domain grounding

## Deferred Features

- Memify (graph optimization) — после MVP
- BM25 keyword search (tantivy/SQLite FTS5) — после MVP
- Iterative probing — после MVP (если benchmark покажет необходимость)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI (:8506)               │
│  [Upload] [Search & QA] [Graph Explorer] [Settings]  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                FastAPI REST API (:8508)               │
│  /health  /ingest  /query  /search  /graph/stats     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              PipelineService (thin wrapper)           │
│  add() → cognify() → search() — delegates to Cognee │
└───┬──────────┬───────────┬──────────────────────────┘
    │          │           │
┌───▼───┐ ┌───▼────┐ ┌────▼─────┐
│Cognee │ │Cognee  │ │ Cognee   │
│  ECL  │ │ Dedup  │ │ Ontology │
│Pipeline│ │Engine  │ │ Layer    │
└───┬───┘ └───┬────┘ └────┬─────┘
    │         │            │
┌───▼─────────▼────────────▼──────────────────────────┐
│              Cognee SDK (pip install cognee)          │
│  Config: Ollama LLM + Ollama Embeddings              │
│          Neo4j (graph) + LanceDB (vector)            │
└───┬──────────┬───────────┬──────────────────────────┘
    │          │           │
┌───▼───┐ ┌───▼────┐ ┌────▼─────┐
│Ollama │ │ Neo4j  │ │ LanceDB  │
│:11434 │ │:7474   │ │ (files)  │
│local  │ │Docker  │ │ embedded │
└───────┘ └────────┘ └──────────┘
```

## Project Structure

```
cog-rag-cognee/
├── cog_rag_cognee/           # Core package
│   ├── config.py             # Pydantic Settings (Ollama, Neo4j, LanceDB)
│   ├── models.py             # Domain models (QAResult, SearchResult)
│   ├── service.py            # PipelineService (Cognee wrapper)
│   ├── cognee_setup.py       # Cognee SDK initialization & config
│   ├── ontology.py           # OWL/RDF ontology loader
│   ├── loader.py             # Docling document parser
│   └── exceptions.py         # Custom exceptions
├── api/
│   ├── app.py                # FastAPI factory + lifespan
│   ├── routes.py             # REST endpoints
│   └── deps.py               # DI (get_service)
├── ui/
│   ├── streamlit_app.py      # 4 tabs
│   ├── i18n.py               # EN/RU translations
│   └── components/
│       └── graph_viz.py      # PyVis visualization
├── scripts/
│   ├── ingest.py             # CLI ingestion
│   └── pull_models.sh        # Ollama model download
├── ontologies/               # OWL/RDF files
│   └── example.owl           # Example domain ontology
├── data/                     # Sample documents
├── tests/                    # pytest suite
├── docker-compose.yml        # Neo4j + Ollama
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/v1/health | Health check (Ollama, Neo4j, LanceDB) |
| POST | /api/v1/ingest | Upload + Extract + Cognify |
| POST | /api/v1/query | RAG: search → generate answer |
| POST | /api/v1/search | Search only (no generation) |
| GET | /api/v1/graph/stats | Knowledge graph statistics |

## Patterns from Cog-RAG (reuse)

- Pydantic Settings with `env_prefix` + `get_settings()` with `@lru_cache`
- FastAPI factory + `asynccontextmanager` lifespan + DI
- Docling loader (lazy init, GPU optional)
- EN/RU i18n with `get_translator(lang)`
- PyVis graph visualization
- API key auth middleware + slowapi rate limiting

## Docker Compose

```yaml
services:
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    volumes: [neo4j_data:/data]
    environment:
      NEO4J_AUTH: neo4j/password

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  neo4j_data:
  ollama_data:
```

## Benchmark

- Reuse `benchmark/questions.json` from Cog-RAG (30 questions, RU + EN)
- Compare with Cog-RAG baseline (93%)
- Modes: vector, graph, hybrid
