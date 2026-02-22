# cog-rag-cognee Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Создать semantic memory layer на базе Cognee SDK с 100% локальным стеком (Ollama + Neo4j + LanceDB), FastAPI REST API и Streamlit UI.

**Architecture:** Cognee SDK как ядро (ECL pipeline, semantic dedup, ontology). PipelineService — thin wrapper. FastAPI для REST endpoints, Streamlit для UI. Docling для парсинга документов.

**Tech Stack:** Python 3.12, Cognee SDK 0.5.x, Ollama (llama3.1:8b + nomic-embed-text-v2-moe), Neo4j 5, LanceDB, FastAPI, Streamlit, Docling, PyVis, pytest, ruff.

---

## Phase 1: Project Scaffolding

### Task 1: Initialize project and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `cog_rag_cognee/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "cog-rag-cognee"
version = "0.1.0"
description = "Semantic memory layer with Cognee SDK — 100% local stack"
requires-python = ">=3.10,<3.14"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**Step 2: Create requirements.txt**

```
cognee[ollama,neo4j]>=0.5.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
streamlit>=1.40.0
docling>=2.0.0
pyvis>=0.3.2
python-dotenv>=1.0.0
pydantic-settings>=2.6.0
httpx>=0.27.0
slowapi>=0.1.9

# Dev
pytest>=8.0.0
pytest-asyncio>=0.24.0
ruff>=0.8.0
```

**Step 3: Create .env.example**

```bash
# LLM
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_ENDPOINT=http://localhost:11434

# Embeddings
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_ENDPOINT=http://localhost:11434/api/embed
EMBEDDING_DIMENSIONS=768

# Graph DB
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=neo4j://localhost:7687
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=password

# Vector DB
VECTOR_DB_PROVIDER=lancedb

# Storage
STORAGE_ROOT_DIR=./cognee_data

# API
API_KEY=
API_HOST=0.0.0.0
API_PORT=8508

# UI
UI_PORT=8506
```

**Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.env
cognee_data/
.lancedb/
.ruff_cache/
.pytest_cache/
*.egg-info/
dist/
build/
```

**Step 5: Create cog_rag_cognee/__init__.py**

```python
"""Cog-RAG Cognee: Semantic memory layer with 100% local stack."""

__version__ = "0.1.0"
```

**Step 6: Init git and commit**

```bash
cd ~/cog-rag-cognee
git init
git add pyproject.toml requirements.txt .env.example .gitignore cog_rag_cognee/__init__.py docs/
git commit -m "feat: project scaffolding — pyproject, requirements, env template"
```

---

### Task 2: Config module (Pydantic Settings)

**Files:**
- Create: `tests/test_config.py`
- Create: `cog_rag_cognee/config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for configuration module."""
import pytest


def test_settings_defaults():
    """Settings should have sensible defaults."""
    from cog_rag_cognee.config import get_settings

    s = get_settings()
    assert s.llm_model == "llama3.1:8b"
    assert s.embedding_model == "nomic-embed-text:latest"
    assert s.embedding_dimensions == 768
    assert s.graph_db_provider == "neo4j"
    assert s.vector_db_provider == "lancedb"
    assert s.api_port == 8508
    assert s.ui_port == 8506


def test_settings_singleton():
    """get_settings() should return the same instance."""
    from cog_rag_cognee.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_ollama_base_url():
    """Ollama base URL should be derived from endpoint."""
    from cog_rag_cognee.config import get_settings

    s = get_settings()
    assert s.ollama_base_url == "http://localhost:11434"
```

**Step 2: Run test to verify it fails**

Run: `cd ~/cog-rag-cognee && python -m pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# cog_rag_cognee/config.py
"""Application configuration via Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration loaded from environment / .env file."""

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_endpoint: str = "http://localhost:11434"

    # Embeddings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_endpoint: str = "http://localhost:11434/api/embed"
    embedding_dimensions: int = 768

    # Graph DB
    graph_db_provider: str = "neo4j"
    graph_db_url: str = "neo4j://localhost:7687"
    graph_db_username: str = "neo4j"
    graph_db_password: str = "password"

    # Vector DB
    vector_db_provider: str = "lancedb"

    # Storage
    storage_root_dir: str = "./cognee_data"

    # API
    api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8508

    # UI
    ui_port: int = 8506

    @property
    def ollama_base_url(self) -> str:
        return self.llm_endpoint

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
```

**Step 4: Run test to verify it passes**

Run: `cd ~/cog-rag-cognee && python -m pytest tests/test_config.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add cog_rag_cognee/config.py tests/test_config.py
git commit -m "feat: config module — Pydantic Settings with Ollama/Neo4j/LanceDB defaults"
```

---

### Task 3: Domain models

**Files:**
- Create: `tests/test_models.py`
- Create: `cog_rag_cognee/models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
"""Tests for domain models."""


def test_search_result_creation():
    from cog_rag_cognee.models import SearchResult

    r = SearchResult(content="test answer", score=0.95, source="doc1.pdf")
    assert r.content == "test answer"
    assert r.score == 0.95
    assert r.source == "doc1.pdf"


def test_qa_result_creation():
    from cog_rag_cognee.models import QAResult, SearchResult

    sources = [SearchResult(content="chunk1", score=0.9, source="doc1.pdf")]
    qa = QAResult(answer="The answer is 42", confidence=0.85, sources=sources, mode="graph")
    assert qa.answer == "The answer is 42"
    assert qa.confidence == 0.85
    assert len(qa.sources) == 1
    assert qa.mode == "graph"


def test_ingest_result():
    from cog_rag_cognee.models import IngestResult

    r = IngestResult(filename="doc.pdf", chunks_added=10, status="success")
    assert r.filename == "doc.pdf"
    assert r.chunks_added == 10


def test_graph_stats():
    from cog_rag_cognee.models import GraphStats

    gs = GraphStats(nodes=100, edges=250, entity_types={"Person": 40, "Organization": 60})
    assert gs.nodes == 100
    assert gs.edges == 250
    assert gs.entity_types["Person"] == 40
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cog_rag_cognee/models.py
"""Domain models for cog-rag-cognee."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result from Cognee."""
    content: str
    score: float = Field(ge=0.0, le=1.0)
    source: str = ""
    entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class QAResult(BaseModel):
    """Full question-answering result."""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SearchResult] = Field(default_factory=list)
    mode: str = "graph"


class IngestResult(BaseModel):
    """Result of document ingestion."""
    filename: str
    chunks_added: int = 0
    entities_extracted: int = 0
    status: str = "success"
    error: str | None = None


class GraphStats(BaseModel):
    """Knowledge graph statistics."""
    nodes: int = 0
    edges: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Service health status."""
    status: str = "ok"
    ollama: bool = False
    neo4j: bool = False
    lancedb: bool = False
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: 4 PASSED

**Step 5: Commit**

```bash
git add cog_rag_cognee/models.py tests/test_models.py
git commit -m "feat: domain models — SearchResult, QAResult, IngestResult, GraphStats"
```

---

### Task 4: Exceptions and Docker Compose

**Files:**
- Create: `cog_rag_cognee/exceptions.py`
- Create: `docker-compose.yml`
- Create: `scripts/pull_models.sh`

**Step 1: Create exceptions**

```python
# cog_rag_cognee/exceptions.py
"""Custom exception hierarchy."""


class CogRagError(Exception):
    """Base exception for cog-rag-cognee."""


class ConfigError(CogRagError):
    """Configuration error."""


class IngestionError(CogRagError):
    """Error during document ingestion."""


class SearchError(CogRagError):
    """Error during search/query."""


class GraphError(CogRagError):
    """Error accessing knowledge graph."""


class OllamaError(CogRagError):
    """Error communicating with Ollama."""
```

**Step 2: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5
    container_name: cog-rag-cognee-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama
    container_name: cog-rag-cognee-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  ollama_data:
```

**Step 3: Create pull_models.sh**

```bash
#!/usr/bin/env bash
# scripts/pull_models.sh — Download required Ollama models
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "Pulling LLM model..."
ollama pull llama3.1:8b

echo "Pulling embedding model..."
ollama pull nomic-embed-text

echo "Verifying models..."
ollama list | grep -E "llama3.1:8b|nomic-embed-text"

echo "Done! Models ready."
```

**Step 4: Commit**

```bash
chmod +x scripts/pull_models.sh
git add cog_rag_cognee/exceptions.py docker-compose.yml scripts/pull_models.sh
git commit -m "feat: exceptions, docker-compose (Neo4j + Ollama), model pull script"
```

---

## Phase 2: Cognee SDK Integration

### Task 5: Cognee setup module

**Files:**
- Create: `tests/test_cognee_setup.py`
- Create: `cog_rag_cognee/cognee_setup.py`

**Step 1: Write the failing test**

```python
# tests/test_cognee_setup.py
"""Tests for Cognee SDK configuration."""
import os
from unittest.mock import patch


def test_build_env_dict():
    """build_cognee_env() should produce correct env vars."""
    from cog_rag_cognee.cognee_setup import build_cognee_env
    from cog_rag_cognee.config import Settings

    settings = Settings()
    env = build_cognee_env(settings)

    assert env["LLM_PROVIDER"] == "ollama"
    assert env["LLM_MODEL"] == "llama3.1:8b"
    assert env["EMBEDDING_PROVIDER"] == "ollama"
    assert env["EMBEDDING_DIMENSIONS"] == "768"
    assert env["GRAPH_DATABASE_PROVIDER"] == "neo4j"
    assert env["VECTOR_DB_PROVIDER"] == "lancedb"


def test_build_env_custom_settings():
    """Custom settings should be reflected in env dict."""
    from cog_rag_cognee.cognee_setup import build_cognee_env
    from cog_rag_cognee.config import Settings

    settings = Settings(llm_model="mistral:7b", embedding_dimensions=1024)
    env = build_cognee_env(settings)

    assert env["LLM_MODEL"] == "mistral:7b"
    assert env["EMBEDDING_DIMENSIONS"] == "1024"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cognee_setup.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cog_rag_cognee/cognee_setup.py
"""Cognee SDK initialization and configuration."""
from __future__ import annotations

import os
import logging

from cog_rag_cognee.config import Settings

logger = logging.getLogger(__name__)


def build_cognee_env(settings: Settings) -> dict[str, str]:
    """Build environment variables dict for Cognee SDK."""
    return {
        # LLM
        "LLM_PROVIDER": settings.llm_provider,
        "LLM_MODEL": settings.llm_model,
        "LLM_ENDPOINT": settings.llm_endpoint,
        # Embeddings
        "EMBEDDING_PROVIDER": settings.embedding_provider,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_ENDPOINT": settings.embedding_endpoint,
        "EMBEDDING_DIMENSIONS": str(settings.embedding_dimensions),
        # Graph
        "GRAPH_DATABASE_PROVIDER": settings.graph_db_provider,
        "GRAPH_DATABASE_URL": settings.graph_db_url,
        "GRAPH_DATABASE_USERNAME": settings.graph_db_username,
        "GRAPH_DATABASE_PASSWORD": settings.graph_db_password,
        # Vector
        "VECTOR_DB_PROVIDER": settings.vector_db_provider,
        # Storage
        "STORAGE_ROOT_DIR": settings.storage_root_dir,
    }


def apply_cognee_env(settings: Settings) -> None:
    """Set environment variables for Cognee SDK before import."""
    env = build_cognee_env(settings)
    for key, value in env.items():
        os.environ[key] = value
    logger.info(
        "Cognee configured: LLM=%s/%s, Graph=%s, Vector=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.graph_db_provider,
        settings.vector_db_provider,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cognee_setup.py -v`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add cog_rag_cognee/cognee_setup.py tests/test_cognee_setup.py
git commit -m "feat: cognee_setup — environment builder for Cognee SDK"
```

---

### Task 6: PipelineService (core wrapper)

**Files:**
- Create: `tests/test_service.py`
- Create: `cog_rag_cognee/service.py`

**Step 1: Write the failing test**

```python
# tests/test_service.py
"""Tests for PipelineService (Cognee wrapper)."""
from unittest.mock import AsyncMock, patch, MagicMock
import pytest


@pytest.fixture
def mock_cognee():
    """Mock cognee module."""
    with patch("cog_rag_cognee.service.cognee") as mock:
        mock.add = AsyncMock()
        mock.cognify = AsyncMock(return_value={"main": MagicMock(
            status="completed", chunks_processed=5, entities_extracted=3
        )})
        mock.search = AsyncMock(return_value=[
            MagicMock(content="Test answer", relevance_score=0.9)
        ])
        mock.prune = MagicMock()
        mock.prune.prune_data = AsyncMock()
        mock.prune.prune_system = AsyncMock()
        yield mock


@pytest.mark.asyncio
async def test_add_text(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.add_text("Hello world")

    mock_cognee.add.assert_called_once_with("Hello world")
    assert result is not None


@pytest.mark.asyncio
async def test_cognify(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    result = await svc.cognify()

    mock_cognee.cognify.assert_called_once()
    assert result["main"].status == "completed"


@pytest.mark.asyncio
async def test_search(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    results = await svc.search("What is Cognee?")

    mock_cognee.search.assert_called_once()
    assert len(results) > 0


@pytest.mark.asyncio
async def test_query_full_pipeline(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    qa = await svc.query("What is Cognee?", search_type="GRAPH_COMPLETION")

    assert qa.answer != ""
    assert 0.0 <= qa.confidence <= 1.0


@pytest.mark.asyncio
async def test_reset(mock_cognee):
    from cog_rag_cognee.service import PipelineService

    svc = PipelineService()
    await svc.reset()

    mock_cognee.prune.prune_data.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cog_rag_cognee/service.py
"""PipelineService — thin wrapper over Cognee SDK."""
from __future__ import annotations

import logging
from typing import Any

import cognee

from cog_rag_cognee.models import QAResult, SearchResult

logger = logging.getLogger(__name__)


class PipelineService:
    """Orchestrates Cognee SDK operations."""

    async def add_text(self, text: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add text data to Cognee."""
        await cognee.add(text, dataset_name=dataset_name)
        logger.info("Added text (%d chars) to dataset '%s'", len(text), dataset_name)
        return {"status": "added", "chars": len(text), "dataset": dataset_name}

    async def add_file(self, file_path: str, dataset_name: str = "main") -> dict[str, Any]:
        """Add file content to Cognee."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        await cognee.add(content, dataset_name=dataset_name)
        logger.info("Added file '%s' (%d chars) to dataset '%s'", file_path, len(content), dataset_name)
        return {"status": "added", "file": file_path, "chars": len(content)}

    async def cognify(self, dataset_name: str | None = None) -> dict[str, Any]:
        """Run Cognee ECL pipeline: extract entities, build graph, embed."""
        kwargs = {}
        if dataset_name:
            kwargs["datasets"] = [dataset_name]
        result = await cognee.cognify(**kwargs)
        logger.info("Cognify completed: %s", result)
        return result

    async def search(
        self,
        query: str,
        search_type: str = "GRAPH_COMPLETION",
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search Cognee knowledge graph."""
        raw_results = await cognee.search(query, search_type=search_type, limit=limit)
        results = []
        for r in raw_results:
            content = r.content if hasattr(r, "content") else str(r)
            score = r.relevance_score if hasattr(r, "relevance_score") else 0.5
            score = max(0.0, min(1.0, float(score)))
            results.append(SearchResult(content=content, score=score))
        return results

    async def query(
        self,
        question: str,
        search_type: str = "GRAPH_COMPLETION",
        limit: int = 5,
    ) -> QAResult:
        """Full RAG pipeline: search + format answer."""
        sources = await self.search(question, search_type=search_type, limit=limit)
        if sources:
            answer = sources[0].content
            confidence = sources[0].score
        else:
            answer = "No relevant information found."
            confidence = 0.0
        return QAResult(
            answer=answer,
            confidence=confidence,
            sources=sources,
            mode=search_type,
        )

    async def reset(self) -> None:
        """Reset all Cognee data."""
        await cognee.prune.prune_data()
        logger.info("Data reset completed")

    async def reset_system(self) -> None:
        """Full system reset including metadata."""
        await cognee.prune.prune_system(metadata=True)
        logger.info("System reset completed")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_service.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add cog_rag_cognee/service.py tests/test_service.py
git commit -m "feat: PipelineService — Cognee wrapper with add/cognify/search/query/reset"
```

---

## Phase 3: FastAPI REST API

### Task 7: FastAPI app factory

**Files:**
- Create: `tests/test_app.py`
- Create: `api/__init__.py`
- Create: `api/deps.py`
- Create: `api/app.py`
- Create: `api/routes.py`
- Create: `run_api.py`

**Step 1: Write the failing test**

```python
# tests/test_app.py
"""Tests for FastAPI application."""
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked service."""
    with patch("api.deps._service") as mock_svc:
        mock_svc.query = AsyncMock(return_value=MagicMock(
            answer="Test answer",
            confidence=0.9,
            sources=[],
            mode="GRAPH_COMPLETION",
            model_dump=lambda: {
                "answer": "Test answer",
                "confidence": 0.9,
                "sources": [],
                "mode": "GRAPH_COMPLETION",
            },
        ))
        mock_svc.search = AsyncMock(return_value=[])
        mock_svc.add_text = AsyncMock(return_value={"status": "added"})
        mock_svc.cognify = AsyncMock(return_value={})
        mock_svc.reset = AsyncMock()

        from api.app import create_app
        app = create_app()
        yield TestClient(app)


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_query(client):
    resp = client.post("/api/v1/query", json={"text": "What is Cognee?", "mode": "GRAPH_COMPLETION"})
    assert resp.status_code == 200


def test_search(client):
    resp = client.post("/api/v1/search", json={"text": "test query", "mode": "GRAPH_COMPLETION"})
    assert resp.status_code == 200


def test_query_missing_text(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL

**Step 3: Write deps.py**

```python
# api/__init__.py
```

```python
# api/deps.py
"""Dependency injection for FastAPI."""
from __future__ import annotations

from cog_rag_cognee.service import PipelineService

_service: PipelineService | None = None


def get_service() -> PipelineService:
    """Return the global PipelineService instance."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service


def set_service(service: PipelineService) -> None:
    """Set the global PipelineService instance (for testing)."""
    global _service
    _service = service
```

**Step 4: Write routes.py**

```python
# api/routes.py
"""REST API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_service
from cog_rag_cognee.service import PipelineService

router = APIRouter(prefix="/api/v1")


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = "GRAPH_COMPLETION"
    limit: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    dataset_name: str = "main"


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/query")
async def query(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    result = await svc.query(req.text, search_type=req.mode, limit=req.limit)
    return result.model_dump()


@router.post("/search")
async def search(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    results = await svc.search(req.text, search_type=req.mode, limit=req.limit)
    return [r.model_dump() for r in results]


@router.post("/ingest")
async def ingest(req: IngestRequest, svc: PipelineService = Depends(get_service)):
    result = await svc.add_text(req.text, dataset_name=req.dataset_name)
    cognify_result = await svc.cognify(dataset_name=req.dataset_name)
    return {"ingest": result, "cognify": str(cognify_result)}


@router.get("/graph/stats")
async def graph_stats():
    return {"nodes": 0, "edges": 0, "entity_types": {}}
```

**Step 5: Write app.py**

```python
# api/app.py
"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    apply_cognee_env(settings)
    logger.info("cog-rag-cognee API started on port %d", settings.api_port)
    yield
    logger.info("cog-rag-cognee API shutting down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="cog-rag-cognee",
        description="Semantic memory layer with Cognee SDK — 100% local stack",
        version="0.1.0",
        lifespan=lifespan,
    )

    from api.routes import router
    app.include_router(router)

    return app
```

**Step 6: Write run_api.py**

```python
# run_api.py
"""Launch the FastAPI server."""
import uvicorn
from dotenv import load_dotenv

from cog_rag_cognee.config import get_settings

if __name__ == "__main__":
    load_dotenv()
    settings = get_settings()
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
```

**Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: 4 PASSED

**Step 8: Commit**

```bash
git add api/ run_api.py tests/test_app.py
git commit -m "feat: FastAPI REST API — /health, /query, /search, /ingest, /graph/stats"
```

---

## Phase 4: Ontology Integration

### Task 8: Ontology loader

**Files:**
- Create: `tests/test_ontology.py`
- Create: `cog_rag_cognee/ontology.py`
- Create: `ontologies/example.owl`

**Step 1: Write the failing test**

```python
# tests/test_ontology.py
"""Tests for ontology loader."""
import os
import tempfile

import pytest


def test_load_owl_file():
    """Should parse OWL file and extract classes and properties."""
    from cog_rag_cognee.ontology import load_ontology

    owl_path = os.path.join(os.path.dirname(__file__), "..", "ontologies", "example.owl")
    if not os.path.exists(owl_path):
        pytest.skip("example.owl not found")

    onto = load_ontology(owl_path)
    assert "classes" in onto
    assert "properties" in onto
    assert len(onto["classes"]) > 0


def test_load_nonexistent_file():
    """Should raise FileNotFoundError for missing file."""
    from cog_rag_cognee.ontology import load_ontology

    with pytest.raises(FileNotFoundError):
        load_ontology("/nonexistent/file.owl")


def test_ontology_to_cognee_schema():
    """Should convert ontology to Cognee-compatible schema hints."""
    from cog_rag_cognee.ontology import ontology_to_schema_hints

    onto = {
        "classes": ["Person", "Organization", "Location"],
        "properties": [
            {"name": "worksFor", "domain": "Person", "range": "Organization"},
            {"name": "locatedIn", "domain": "Organization", "range": "Location"},
        ],
    }
    hints = ontology_to_schema_hints(onto)
    assert "Person" in hints["entity_types"]
    assert any(r["name"] == "worksFor" for r in hints["relationship_types"])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ontology.py -v`
Expected: FAIL

**Step 3: Create example.owl**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:ex="http://example.org/ontology#">

  <owl:Ontology rdf:about="http://example.org/ontology"/>

  <!-- Classes -->
  <owl:Class rdf:about="http://example.org/ontology#Person"/>
  <owl:Class rdf:about="http://example.org/ontology#Organization"/>
  <owl:Class rdf:about="http://example.org/ontology#Location"/>
  <owl:Class rdf:about="http://example.org/ontology#Document"/>

  <!-- Object Properties -->
  <owl:ObjectProperty rdf:about="http://example.org/ontology#worksFor">
    <rdfs:domain rdf:resource="http://example.org/ontology#Person"/>
    <rdfs:range rdf:resource="http://example.org/ontology#Organization"/>
  </owl:ObjectProperty>

  <owl:ObjectProperty rdf:about="http://example.org/ontology#locatedIn">
    <rdfs:domain rdf:resource="http://example.org/ontology#Organization"/>
    <rdfs:range rdf:resource="http://example.org/ontology#Location"/>
  </owl:ObjectProperty>

  <owl:ObjectProperty rdf:about="http://example.org/ontology#mentionedIn">
    <rdfs:domain rdf:resource="http://example.org/ontology#Person"/>
    <rdfs:range rdf:resource="http://example.org/ontology#Document"/>
  </owl:ObjectProperty>

</rdf:RDF>
```

**Step 4: Write implementation**

```python
# cog_rag_cognee/ontology.py
"""OWL/RDF ontology loader for domain grounding."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"


def _local_name(uri: str) -> str:
    """Extract local name from URI (after # or last /)."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rsplit("/", 1)[-1]


def load_ontology(file_path: str) -> dict[str, Any]:
    """Parse OWL/RDF file and extract classes and properties."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {file_path}")

    tree = ET.parse(path)
    root = tree.getroot()

    classes = []
    for cls in root.findall(f".//{{{OWL_NS}}}Class"):
        about = cls.get(f"{{{RDF_NS}}}about", "")
        if about:
            classes.append(_local_name(about))

    properties = []
    for prop in root.findall(f".//{{{OWL_NS}}}ObjectProperty"):
        about = prop.get(f"{{{RDF_NS}}}about", "")
        if not about:
            continue
        name = _local_name(about)
        domain_el = prop.find(f"{{{RDFS_NS}}}domain")
        range_el = prop.find(f"{{{RDFS_NS}}}range")
        domain = _local_name(domain_el.get(f"{{{RDF_NS}}}resource", "")) if domain_el is not None else ""
        range_ = _local_name(range_el.get(f"{{{RDF_NS}}}resource", "")) if range_el is not None else ""
        properties.append({"name": name, "domain": domain, "range": range_})

    logger.info("Loaded ontology: %d classes, %d properties from %s", len(classes), len(properties), file_path)
    return {"classes": classes, "properties": properties}


def ontology_to_schema_hints(onto: dict[str, Any]) -> dict[str, Any]:
    """Convert parsed ontology to Cognee-compatible schema hints."""
    return {
        "entity_types": onto.get("classes", []),
        "relationship_types": onto.get("properties", []),
    }
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_ontology.py -v`
Expected: 3 PASSED

**Step 6: Commit**

```bash
git add cog_rag_cognee/ontology.py tests/test_ontology.py ontologies/example.owl
git commit -m "feat: ontology loader — OWL/RDF parsing with schema hints for Cognee"
```

---

## Phase 5: Streamlit UI

### Task 9: i18n module

**Files:**
- Create: `tests/test_i18n.py`
- Create: `ui/__init__.py`
- Create: `ui/i18n.py`

**Step 1: Write the failing test**

```python
# tests/test_i18n.py
"""Tests for i18n translations."""


def test_english_translator():
    from ui.i18n import get_translator

    t = get_translator("en")
    assert t("app_title") == "Cog-RAG Cognee"
    assert t("tab_upload") == "Upload"
    assert t("tab_search") == "Search & Q&A"


def test_russian_translator():
    from ui.i18n import get_translator

    t = get_translator("ru")
    assert t("app_title") == "Cog-RAG Cognee"
    assert t("tab_upload") == "Загрузка"
    assert t("tab_search") == "Поиск и Q&A"


def test_missing_key_fallback():
    from ui.i18n import get_translator

    t = get_translator("en")
    assert t("nonexistent_key") == "nonexistent_key"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_i18n.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# ui/__init__.py
```

```python
# ui/i18n.py
"""Internationalization: EN/RU translations."""
from __future__ import annotations

from typing import Callable

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "Cog-RAG Cognee",
        "tab_upload": "Upload",
        "tab_search": "Search & Q&A",
        "tab_graph": "Graph Explorer",
        "tab_settings": "Settings",
        "upload_header": "Upload Documents",
        "upload_drag": "Drag and drop files here",
        "upload_text": "Or paste text below",
        "upload_btn": "Ingest & Cognify",
        "upload_success": "Document ingested successfully!",
        "search_header": "Search Knowledge Graph",
        "search_placeholder": "Ask a question...",
        "search_btn": "Search",
        "search_mode": "Search mode",
        "search_results": "Results",
        "search_answer": "Answer",
        "search_confidence": "Confidence",
        "search_sources": "Sources",
        "graph_header": "Knowledge Graph Explorer",
        "graph_nodes": "Nodes",
        "graph_edges": "Edges",
        "settings_header": "Settings",
        "settings_config": "Current Configuration",
        "settings_clear": "Clear all data",
        "settings_clear_confirm": "Are you sure? This will delete all data.",
        "health_ok": "All services running",
        "health_fail": "Some services unavailable",
        "language": "Language",
    },
    "ru": {
        "app_title": "Cog-RAG Cognee",
        "tab_upload": "Загрузка",
        "tab_search": "Поиск и Q&A",
        "tab_graph": "Граф знаний",
        "tab_settings": "Настройки",
        "upload_header": "Загрузка документов",
        "upload_drag": "Перетащите файлы сюда",
        "upload_text": "Или вставьте текст ниже",
        "upload_btn": "Загрузить и обработать",
        "upload_success": "Документ успешно загружен!",
        "search_header": "Поиск по графу знаний",
        "search_placeholder": "Задайте вопрос...",
        "search_btn": "Найти",
        "search_mode": "Режим поиска",
        "search_results": "Результаты",
        "search_answer": "Ответ",
        "search_confidence": "Уверенность",
        "search_sources": "Источники",
        "graph_header": "Граф знаний",
        "graph_nodes": "Узлы",
        "graph_edges": "Связи",
        "settings_header": "Настройки",
        "settings_config": "Текущая конфигурация",
        "settings_clear": "Очистить все данные",
        "settings_clear_confirm": "Вы уверены? Все данные будут удалены.",
        "health_ok": "Все сервисы работают",
        "health_fail": "Некоторые сервисы недоступны",
        "language": "Язык",
    },
}


def get_translator(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the given language."""
    translations = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

    def translate(key: str) -> str:
        return translations.get(key, key)

    return translate
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_i18n.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add ui/ tests/test_i18n.py
git commit -m "feat: i18n module — EN/RU translations for Streamlit UI"
```

---

### Task 10: Streamlit app (4 tabs)

**Files:**
- Create: `ui/streamlit_app.py`
- Create: `ui/components/__init__.py`
- Create: `ui/components/graph_viz.py`

**Step 1: Create graph_viz.py**

```python
# ui/components/__init__.py
```

```python
# ui/components/graph_viz.py
"""PyVis interactive graph visualization."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_graph(nodes: list[dict], edges: list[dict]) -> None:
    """Render interactive graph using PyVis."""
    if not nodes:
        st.info("No graph data available. Ingest documents first.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        st.error("PyVis not installed: pip install pyvis")
        return

    net = Network(height="600px", width="100%", directed=True)
    net.barnes_hut()

    color_map = {
        "Person": "#e74c3c",
        "Organization": "#3498db",
        "Location": "#2ecc71",
        "Date": "#f39c12",
        "Document": "#9b59b6",
        "Chunk": "#95a5a6",
    }

    for node in nodes:
        color = color_map.get(node.get("type", ""), "#bdc3c7")
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            color=color,
            title=f"{node.get('type', 'Unknown')}: {node.get('label', '')}",
        )

    for edge in edges:
        net.add_edge(edge["source"], edge["target"], label=edge.get("type", ""))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_content = Path(f.name).read_text()
        components.html(html_content, height=620)
```

**Step 2: Create streamlit_app.py**

```python
# ui/streamlit_app.py
"""Streamlit UI for cog-rag-cognee — 4 tabs."""
from __future__ import annotations

import asyncio

import streamlit as st

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import get_settings
from cog_rag_cognee.service import PipelineService
from ui.i18n import get_translator
from ui.components.graph_viz import render_graph

# --- Page config ---
st.set_page_config(page_title="Cog-RAG Cognee", page_icon="🧠", layout="wide")

# --- Sidebar ---
lang = st.sidebar.selectbox("Language / Язык", ["en", "ru"], index=0)
t = get_translator(lang)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{t('app_title')}** v0.1.0")
st.sidebar.markdown("100% local stack")


@st.cache_resource
def init_service() -> PipelineService:
    settings = get_settings()
    apply_cognee_env(settings)
    return PipelineService()


svc = init_service()

# --- Tabs ---
tab_upload, tab_search, tab_graph, tab_settings = st.tabs([
    t("tab_upload"), t("tab_search"), t("tab_graph"), t("tab_settings")
])

# --- Tab 1: Upload ---
with tab_upload:
    st.header(t("upload_header"))

    uploaded_file = st.file_uploader(t("upload_drag"), type=["pdf", "txt", "md", "docx", "html"])
    text_input = st.text_area(t("upload_text"), height=200)

    if st.button(t("upload_btn")):
        with st.spinner("Processing..."):
            if text_input.strip():
                result = asyncio.run(svc.add_text(text_input))
                cognify_result = asyncio.run(svc.cognify())
                st.success(t("upload_success"))
                st.json({"ingest": result, "cognify": str(cognify_result)})
            elif uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
                result = asyncio.run(svc.add_text(content))
                cognify_result = asyncio.run(svc.cognify())
                st.success(t("upload_success"))
                st.json({"ingest": result, "cognify": str(cognify_result)})
            else:
                st.warning("Please upload a file or enter text.")

# --- Tab 2: Search & Q&A ---
with tab_search:
    st.header(t("search_header"))

    search_mode = st.selectbox(
        t("search_mode"),
        ["GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES"],
        index=0,
    )

    query = st.text_input(t("search_placeholder"))

    if st.button(t("search_btn")) and query:
        with st.spinner("Searching..."):
            qa = asyncio.run(svc.query(query, search_type=search_mode))

            st.subheader(t("search_answer"))
            st.text(qa.answer)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(t("search_confidence"), f"{qa.confidence:.0%}")
            with col2:
                st.metric(t("search_results"), len(qa.sources))

            if qa.sources:
                with st.expander(t("search_sources")):
                    for i, src in enumerate(qa.sources, 1):
                        st.markdown(f"**{i}.** [{src.score:.2f}] {src.content[:200]}")

# --- Tab 3: Graph Explorer ---
with tab_graph:
    st.header(t("graph_header"))
    # Placeholder: will query Neo4j directly for graph data
    render_graph([], [])

# --- Tab 4: Settings ---
with tab_settings:
    st.header(t("settings_header"))

    settings = get_settings()
    st.subheader(t("settings_config"))
    st.json({
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "embeddings": f"{settings.embedding_provider}/{settings.embedding_model}",
        "graph_db": settings.graph_db_provider,
        "vector_db": settings.vector_db_provider,
    })

    st.markdown("---")
    if st.button(t("settings_clear"), type="secondary"):
        if st.checkbox(t("settings_clear_confirm")):
            asyncio.run(svc.reset())
            st.success("Data cleared!")
```

**Step 3: Commit**

```bash
git add ui/
git commit -m "feat: Streamlit UI — 4 tabs (Upload, Search, Graph, Settings) with i18n"
```

---

## Phase 6: Tests, Benchmark, and README

### Task 11: conftest and test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create conftest**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared pytest fixtures."""
import os
import pytest

# Ensure tests don't connect to real services
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.1:8b")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "neo4j")
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
```

**Step 2: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "feat: test infrastructure — conftest with env defaults"
```

---

### Task 12: Sample data and benchmark setup

**Files:**
- Create: `data/sample_en.txt`
- Create: `data/sample_ru.txt`
- Create: `benchmark/questions.json`
- Create: `scripts/ingest.py`

**Step 1: Create sample data**

```
# data/sample_en.txt
Cognee is a knowledge engine that transforms documents into AI memory.
It uses knowledge graphs to store entities and their relationships.
Neo4j is used as the graph database backend for persistent storage.
LanceDB provides embedded vector search without requiring Docker.
Ollama enables local LLM inference with models like Llama 3.1.
```

```
# data/sample_ru.txt
Cognee — это движок знаний, который превращает документы в память AI.
Он использует графы знаний для хранения сущностей и их связей.
Neo4j используется как графовая база данных для постоянного хранения.
LanceDB обеспечивает встроенный векторный поиск без необходимости Docker.
Ollama позволяет запускать LLM локально с моделями типа Llama 3.1.
```

**Step 2: Create benchmark questions**

```json
[
  {"question": "What is Cognee?", "expected_keywords": ["knowledge", "engine", "memory"], "category": "simple", "lang": "en"},
  {"question": "What database stores the knowledge graph?", "expected_keywords": ["neo4j", "graph"], "category": "simple", "lang": "en"},
  {"question": "How does vector search work without Docker?", "expected_keywords": ["lancedb", "embedded"], "category": "simple", "lang": "en"},
  {"question": "What model does Ollama use?", "expected_keywords": ["llama", "3.1"], "category": "simple", "lang": "en"},
  {"question": "What is the relationship between Cognee and Neo4j?", "expected_keywords": ["graph", "storage", "entities"], "category": "relation", "lang": "en"},
  {"question": "Что такое Cognee?", "expected_keywords": ["знаний", "память", "документы"], "category": "simple", "lang": "ru"},
  {"question": "Какая база данных хранит граф знаний?", "expected_keywords": ["neo4j", "графовая"], "category": "simple", "lang": "ru"},
  {"question": "Как работает векторный поиск без Docker?", "expected_keywords": ["lancedb", "встроенный"], "category": "simple", "lang": "ru"},
  {"question": "Какие модели использует Ollama?", "expected_keywords": ["llama", "3.1"], "category": "simple", "lang": "ru"},
  {"question": "Какая связь между Cognee и Neo4j?", "expected_keywords": ["граф", "хранения", "сущност"], "category": "relation", "lang": "ru"}
]
```

**Step 3: Create CLI ingest script**

```python
# scripts/ingest.py
"""CLI ingestion script."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import get_settings
from cog_rag_cognee.service import PipelineService


async def main(file_paths: list[str]) -> None:
    settings = get_settings()
    apply_cognee_env(settings)

    svc = PipelineService()

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"SKIP: {fp} not found")
            continue

        print(f"Ingesting: {fp}")
        content = path.read_text(encoding="utf-8")
        await svc.add_text(content)
        print(f"  Added {len(content)} chars")

    print("Running cognify...")
    result = await svc.cognify()
    print(f"Cognify result: {result}")
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file1> [file2] ...")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
```

**Step 4: Commit**

```bash
git add data/ benchmark/ scripts/ingest.py
git commit -m "feat: sample data, benchmark questions (10 EN/RU), CLI ingest script"
```

---

### Task 13: Run all tests and lint

**Step 1: Run full test suite**

Run: `cd ~/cog-rag-cognee && python -m pytest tests/ -v`
Expected: All tests PASSED (config: 3, models: 4, cognee_setup: 2, service: 5, app: 4, ontology: 3, i18n: 3 = **24 total**)

**Step 2: Run ruff lint**

Run: `cd ~/cog-rag-cognee && python -m ruff check .`
Expected: Clean (0 errors)

**Step 3: Fix any issues, then commit**

```bash
git add -A
git commit -m "chore: lint clean, all 24 tests passing"
```

---

### Task 14: README and final commit

**Files:**
- Create: `README.md`

**Step 1: Write README**

Key sections:
- Title + description
- Architecture diagram (ASCII)
- Prerequisites (Ollama, Neo4j, Python 3.10+)
- Quick Start (5 steps: clone, install, docker, models, run)
- API endpoints table
- Streamlit UI screenshots placeholder
- Benchmark results placeholder
- Tech stack table
- License

**Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: README with architecture, quick start, and API docs"
```

---

## Summary

| Phase | Tasks | Tests | Commits |
|-------|-------|-------|---------|
| 1. Scaffolding | 1-4 | 10 | 4 |
| 2. Cognee SDK | 5-6 | 7 | 2 |
| 3. FastAPI API | 7 | 4 | 1 |
| 4. Ontology | 8 | 3 | 1 |
| 5. Streamlit UI | 9-10 | 3 | 2 |
| 6. Tests & Docs | 11-14 | 0 (infra) | 4 |
| **Total** | **14 tasks** | **27 tests** | **14 commits** |

## Post-MVP (deferred)

- [ ] BM25 via tantivy or SQLite FTS5
- [ ] Memify graph optimization pipeline
- [ ] Iterative probing (from Cog-RAG cognitive mode)
- [ ] Docling loader integration (GPU support)
- [ ] Graph Explorer tab: live Neo4j query
- [ ] Benchmark runner script
- [ ] CI/CD (GitHub Actions)
- [ ] Semantic cache (LRU + cosine + TTL)
