"""Tests for FastAPI application."""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from cog_rag_cognee.models import QAResult


@pytest.fixture
def mock_service():
    """Create a mock PipelineService."""
    svc = MagicMock()
    svc.query = AsyncMock(
        return_value=QAResult(
            answer="Test answer",
            confidence=0.9,
            sources=[],
            mode="GRAPH_COMPLETION",
        )
    )
    svc.search = AsyncMock(return_value=[])
    svc.add_text = AsyncMock(return_value={"status": "added", "chars": 10, "dataset": "main"})
    svc.add_bytes = AsyncMock(return_value={"status": "added", "file": "test.txt", "chars": 12})
    svc.cognify = AsyncMock(return_value={"main": "completed"})
    svc.reset = AsyncMock()
    svc.list_datasets = AsyncMock(return_value=["main", "papers", "docs"])
    return svc


@pytest.fixture
def mock_graph_client():
    """Create a mock async GraphClient."""
    gc = MagicMock()
    gc.get_stats = AsyncMock(return_value={
        "nodes": 10,
        "edges": 15,
        "entity_types": {"Person": 5, "Organization": 3, "Location": 2},
    })
    gc.get_entities = AsyncMock(return_value=[
        {"id": 1, "label": "Alice", "type": "Person"},
        {"id": 2, "label": "Acme Corp", "type": "Organization"},
    ])
    gc.get_relationships = AsyncMock(return_value=[
        {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT"},
    ])
    gc.health_check = AsyncMock(return_value=True)
    gc.close = AsyncMock()
    return gc


@pytest.fixture
def client(mock_service, mock_graph_client):
    """Create test client with mocked service and graph client."""
    from api.deps import set_graph_client, set_service

    set_service(mock_service)
    set_graph_client(mock_graph_client)

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    # Cleanup
    set_service(None)
    set_graph_client(None)


def test_health(client, mock_graph_client, monkeypatch):
    """Health endpoint checks Neo4j and Ollama connectivity."""
    mock_graph_client.health_check = AsyncMock(return_value=True)
    monkeypatch.setattr("api.routes.check_ollama", AsyncMock(return_value=True))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["neo4j"] is True


def test_health_neo4j_down(client, mock_graph_client):
    """Health returns degraded when Neo4j is unreachable."""
    mock_graph_client.health_check = AsyncMock(return_value=False)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is False


def test_health_neo4j_exception(client, mock_graph_client):
    """Health returns degraded when Neo4j health check throws."""
    mock_graph_client.health_check = AsyncMock(side_effect=Exception("refused"))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is False


def test_liveness_always_200(client, mock_graph_client):
    """Liveness probe returns 200 without calling any dependencies."""
    mock_graph_client.health_check = AsyncMock(side_effect=Exception("down"))
    resp = client.get("/api/v1/liveness")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_readiness_200_when_all_healthy(client, mock_graph_client, monkeypatch):
    """Readiness returns 200 when all deps are healthy."""
    mock_graph_client.health_check = AsyncMock(return_value=True)
    monkeypatch.setattr("api.routes.check_ollama", AsyncMock(return_value=True))
    resp = client.get("/api/v1/readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["checks"]["neo4j"] is True


def test_readiness_503_when_neo4j_down(client, mock_graph_client):
    """Readiness returns 503 when Neo4j is unreachable."""
    mock_graph_client.health_check = AsyncMock(return_value=False)
    resp = client.get("/api/v1/readiness")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["neo4j"] is False


def test_query(client):
    """Query endpoint returns answer with confidence."""
    resp = client.post("/api/v1/query", json={"text": "What is Cognee?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer"
    assert data["confidence"] == 0.9


def test_search(client):
    """Search endpoint returns list of results."""
    resp = client.post("/api/v1/search", json={"text": "test query"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_query_missing_text(client):
    """Query endpoint rejects missing text field."""
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_query_empty_text(client):
    """Query endpoint rejects empty text."""
    resp = client.post("/api/v1/query", json={"text": ""})
    assert resp.status_code == 422


def test_query_whitespace_only_text_rejected(client):
    """Query endpoint rejects whitespace-only text."""
    resp = client.post("/api/v1/query", json={"text": "   "})
    assert resp.status_code == 422


def test_ingest_whitespace_only_text_rejected(client):
    """Ingest endpoint rejects whitespace-only text."""
    resp = client.post("/api/v1/ingest", json={"text": "\n\t\n"})
    assert resp.status_code == 422


def test_query_limit_zero_rejected(client):
    """Query endpoint rejects limit=0."""
    resp = client.post("/api/v1/query", json={"text": "test", "limit": 0})
    assert resp.status_code == 422


def test_query_limit_above_max_rejected(client):
    """Query endpoint rejects limit=51."""
    resp = client.post("/api/v1/query", json={"text": "test", "limit": 51})
    assert resp.status_code == 422


def test_graph_entities_limit_zero_rejected(client):
    """Graph entities rejects limit=0."""
    resp = client.get("/api/v1/graph/entities?limit=0")
    assert resp.status_code == 422


def test_graph_entities_limit_above_max_rejected(client):
    """Graph entities rejects limit=1001."""
    resp = client.get("/api/v1/graph/entities?limit=1001")
    assert resp.status_code == 422


def test_ingest_text_whitespace_stripped(client, mock_service):
    """Ingest strips leading/trailing whitespace from text."""
    resp = client.post("/api/v1/ingest", json={"text": "  hello world  "})
    assert resp.status_code == 200
    call_text = mock_service.add_text.call_args[0][0]
    assert call_text == "hello world"


def test_ingest(client):
    """Ingest endpoint adds text and runs cognify."""
    resp = client.post("/api/v1/ingest", json={"text": "Hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ingest" in data
    assert "cognify_status" in data


def test_graph_stats(client):
    """Graph stats endpoint returns live node/edge counts."""
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == 10
    assert data["edges"] == 15
    assert data["entity_types"]["Person"] == 5


def test_graph_entities(client):
    """Graph entities endpoint returns nodes and edges."""
    resp = client.get("/api/v1/graph/entities")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["label"] == "Alice"
    assert len(data["edges"]) == 1
    assert data["edges"][0]["type"] == "WORKS_AT"


def test_graph_entities_with_filter(client, mock_graph_client):
    """Graph entities endpoint passes entity type filter."""
    client.get("/api/v1/graph/entities?entity_types=Person,Organization&limit=50")
    mock_graph_client.get_entities.assert_called_once_with(
        limit=50, entity_types=["Person", "Organization"]
    )


def test_graph_entities_rejects_invalid_type(client):
    """Graph entities endpoint rejects entity_types with special characters."""
    resp = client.get("/api/v1/graph/entities?entity_types=Person;DROP+NODE")
    assert resp.status_code == 422


def test_graph_stats_fallback(client, mock_graph_client):
    """Graph stats returns 502 when Neo4j is unavailable."""
    mock_graph_client.get_stats = AsyncMock(side_effect=Exception("Connection refused"))
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 502


def test_query_custom_mode(client, mock_service):
    """Query endpoint passes custom mode to service."""
    client.post("/api/v1/query", json={"text": "test", "mode": "RAG_COMPLETION", "limit": 10})
    mock_service.query.assert_called_once_with("test", search_type="RAG_COMPLETION", limit=10)


def test_query_invalid_mode(client):
    """Query endpoint rejects invalid search mode."""
    resp = client.post("/api/v1/query", json={"text": "test", "mode": "INVALID"})
    assert resp.status_code == 422


def test_ingest_custom_dataset(client, mock_service):
    """Ingest endpoint passes custom dataset name."""
    client.post("/api/v1/ingest", json={"text": "data", "dataset_name": "custom"})
    mock_service.add_text.assert_called_once_with("data", dataset_name="custom")
    mock_service.cognify.assert_called_once_with(dataset_name="custom")


def test_ingest_cognify_failure_returns_partial_success(client, mock_service):
    """Ingest should return 200 with cognify_status='failed' when cognify fails."""
    mock_service.add_text = AsyncMock(
        return_value={"status": "added", "chars": 5, "dataset": "main"},
    )
    mock_service.cognify = AsyncMock(side_effect=Exception("cognify timeout"))
    resp = client.post("/api/v1/ingest", json={"text": "hello", "dataset_name": "main"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cognify_status"] == "failed"
    assert "cognify timeout" in data["cognify_detail"]


def test_ingest_file_txt(client, mock_service):
    """Ingest-file endpoint accepts multipart file upload."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("test.txt", b"Hello from file", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ingest" in data
    assert "cognify_status" in data
    mock_service.add_bytes.assert_called_once_with(
        b"Hello from file", "test.txt", "main"
    )


def test_ingest_file_custom_dataset(client, mock_service):
    """Ingest-file endpoint passes custom dataset name."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("doc.pdf", b"%PDF fake", "application/pdf")},
        data={"dataset_name": "papers"},
    )
    assert resp.status_code == 200
    mock_service.add_bytes.assert_called_once_with(
        b"%PDF fake", "doc.pdf", "papers"
    )


def test_ingest_invalid_dataset_name(client):
    """Ingest endpoint rejects invalid dataset_name characters."""
    resp = client.post(
        "/api/v1/ingest", json={"text": "data", "dataset_name": "../etc/passwd"}
    )
    assert resp.status_code == 422


def test_ingest_file_invalid_dataset_name(client):
    """Ingest-file endpoint rejects invalid dataset_name."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("test.txt", b"data", "text/plain")},
        data={"dataset_name": "../../hack"},
    )
    assert resp.status_code == 422


def test_ingest_file_size_limit(client):
    """Ingest-file endpoint rejects oversized uploads."""
    big_data = b"x" * (50 * 1024 * 1024 + 1)  # 50MB + 1 byte
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("big.txt", big_data, "text/plain")},
    )
    assert resp.status_code == 413


def test_ingest_text_too_long(client):
    """Ingest endpoint rejects text exceeding max_length."""
    resp = client.post("/api/v1/ingest", json={"text": "x" * 500_001})
    assert resp.status_code == 422


def test_graph_entities_fallback(client, mock_graph_client):
    """Graph entities returns 502 when Neo4j is unavailable."""
    mock_graph_client.get_entities = AsyncMock(side_effect=Exception("Connection refused"))
    resp = client.get("/api/v1/graph/entities")
    assert resp.status_code == 502


def test_filename_sanitized(client, mock_service):
    """Ingest-file sanitizes filenames with special characters."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("../../etc/passwd.txt", b"data", "text/plain")},
    )
    assert resp.status_code == 200
    call_args = mock_service.add_bytes.call_args
    filename = call_args[0][1]
    assert "/" not in filename
    assert ".." not in filename


def test_filename_dots_only(client, mock_service):
    """Ingest-file with dots-only filename gets a safe fallback."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("...", b"data", "text/plain")},
    )
    assert resp.status_code == 200
    call_args = mock_service.add_bytes.call_args
    filename = call_args[0][1]
    assert len(filename) > 0


def test_reset(client, mock_service):
    """Reset endpoint denied without API key configured."""
    resp = client.post("/api/v1/reset", json={"confirm": True})
    assert resp.status_code == 403
    assert "API key" in resp.json()["detail"]


def test_reset_requires_confirmation(client):
    """Reset endpoint denied without API key (confirmation check unreachable)."""
    resp = client.post("/api/v1/reset", json={})
    assert resp.status_code == 403


def test_reset_rejects_false_confirmation(client):
    """Reset endpoint denied without API key (confirmation check unreachable)."""
    resp = client.post("/api/v1/reset", json={"confirm": False})
    assert resp.status_code == 403


def test_list_datasets(client, mock_service):
    """Datasets endpoint returns list of dataset names."""
    resp = client.get("/api/v1/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert data == ["main", "papers", "docs"]
    mock_service.list_datasets.assert_called_once()


def test_health_response_model(client, mock_graph_client):
    """Health endpoint returns all HealthStatus fields."""
    mock_graph_client.health_check = AsyncMock(return_value=True)
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert "status" in data
    assert "neo4j" in data
    assert "ollama" in data
    assert "lancedb" not in data  # embedded DB, no health-check


def test_health_includes_version(client):
    """Health response includes app version."""
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert "version" in data
    assert data["version"] == "0.1.0"


def test_health_includes_uptime(client):
    """Health response includes non-negative uptime_seconds."""
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_check_ollama_models_returns_availability():
    """check_ollama_models maps each model name to its availability."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from cog_rag_cognee.health import check_ollama_models

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "models": [{"name": "llama3.1:8b"}, {"name": "nomic-embed-text:latest"}]
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=fake_response)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama_models(
            "http://localhost:11434/v1",
            ["llama3.1:8b", "unknown-model:latest"],
        )

    assert isinstance(result, dict)
    assert result["llama3.1:8b"] is True
    assert result["unknown-model:latest"] is False


@pytest.mark.asyncio
async def test_check_ollama_models_non_200_returns_all_false():
    """check_ollama_models returns False for all models when status != 200."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from cog_rag_cognee.health import check_ollama_models

    fake_response = MagicMock()
    fake_response.status_code = 500

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=fake_response)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama_models(
            "http://localhost:11434/v1",
            ["llama3.1:8b", "nomic-embed-text:latest"],
        )

    assert result == {"llama3.1:8b": False, "nomic-embed-text:latest": False}


@pytest.mark.asyncio
async def test_check_ollama_models_connection_error_returns_all_false():
    """check_ollama_models returns False for all models on connection error."""
    from unittest.mock import AsyncMock, patch

    from cog_rag_cognee.health import check_ollama_models

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama_models(
            "http://localhost:11434/v1",
            ["llama3.1:8b", "nomic-embed-text:latest"],
        )

    assert result == {"llama3.1:8b": False, "nomic-embed-text:latest": False}


def test_request_id_in_response(client):
    """Every response includes X-Request-ID header."""
    resp = client.get("/api/v1/health")
    assert "x-request-id" in resp.headers


def test_custom_request_id_echoed(client):
    """Client-provided X-Request-ID is echoed back."""
    resp = client.get("/api/v1/health", headers={"X-Request-ID": "my-id-42"})
    assert resp.headers["x-request-id"] == "my-id-42"


def test_exception_handler_ingestion_error(client, mock_service):
    """IngestionError returns 502 with code and generic message."""
    from cog_rag_cognee.exceptions import IngestionError

    mock_service.add_text = AsyncMock(side_effect=IngestionError("Cognee down"))
    resp = client.post("/api/v1/ingest", json={"text": "hello"})
    assert resp.status_code == 502
    data = resp.json()
    assert data["error"] == "IngestionError"
    assert data["code"] == "ERR_INGESTION"
    # Internal details must NOT leak in production mode
    assert "Cognee down" not in data["detail"]
    assert "internal error" in data["detail"].lower()


def test_exception_handler_search_error(client, mock_service):
    """SearchError returns 502 with code and generic message."""
    from cog_rag_cognee.exceptions import SearchError

    mock_service.query = AsyncMock(side_effect=SearchError("Search failed"))
    resp = client.post("/api/v1/query", json={"text": "test"})
    assert resp.status_code == 502
    data = resp.json()
    assert data["error"] == "SearchError"
    assert data["code"] == "ERR_SEARCH"
    assert "internal error" in data["detail"].lower()


def test_validation_error_has_code(client):
    """Validation errors include structured error code."""
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422
    data = resp.json()
    assert data["code"] == "ERR_VALIDATION"
    assert "detail" in data


def test_error_code_attributes():
    """Exception classes expose error code as class attribute."""
    from cog_rag_cognee.exceptions import (
        ConfigError,
        GraphError,
        IngestionError,
        OllamaError,
        SearchError,
    )

    assert IngestionError.code == "ERR_INGESTION"
    assert SearchError.code == "ERR_SEARCH"
    assert GraphError.code == "ERR_GRAPH"
    assert ConfigError.code == "ERR_CONFIG"
    assert OllamaError.code == "ERR_OLLAMA"


def test_ingest_file_empty_rejected(client):
    """Ingest-file rejects 0-byte file uploads."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()


def test_ingest_file_reserved_filename(client, mock_service):
    """Ingest-file prefixes Windows reserved filenames."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("CON.txt", b"data", "text/plain")},
    )
    assert resp.status_code == 200
    filename = mock_service.add_bytes.call_args[0][1]
    # Reserved name must be escaped (not bare "CON")
    assert filename.split(".")[0].upper() != "CON"


def test_query_null_bytes_stripped(client, mock_service):
    """Null bytes are stripped from query text."""
    resp = client.post("/api/v1/query", json={"text": "hello\x00world"})
    assert resp.status_code == 200
    call_text = mock_service.query.call_args[0][0]
    assert "\x00" not in call_text


def test_ingest_null_bytes_stripped(client, mock_service):
    """Null bytes are stripped from ingest text."""
    resp = client.post("/api/v1/ingest", json={"text": "hello\x00world"})
    assert resp.status_code == 200
    call_text = mock_service.add_text.call_args[0][0]
    assert "\x00" not in call_text


def test_security_headers(client):
    """Every response includes security headers."""
    resp = client.get("/api/v1/liveness")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["cache-control"] == "no-store"


def test_exception_handler_shows_details_in_debug(mock_service, mock_graph_client):
    """Error handler shows full details when DEBUG=true."""
    import os
    from unittest.mock import AsyncMock

    from api.deps import set_graph_client, set_service
    from cog_rag_cognee.exceptions import IngestionError

    mock_service.add_text = AsyncMock(
        side_effect=IngestionError("neo4j://localhost:7687 connection refused")
    )
    set_service(mock_service)
    set_graph_client(mock_graph_client)

    old_debug = os.environ.get("DEBUG")
    os.environ["DEBUG"] = "true"
    try:
        # Must recreate app to pick up new settings
        from cog_rag_cognee.config import get_settings
        get_settings.cache_clear()

        from api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/ingest", json={"text": "hello"})
        assert resp.status_code == 502
        assert "neo4j://" in resp.json()["detail"]
    finally:
        if old_debug is None:
            os.environ.pop("DEBUG", None)
        else:
            os.environ["DEBUG"] = old_debug
        get_settings.cache_clear()
        set_service(None)
        set_graph_client(None)


def test_exception_handler_shows_caused_by_in_debug(mock_service, mock_graph_client):
    """In debug mode, caused_by field contains the original exception message."""
    import os
    from unittest.mock import AsyncMock

    from api.deps import set_graph_client, set_service
    from cog_rag_cognee.exceptions import IngestionError

    cause = RuntimeError("disk full")
    wrapped = IngestionError("Failed to add text: disk full")
    wrapped.__cause__ = cause
    mock_service.add_text = AsyncMock(side_effect=wrapped)
    set_service(mock_service)
    set_graph_client(mock_graph_client)

    old_debug = os.environ.get("DEBUG")
    os.environ["DEBUG"] = "true"
    try:
        from cog_rag_cognee.config import get_settings
        get_settings.cache_clear()

        from api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/ingest", json={"text": "hello"})
        assert resp.status_code == 502
        data = resp.json()
        assert "caused_by" in data
        assert "disk full" in data["caused_by"]
    finally:
        if old_debug is None:
            os.environ.pop("DEBUG", None)
        else:
            os.environ["DEBUG"] = old_debug
        get_settings.cache_clear()
        set_service(None)
        set_graph_client(None)


def test_exception_handler_no_caused_by_in_prod(client, mock_service):
    """In non-debug (production) mode, caused_by field is absent from response."""
    from cog_rag_cognee.exceptions import IngestionError

    cause = RuntimeError("disk full")
    wrapped = IngestionError("Failed to add text: disk full")
    wrapped.__cause__ = cause
    mock_service.add_text = AsyncMock(side_effect=wrapped)
    resp = client.post("/api/v1/ingest", json={"text": "hello"})
    assert resp.status_code == 502
    data = resp.json()
    assert "caused_by" not in data


def test_exception_handler_no_caused_by_when_no_cause_in_debug(mock_service, mock_graph_client):
    """In debug mode, caused_by is absent when there is no __cause__."""
    import os
    from unittest.mock import AsyncMock

    from api.deps import set_graph_client, set_service
    from cog_rag_cognee.exceptions import IngestionError

    mock_service.add_text = AsyncMock(side_effect=IngestionError("standalone error"))
    set_service(mock_service)
    set_graph_client(mock_graph_client)

    old_debug = os.environ.get("DEBUG")
    os.environ["DEBUG"] = "true"
    try:
        from cog_rag_cognee.config import get_settings
        get_settings.cache_clear()

        from api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            resp = c.post("/api/v1/ingest", json={"text": "hello"})
        assert resp.status_code == 502
        data = resp.json()
        assert "caused_by" not in data
    finally:
        if old_debug is None:
            os.environ.pop("DEBUG", None)
        else:
            os.environ["DEBUG"] = old_debug
        get_settings.cache_clear()
        set_service(None)
        set_graph_client(None)


def test_startup_refuses_without_api_key_in_prod(monkeypatch):
    """In non-debug mode, startup must fail if API_KEY is empty and ALLOW_ANONYMOUS is false."""
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "false")
    from cog_rag_cognee.config import get_settings
    get_settings.cache_clear()

    from api.app import create_app
    from api.deps import set_graph_client, set_service

    set_service(MagicMock())
    set_graph_client(MagicMock())

    app = create_app()
    with pytest.raises(Exception, match="API_KEY.*required"), TestClient(app):
        pass

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


def test_startup_allows_anonymous_when_explicit(monkeypatch, mock_service, mock_graph_client):
    """When ALLOW_ANONYMOUS=true, startup should succeed without API_KEY."""
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "true")
    from cog_rag_cognee.config import get_settings
    get_settings.cache_clear()

    from api.app import create_app
    from api.deps import set_graph_client, set_service

    set_service(mock_service)
    set_graph_client(mock_graph_client)

    app = create_app()
    with TestClient(app) as c:
        resp = c.get("/api/v1/liveness")
        assert resp.status_code == 200

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


def test_graph_stats_returns_502_on_error(client, mock_graph_client):
    """Graph stats should return 502, not empty 200, when Neo4j fails."""
    mock_graph_client.get_stats = AsyncMock(side_effect=ConnectionError("Neo4j down"))
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 502


def test_graph_entities_returns_502_on_error(client, mock_graph_client):
    """Graph entities should return 502, not empty 200, when Neo4j fails."""
    mock_graph_client.get_entities = AsyncMock(side_effect=ConnectionError("Neo4j down"))
    resp = client.get("/api/v1/graph/entities")
    assert resp.status_code == 502


def test_reset_denied_in_anonymous_mode(monkeypatch, mock_service, mock_graph_client):
    """Even with ALLOW_ANONYMOUS=true, /reset must be denied without API key."""
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("DEBUG", "false")
    from cog_rag_cognee.config import get_settings
    get_settings.cache_clear()

    from api.app import create_app
    from api.deps import set_graph_client, set_service

    set_service(mock_service)
    set_graph_client(mock_graph_client)

    app = create_app()
    with TestClient(app) as c:
        resp = c.post("/api/v1/reset", json={"confirm": True})
        assert resp.status_code == 403
        assert "API key" in resp.json()["detail"]

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


def test_reset_allowed_with_api_key(monkeypatch, mock_service, mock_graph_client):
    """With API key set, /reset should work."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "false")
    from cog_rag_cognee.config import get_settings
    get_settings.cache_clear()

    from api.app import create_app
    from api.deps import set_graph_client, set_service

    set_service(mock_service)
    set_graph_client(mock_graph_client)

    app = create_app()
    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/reset",
            json={"confirm": True},
            headers={"X-API-Key": "secret-key-123"},
        )
        assert resp.status_code == 200

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


def test_request_metrics_logged(client, caplog):
    """Every request should log method, path, status, and duration."""
    with caplog.at_level(logging.INFO, logger="api.app"):
        client.get("/api/v1/liveness")

    metrics_logs = [r for r in caplog.records if "GET /api/v1/liveness" in r.getMessage()]
    assert len(metrics_logs) >= 1
    msg = metrics_logs[0].getMessage()
    assert "200" in msg
    assert "ms" in msg


def test_log_records_contain_request_id(client, caplog):
    """Log records should include the request ID from the response header."""
    from cog_rag_cognee.logging_config import _RequestIdFilter

    # Install the filter on the api.app logger so caplog records receive request_id.
    api_logger = logging.getLogger("api.app")
    filt = _RequestIdFilter()
    api_logger.addFilter(filt)
    try:
        with caplog.at_level(logging.INFO):
            resp = client.get("/api/v1/liveness")
    finally:
        api_logger.removeFilter(filt)

    request_id = resp.headers["X-Request-ID"]
    # _RequestIdFilter injects request_id as a record attribute (not in getMessage()).
    # Verify at least one INFO record from api.app carries the correct request_id.
    api_records = [r for r in caplog.records if r.name == "api.app"]
    assert api_records, "Expected at least one log record from api.app"
    assert any(getattr(r, "request_id", None) == request_id for r in api_records)


def test_validation_error_still_gets_security_headers(client):
    """Validation errors processed through middleware still get security headers."""
    resp = client.post("/api/v1/query", json={})  # triggers 422
    assert resp.status_code == 422
    # Verify middleware ran (security headers applied even to error responses)
    assert "x-request-id" in resp.headers
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("cache-control") == "no-store"


def test_cors_wildcard_rejected_in_prod(monkeypatch, mock_service, mock_graph_client):
    """CORS '*' should be stripped in non-debug mode."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "false")
    from cog_rag_cognee.config import get_settings

    get_settings.cache_clear()

    from api.app import create_app
    from api.deps import set_graph_client, set_service

    set_service(mock_service)
    set_graph_client(mock_graph_client)

    app = create_app()
    with TestClient(app) as c:
        resp = c.options(
            "/api/v1/health",
            headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"},
        )
        # Wildcard stripped — evil.com should NOT be reflected
        assert resp.headers.get("access-control-allow-origin") != "*"

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


def test_ingest_file_cognify_failure_returns_partial_success(client, mock_service):
    """Ingest-file returns 200 with cognify_status='failed' when cognify raises."""
    mock_service.add_bytes = AsyncMock(
        return_value={"status": "added", "file": "test.txt", "chars": 12},
    )
    mock_service.cognify = AsyncMock(side_effect=Exception("cognify boom"))
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("test.txt", b"some content", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cognify_status"] == "failed"
    assert "cognify boom" in data["cognify_detail"]


def test_neo4j_timeout_lower_bound():
    """neo4j_timeout=1 is the minimum valid value."""
    from cog_rag_cognee.config import Settings

    s = Settings(neo4j_timeout=1)
    assert s.neo4j_timeout == 1


def test_neo4j_timeout_zero_rejected():
    """neo4j_timeout=0 is below minimum."""
    from pydantic import ValidationError

    from cog_rag_cognee.config import Settings

    with pytest.raises(ValidationError, match="neo4j_timeout"):
        Settings(neo4j_timeout=0)


def test_neo4j_timeout_upper_bound():
    """neo4j_timeout=300 is the maximum valid value."""
    from cog_rag_cognee.config import Settings

    s = Settings(neo4j_timeout=300)
    assert s.neo4j_timeout == 300


def test_neo4j_timeout_above_max_rejected():
    """neo4j_timeout=301 exceeds maximum."""
    from pydantic import ValidationError

    from cog_rag_cognee.config import Settings

    with pytest.raises(ValidationError, match="neo4j_timeout"):
        Settings(neo4j_timeout=301)


def test_cognee_timeout_lower_bound():
    """cognee_timeout=10 is the minimum valid value."""
    from cog_rag_cognee.config import Settings

    s = Settings(cognee_timeout=10)
    assert s.cognee_timeout == 10


def test_cognee_timeout_below_min_rejected():
    """cognee_timeout=9 is below minimum."""
    from pydantic import ValidationError

    from cog_rag_cognee.config import Settings

    with pytest.raises(ValidationError, match="cognee_timeout"):
        Settings(cognee_timeout=9)


def test_max_upload_bytes_zero_rejected():
    """max_upload_bytes=0 is not positive."""
    from pydantic import ValidationError

    from cog_rag_cognee.config import Settings

    with pytest.raises(ValidationError, match="max_upload_bytes"):
        Settings(max_upload_bytes=0)
