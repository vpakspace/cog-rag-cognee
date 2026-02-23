"""Tests for FastAPI application."""
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


def test_health(client, mock_graph_client):
    """Health endpoint checks Neo4j and Ollama connectivity."""
    mock_graph_client.health_check = AsyncMock(return_value=True)
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


def test_ingest(client):
    """Ingest endpoint adds text and runs cognify."""
    resp = client.post("/api/v1/ingest", json={"text": "Hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ingest" in data
    assert "cognify" in data


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
    """Graph stats returns zeros when Neo4j is unavailable."""
    mock_graph_client.get_stats = AsyncMock(side_effect=Exception("Connection refused"))
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == 0


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


def test_ingest_file_txt(client, mock_service):
    """Ingest-file endpoint accepts multipart file upload."""
    resp = client.post(
        "/api/v1/ingest-file",
        files={"file": ("test.txt", b"Hello from file", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ingest" in data
    assert "cognify" in data
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
    """Graph entities returns empty lists when Neo4j is unavailable."""
    mock_graph_client.get_entities = AsyncMock(side_effect=Exception("Connection refused"))
    resp = client.get("/api/v1/graph/entities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []


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
    """Reset endpoint calls service.reset() with confirmation."""
    resp = client.post("/api/v1/reset", json={"confirm": True})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mock_service.reset.assert_called_once()


def test_reset_requires_confirmation(client):
    """Reset endpoint rejects requests without confirm=true."""
    resp = client.post("/api/v1/reset", json={})
    assert resp.status_code == 400
    assert "confirm" in resp.json()["detail"].lower()


def test_reset_rejects_false_confirmation(client):
    """Reset endpoint rejects confirm=false."""
    resp = client.post("/api/v1/reset", json={"confirm": False})
    assert resp.status_code == 400


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


def test_exception_handler_ingestion_error(client, mock_service):
    """IngestionError returns 502 with generic message (DEBUG=false)."""
    from cog_rag_cognee.exceptions import IngestionError

    mock_service.add_text = AsyncMock(side_effect=IngestionError("Cognee down"))
    resp = client.post("/api/v1/ingest", json={"text": "hello"})
    assert resp.status_code == 502
    data = resp.json()
    assert data["error"] == "IngestionError"
    # Internal details must NOT leak in production mode
    assert "Cognee down" not in data["detail"]
    assert "internal error" in data["detail"].lower()


def test_exception_handler_search_error(client, mock_service):
    """SearchError returns 502 with generic message (DEBUG=false)."""
    from cog_rag_cognee.exceptions import SearchError

    mock_service.query = AsyncMock(side_effect=SearchError("Search failed"))
    resp = client.post("/api/v1/query", json={"text": "test"})
    assert resp.status_code == 502
    data = resp.json()
    assert data["error"] == "SearchError"
    assert "internal error" in data["detail"].lower()


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
