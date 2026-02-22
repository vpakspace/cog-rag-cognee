"""Tests for FastAPI application."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_service():
    """Create a mock PipelineService."""
    svc = MagicMock()
    svc.query = AsyncMock(
        return_value=MagicMock(
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
        )
    )
    svc.search = AsyncMock(return_value=[])
    svc.add_text = AsyncMock(return_value={"status": "added", "chars": 10, "dataset": "main"})
    svc.add_bytes = AsyncMock(return_value={"status": "added", "file": "test.txt", "chars": 12})
    svc.cognify = AsyncMock(return_value={"main": "completed"})
    return svc


@pytest.fixture
def mock_graph_client():
    """Create a mock GraphClient."""
    gc = MagicMock()
    gc.get_stats.return_value = {
        "nodes": 10,
        "edges": 15,
        "entity_types": {"Person": 5, "Organization": 3, "Location": 2},
    }
    gc.get_entities.return_value = [
        {"id": 1, "label": "Alice", "type": "Person"},
        {"id": 2, "label": "Acme Corp", "type": "Organization"},
    ]
    gc.get_relationships.return_value = [
        {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT"},
    ]
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


def test_health(client):
    """Health endpoint returns ok."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


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


def test_graph_stats_fallback(client, mock_graph_client):
    """Graph stats returns zeros when Neo4j is unavailable."""
    mock_graph_client.get_stats.side_effect = Exception("Connection refused")
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == 0


def test_query_custom_mode(client, mock_service):
    """Query endpoint passes custom mode to service."""
    client.post("/api/v1/query", json={"text": "test", "mode": "INSIGHTS", "limit": 10})
    mock_service.query.assert_called_once_with("test", search_type="INSIGHTS", limit=10)


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
