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
    svc.cognify = AsyncMock(return_value={"main": "completed"})
    return svc


@pytest.fixture
def client(mock_service):
    """Create test client with mocked service."""
    from api.deps import set_service

    set_service(mock_service)

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    # Cleanup
    set_service(None)


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
    """Graph stats endpoint returns node/edge counts."""
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "entity_types" in data


def test_query_custom_mode(client, mock_service):
    """Query endpoint passes custom mode to service."""
    client.post("/api/v1/query", json={"text": "test", "mode": "INSIGHTS", "limit": 10})
    mock_service.query.assert_called_once_with("test", search_type="INSIGHTS", limit=10)


def test_ingest_custom_dataset(client, mock_service):
    """Ingest endpoint passes custom dataset name."""
    client.post("/api/v1/ingest", json={"text": "data", "dataset_name": "custom"})
    mock_service.add_text.assert_called_once_with("data", dataset_name="custom")
    mock_service.cognify.assert_called_once_with(dataset_name="custom")
