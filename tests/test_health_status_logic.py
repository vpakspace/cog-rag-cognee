"""Tests for health status logic — both services considered."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _mock_deps():
    """Set up mocked service and graph client."""
    from api.deps import set_graph_client, set_service

    svc = MagicMock()
    gc = MagicMock()
    gc.close = AsyncMock()

    set_service(svc)
    set_graph_client(gc)
    yield gc
    set_service(None)
    set_graph_client(None)


def _make_client():
    from api.app import create_app

    app = create_app()
    return TestClient(app)


def test_health_both_up(_mock_deps):
    """Both Neo4j and Ollama up → status 'ok'."""
    gc = _mock_deps
    gc.health_check = AsyncMock(return_value=True)
    with patch("api.routes.check_ollama", new_callable=AsyncMock, return_value=True):
        with _make_client() as c:
            resp = c.get("/api/v1/health")
    data = resp.json()
    assert data["status"] == "ok"
    assert data["neo4j"] is True
    assert data["ollama"] is True


def test_health_neo4j_down_ollama_up(_mock_deps):
    """Neo4j down, Ollama up → status 'degraded'."""
    gc = _mock_deps
    gc.health_check = AsyncMock(return_value=False)
    with patch("api.routes.check_ollama", new_callable=AsyncMock, return_value=True):
        with _make_client() as c:
            resp = c.get("/api/v1/health")
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is False
    assert data["ollama"] is True


def test_health_neo4j_up_ollama_down(_mock_deps):
    """Neo4j up, Ollama down → status 'degraded'."""
    gc = _mock_deps
    gc.health_check = AsyncMock(return_value=True)
    with patch("api.routes.check_ollama", new_callable=AsyncMock, return_value=False):
        with _make_client() as c:
            resp = c.get("/api/v1/health")
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is True
    assert data["ollama"] is False


def test_health_both_down(_mock_deps):
    """Both services down → status 'degraded'."""
    gc = _mock_deps
    gc.health_check = AsyncMock(return_value=False)
    with patch("api.routes.check_ollama", new_callable=AsyncMock, return_value=False):
        with _make_client() as c:
            resp = c.get("/api/v1/health")
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is False
    assert data["ollama"] is False
