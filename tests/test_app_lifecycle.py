"""Tests for app lifespan — startup/shutdown edge cases."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _setup_deps():
    """Provide mocked deps for lifespan tests."""
    from api.deps import set_graph_client, set_service

    svc = MagicMock()
    gc = MagicMock()
    gc.health_check = AsyncMock(return_value=True)
    gc.close = AsyncMock()

    set_service(svc)
    set_graph_client(gc)
    yield gc
    set_service(None)
    set_graph_client(None)


def test_startup_neo4j_unreachable(_setup_deps):
    """Startup logs warning when Neo4j is unreachable (doesn't crash)."""
    gc = _setup_deps
    gc.health_check = AsyncMock(return_value=False)

    with patch("api.app.check_ollama", new_callable=AsyncMock, return_value=True):
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
        assert resp.status_code == 200


def test_startup_ollama_unreachable(_setup_deps):
    """Startup logs warning when Ollama is unreachable (doesn't crash)."""
    gc = _setup_deps
    gc.health_check = AsyncMock(return_value=True)

    with patch("api.app.check_ollama", new_callable=AsyncMock, return_value=False):
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
        assert resp.status_code == 200


def test_startup_neo4j_exception(_setup_deps):
    """Startup handles Neo4j exception gracefully."""
    gc = _setup_deps
    gc.health_check = AsyncMock(side_effect=Exception("connection refused"))

    with patch("api.app.check_ollama", new_callable=AsyncMock, return_value=True):
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
        assert resp.status_code == 200


def test_shutdown_close_exception(_setup_deps):
    """Shutdown handles GraphClient.close() exception gracefully."""
    gc = _setup_deps
    gc.close = AsyncMock(side_effect=Exception("already closed"))

    with patch("api.app.check_ollama", new_callable=AsyncMock, return_value=True):
        from api.app import create_app

        app = create_app()
        # TestClient __exit__ triggers shutdown — should not raise
        with TestClient(app):
            pass
