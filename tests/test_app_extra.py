"""Additional tests for edge cases and improved coverage."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from cog_rag_cognee.models import QAResult


@pytest.fixture
def _mock_deps():
    """Set up mocked service and graph client."""
    from api.deps import set_graph_client, set_service

    svc = MagicMock()
    svc.query = AsyncMock(
        return_value=QAResult(answer="ok", confidence=0.5, sources=[], mode="CHUNKS")
    )
    svc.search = AsyncMock(return_value=[])
    svc.add_text = AsyncMock(return_value={"status": "added", "chars": 5, "dataset": "main"})
    svc.cognify = AsyncMock(return_value={"main": "completed"})
    svc.reset = AsyncMock()

    gc = MagicMock()
    gc.health_check = AsyncMock(return_value=True)
    gc.get_stats = AsyncMock(return_value={"nodes": 0, "edges": 0, "entity_types": {}})
    gc.close = AsyncMock()

    set_service(svc)
    set_graph_client(gc)
    yield svc, gc
    set_service(None)
    set_graph_client(None)


def test_cors_multi_origin(_mock_deps):
    """CORS allows multiple origins when configured."""
    old = os.environ.get("CORS_ORIGINS")
    os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://localhost:8506"
    from cog_rag_cognee.config import get_settings

    get_settings.cache_clear()
    try:
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.options(
                "/api/v1/health",
                headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
            )
            assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    finally:
        if old is None:
            os.environ.pop("CORS_ORIGINS", None)
        else:
            os.environ["CORS_ORIGINS"] = old
        get_settings.cache_clear()


def test_base_cograg_error_returns_500(_mock_deps):
    """CogRagError (not IngestionError/SearchError) returns 500."""
    from cog_rag_cognee.exceptions import CogRagError

    svc, _ = _mock_deps
    svc.add_text = AsyncMock(side_effect=CogRagError("generic error"))

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        resp = c.post("/api/v1/ingest", json={"text": "test"})
    assert resp.status_code == 500
    assert resp.json()["error"] == "CogRagError"


def test_api_key_required_when_set(_mock_deps):
    """Requests without X-API-Key header are rejected when API_KEY is set."""
    old = os.environ.get("API_KEY")
    os.environ["API_KEY"] = "secret-key-123"
    from cog_rag_cognee.config import get_settings

    get_settings.cache_clear()
    try:
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/health")
        assert resp.status_code == 401
    finally:
        if old is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = old
        get_settings.cache_clear()


def test_api_key_valid_passes(_mock_deps):
    """Requests with correct X-API-Key header pass authentication."""
    old = os.environ.get("API_KEY")
    os.environ["API_KEY"] = "secret-key-123"
    from cog_rag_cognee.config import get_settings

    get_settings.cache_clear()
    try:
        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            resp = c.get("/api/v1/health", headers={"X-API-Key": "secret-key-123"})
        assert resp.status_code == 200
    finally:
        if old is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = old
        get_settings.cache_clear()
