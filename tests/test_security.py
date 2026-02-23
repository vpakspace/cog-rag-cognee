"""Security regression tests — Sprint 12."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _secured_app(monkeypatch):
    """Create a test app with API key enforced."""
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("ALLOW_ANONYMOUS", "false")
    from cog_rag_cognee.config import get_settings
    get_settings.cache_clear()

    svc = MagicMock()
    svc.query = AsyncMock(
        return_value={"answer": "ok", "confidence": 1.0, "sources": [], "mode": "default"}
    )
    svc.search = AsyncMock(return_value=[])
    svc.add_text = AsyncMock(return_value={"status": "added", "chars": 5, "dataset": "main"})
    svc.cognify = AsyncMock(return_value={})
    svc.reset = AsyncMock()
    svc.list_datasets = AsyncMock(return_value=[])

    gc = MagicMock()
    gc.get_stats = AsyncMock(return_value={"nodes": 0, "edges": 0, "entity_types": {}})
    gc.get_entities = AsyncMock(return_value=[])
    gc.get_relationships = AsyncMock(return_value=[])
    gc.health_check = AsyncMock(return_value=True)
    gc.close = AsyncMock()

    from api.deps import set_graph_client, set_service
    set_service(svc)
    set_graph_client(gc)

    from api.app import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c, svc

    set_service(None)
    set_graph_client(None)
    get_settings.cache_clear()


class TestAPIKeyEnforcement:
    """All protected endpoints must reject unauthenticated requests."""

    PROTECTED_ENDPOINTS = [
        ("POST", "/api/v1/query", {"text": "test"}),
        ("POST", "/api/v1/search", {"text": "test"}),
        ("POST", "/api/v1/ingest", {"text": "test data", "dataset_name": "main"}),
        ("GET", "/api/v1/datasets", None),
        ("GET", "/api/v1/graph/stats", None),
        ("GET", "/api/v1/graph/entities", None),
        ("POST", "/api/v1/reset", {"confirm": True}),
        ("GET", "/api/v1/health", None),
        ("GET", "/api/v1/readiness", None),
    ]

    @pytest.mark.parametrize("method,path,body", PROTECTED_ENDPOINTS)
    def test_endpoint_requires_api_key(self, _secured_app, method, path, body):
        client, _ = _secured_app
        resp = client.get(path) if method == "GET" else client.post(path, json=body)
        assert resp.status_code == 401, f"{method} {path} should require API key"

    @pytest.mark.parametrize("method,path,body", PROTECTED_ENDPOINTS[:3])
    def test_endpoint_accepts_valid_key(self, _secured_app, method, path, body):
        client, _ = _secured_app
        headers = {"X-API-Key": "test-secret-key"}
        if method == "GET":
            resp = client.get(path, headers=headers)
        else:
            resp = client.post(path, json=body, headers=headers)
        assert resp.status_code != 401, f"{method} {path} should accept valid key"


class TestResetProtection:
    """Reset endpoint has additional protections."""

    def test_reset_requires_confirm(self, _secured_app):
        client, _ = _secured_app
        resp = client.post(
            "/api/v1/reset",
            json={"confirm": False},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 400


class TestSecurityHeaders:
    """Every response must include security headers."""

    def test_security_headers_present(self, _secured_app):
        client, _ = _secured_app
        resp = client.get("/api/v1/liveness", headers={"X-API-Key": "test-secret-key"})
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Cache-Control") == "no-store"
        assert "X-Request-ID" in resp.headers
