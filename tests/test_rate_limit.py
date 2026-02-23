"""Tests for rate limiting on API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from cog_rag_cognee.models import QAResult


@pytest.fixture
def rate_limited_client():
    """Create test client with a tight rate limit (2/minute) to test 429."""
    from api.deps import set_graph_client, set_service

    svc = MagicMock()
    svc.query = AsyncMock(
        return_value=QAResult(answer="ok", confidence=0.5, sources=[], mode="CHUNKS")
    )
    svc.search = AsyncMock(return_value=[])
    svc.add_text = AsyncMock(return_value={"status": "added", "chars": 5, "dataset": "main"})
    svc.cognify = AsyncMock(return_value={"main": "completed"})

    gc = MagicMock()
    gc.health_check = AsyncMock(return_value=True)
    gc.get_stats = AsyncMock(return_value={"nodes": 0, "edges": 0, "entity_types": {}})
    gc.close = AsyncMock()

    set_service(svc)
    set_graph_client(gc)

    # Temporarily patch the limiter with a very tight limit
    import api.app as app_module

    original_limiter = app_module.limiter
    tight_limiter = Limiter(key_func=get_remote_address, default_limits=["2/minute"])
    app_module.limiter = tight_limiter

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    app_module.limiter = original_limiter
    set_service(None)
    set_graph_client(None)


def test_rate_limit_returns_429(rate_limited_client):
    """Exceeding rate limit should return 429 Too Many Requests."""
    # First 2 requests should succeed (limit is 2/minute)
    for _ in range(2):
        resp = rate_limited_client.get("/api/v1/health")
        assert resp.status_code == 200

    # Third request should be rate limited
    resp = rate_limited_client.get("/api/v1/health")
    assert resp.status_code == 429


def test_rate_limit_under_threshold(rate_limited_client):
    """Requests within rate limit should succeed."""
    resp = rate_limited_client.post("/api/v1/query", json={"text": "test"})
    assert resp.status_code == 200
