"""Tests for rate limiting on API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

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

    # Use config to set a tight rate limit
    import os

    original = os.environ.get("RATE_LIMIT_PER_MINUTE")
    os.environ["RATE_LIMIT_PER_MINUTE"] = "2"

    # Clear settings cache so new value is picked up
    from cog_rag_cognee.config import get_settings

    get_settings.cache_clear()

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    if original is not None:
        os.environ["RATE_LIMIT_PER_MINUTE"] = original
    else:
        os.environ.pop("RATE_LIMIT_PER_MINUTE", None)
    get_settings.cache_clear()
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
