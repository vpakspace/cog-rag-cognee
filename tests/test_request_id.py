"""Tests for X-Request-ID middleware."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked deps."""
    from api.deps import set_graph_client, set_service

    svc = MagicMock()
    gc = MagicMock()
    gc.health_check = AsyncMock(return_value=True)
    gc.close = AsyncMock()

    set_service(svc)
    set_graph_client(gc)

    from api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    set_service(None)
    set_graph_client(None)


def test_response_has_request_id(client):
    """Every response should include X-Request-ID header."""
    resp = client.get("/api/v1/health")
    assert "x-request-id" in resp.headers
    # Should be a valid UUID
    uuid.UUID(resp.headers["x-request-id"])


def test_request_id_unique(client):
    """Each request gets a unique X-Request-ID."""
    resp1 = client.get("/api/v1/health")
    resp2 = client.get("/api/v1/health")
    assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


def test_request_id_passthrough(client):
    """Client-provided X-Request-ID should be preserved."""
    custom_id = "my-trace-id-12345"
    resp = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert resp.headers["x-request-id"] == custom_id
