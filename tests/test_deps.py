"""Tests for dependency injection and API key validation."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.deps import verify_api_key


@pytest.mark.asyncio
async def test_verify_api_key_disabled():
    """When API_KEY is empty, auth is skipped."""
    with patch("api.deps.get_settings") as mock:
        mock.return_value.api_key = ""
        result = await verify_api_key(api_key="anything")
        assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_valid():
    """Valid API key returns the key."""
    with patch("api.deps.get_settings") as mock:
        mock.return_value.api_key = "secret-key-12345"
        result = await verify_api_key(api_key="secret-key-12345")
        assert result == "secret-key-12345"


@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    """Invalid API key raises 401."""
    with patch("api.deps.get_settings") as mock:
        mock.return_value.api_key = "secret-key-12345"
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="wrong-key")
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_none():
    """Missing API key (None) raises 401 when auth is enabled."""
    with patch("api.deps.get_settings") as mock:
        mock.return_value.api_key = "secret-key-12345"
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key=None)
        assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_uses_constant_time_comparison():
    """API key comparison must use hmac.compare_digest."""
    with patch("api.deps.get_settings") as mock_settings, \
         patch("api.deps.hmac") as mock_hmac:
        mock_settings.return_value.api_key = "secret"
        mock_hmac.compare_digest.return_value = True
        result = await verify_api_key(api_key="secret")
        mock_hmac.compare_digest.assert_called_once_with("secret", "secret")
        assert result == "secret"


def test_get_service_lazy_init():
    """get_service() creates PipelineService when global is None."""
    import api.deps as deps_mod

    deps_mod._service = None
    try:
        with patch.object(deps_mod, "PipelineService") as mock_cls:
            mock_cls.return_value = MagicMock()
            svc = deps_mod.get_service()
            mock_cls.assert_called_once()
            assert svc is mock_cls.return_value
    finally:
        deps_mod._service = None


def test_get_graph_client_lazy_init():
    """get_graph_client() creates GraphClient from settings when global is None."""
    import api.deps as deps_mod

    deps_mod._graph_client = None
    try:
        with patch.object(deps_mod, "GraphClient") as mock_cls, \
             patch.object(deps_mod, "get_settings") as mock_settings:
            mock_settings.return_value.graph_database_url = "neo4j://localhost:7687"
            mock_settings.return_value.graph_database_username = "neo4j"
            mock_settings.return_value.graph_database_password = "password"
            mock_cls.return_value = MagicMock()
            gc = deps_mod.get_graph_client()
            mock_cls.assert_called_once_with(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
            )
            assert gc is mock_cls.return_value
    finally:
        deps_mod._graph_client = None
