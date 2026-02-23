"""Tests for shared health-check utilities."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cog_rag_cognee.health import check_ollama


@pytest.mark.asyncio
async def test_check_ollama_reachable():
    """Returns True when Ollama responds 200."""
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama("http://localhost:11434/v1")

    assert result is True
    mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")


@pytest.mark.asyncio
async def test_check_ollama_unreachable():
    """Returns False when Ollama connection fails."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama("http://localhost:11434/v1")

    assert result is False


@pytest.mark.asyncio
async def test_check_ollama_non_200():
    """Returns False when Ollama responds with non-200."""
    mock_resp = MagicMock(status_code=500)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_ollama("http://localhost:11434/v1")

    assert result is False


@pytest.mark.asyncio
async def test_check_ollama_strips_v1_suffix():
    """URL with /v1 suffix is correctly stripped."""
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        await check_ollama("http://localhost:11434/v1/")

    mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")


@pytest.mark.asyncio
async def test_check_ollama_no_v1_suffix():
    """URL without /v1 suffix is used as-is."""
    mock_resp = MagicMock(status_code=200)
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("cog_rag_cognee.health.httpx.AsyncClient", return_value=mock_client):
        await check_ollama("http://localhost:11434")

    mock_client.get.assert_called_once_with("http://localhost:11434/api/tags")
