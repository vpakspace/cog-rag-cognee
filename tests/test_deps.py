"""Tests for dependency injection and API key validation."""
from unittest.mock import patch

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
