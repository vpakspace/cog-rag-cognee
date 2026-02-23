"""Shared health-check utilities for external services."""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


async def check_ollama(llm_endpoint: str) -> bool:
    """Return True if Ollama is reachable via /api/tags.

    Args:
        llm_endpoint: The LLM endpoint URL (e.g. ``http://localhost:11434/v1``).
    """
    base = llm_endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{base}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
