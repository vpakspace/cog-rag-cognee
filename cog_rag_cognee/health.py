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


async def check_ollama_models(llm_endpoint: str, required_models: list[str]) -> dict[str, bool]:
    """Check which required models are available in Ollama.

    Returns dict mapping model name to availability (True/False).
    """
    base = llm_endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base}/api/tags")
            if resp.status_code != 200:
                return {m: False for m in required_models}
            available = {
                m.get("name") for m in resp.json().get("models", []) if m.get("name")
            }
            return {m: m in available for m in required_models}
    except Exception:
        return {m: False for m in required_models}
