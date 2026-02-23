"""Tests for apply_cognee_env — verifies env vars are set."""
from __future__ import annotations

import os

import pytest

from cog_rag_cognee.cognee_setup import apply_cognee_env, build_cognee_env
from cog_rag_cognee.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_cognee_env():
    """Restore env vars and settings cache after each test."""
    env_keys = list(build_cognee_env(Settings()).keys())
    saved = {k: os.environ.get(k) for k in env_keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()


def test_apply_cognee_env_sets_os_environ():
    """apply_cognee_env() sets expected env vars in os.environ."""
    settings = Settings()
    apply_cognee_env(settings)

    assert os.environ["LLM_PROVIDER"] == "ollama"
    assert os.environ["LLM_MODEL"] == "llama3.1:8b"
    assert os.environ["GRAPH_DATABASE_PROVIDER"] == "neo4j"
    assert os.environ["VECTOR_DB_PROVIDER"] == "lancedb"
    assert os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] == "false"


def test_apply_cognee_env_custom_values():
    """apply_cognee_env() propagates custom settings values."""
    settings = Settings(llm_model="gemma2:2b", vector_db_provider="qdrant")
    apply_cognee_env(settings)

    assert os.environ["LLM_MODEL"] == "gemma2:2b"
    assert os.environ["VECTOR_DB_PROVIDER"] == "qdrant"
