"""Tests for configuration module."""

from cog_rag_cognee.config import Settings, get_settings


def test_settings_defaults():
    """Verify default values for all key settings."""
    s = Settings()
    assert s.llm_model == "llama3.1:8b"
    assert s.embedding_model == "nomic-embed-text:latest"
    assert s.embedding_dimensions == 768
    assert s.graph_database_provider == "neo4j"
    assert s.vector_db_provider == "lancedb"
    assert s.api_port == 8508
    assert s.ui_port == 8506


def test_settings_singleton():
    """get_settings() returns the same cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_ollama_base_url():
    """ollama_base_url property derives from llm_endpoint."""
    s = Settings()
    assert s.ollama_base_url == s.llm_endpoint
    assert s.ollama_base_url == "http://localhost:11434/v1"


def test_docling_use_gpu_default():
    """docling_use_gpu defaults to False."""
    s = Settings()
    assert s.docling_use_gpu is False
