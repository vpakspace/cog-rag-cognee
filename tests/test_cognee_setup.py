"""Tests for Cognee SDK configuration."""


def test_build_env_dict():
    """build_cognee_env() should produce correct env vars."""
    from cog_rag_cognee.cognee_setup import build_cognee_env
    from cog_rag_cognee.config import Settings

    settings = Settings()
    env = build_cognee_env(settings)

    assert env["LLM_PROVIDER"] == "ollama"
    assert env["LLM_MODEL"] == "llama3.1:8b"
    assert env["EMBEDDING_PROVIDER"] == "ollama"
    assert env["EMBEDDING_DIMENSIONS"] == "768"
    assert env["GRAPH_DATABASE_PROVIDER"] == "neo4j"
    assert env["VECTOR_DB_PROVIDER"] == "lancedb"


def test_build_env_custom_settings():
    """Custom settings should be reflected in env dict."""
    from cog_rag_cognee.cognee_setup import build_cognee_env
    from cog_rag_cognee.config import Settings

    settings = Settings(llm_model="mistral:7b", embedding_dimensions=1024)
    env = build_cognee_env(settings)

    assert env["LLM_MODEL"] == "mistral:7b"
    assert env["EMBEDDING_DIMENSIONS"] == "1024"
