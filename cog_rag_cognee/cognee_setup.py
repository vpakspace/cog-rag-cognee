"""Cognee SDK initialization and configuration."""
from __future__ import annotations

import logging
import os

from cog_rag_cognee.config import Settings

logger = logging.getLogger(__name__)


def build_cognee_env(settings: Settings) -> dict[str, str]:
    """Build environment variables dict for Cognee SDK."""
    return {
        "LLM_PROVIDER": settings.llm_provider,
        "LLM_MODEL": settings.llm_model,
        "LLM_ENDPOINT": settings.llm_endpoint,
        "EMBEDDING_PROVIDER": settings.embedding_provider,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_ENDPOINT": settings.embedding_endpoint,
        "EMBEDDING_DIMENSIONS": str(settings.embedding_dimensions),
        "GRAPH_DATABASE_PROVIDER": settings.graph_database_provider,
        "GRAPH_DATABASE_URL": settings.graph_database_url,
        "GRAPH_DATABASE_USERNAME": settings.graph_database_username,
        "GRAPH_DATABASE_PASSWORD": settings.graph_database_password,
        "VECTOR_DB_PROVIDER": settings.vector_db_provider,
        "STORAGE_ROOT_DIR": settings.storage_root_dir,
    }


def apply_cognee_env(settings: Settings) -> None:
    """Set environment variables for Cognee SDK before import."""
    env = build_cognee_env(settings)
    for key, value in env.items():
        os.environ[key] = value
    logger.info(
        "Cognee configured: LLM=%s/%s, Graph=%s, Vector=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.graph_database_provider,
        settings.vector_db_provider,
    )
