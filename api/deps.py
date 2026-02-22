"""Dependency injection for FastAPI."""
from __future__ import annotations

from cog_rag_cognee.config import get_settings
from cog_rag_cognee.graph_client import GraphClient
from cog_rag_cognee.service import PipelineService

_service: PipelineService | None = None
_graph_client: GraphClient | None = None


def get_service() -> PipelineService:
    """Return the global PipelineService instance."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service


def set_service(service: PipelineService | None) -> None:
    """Set the global PipelineService instance (for testing)."""
    global _service
    _service = service


def get_graph_client() -> GraphClient:
    """Return the global GraphClient instance (lazy init)."""
    global _graph_client
    if _graph_client is None:
        settings = get_settings()
        _graph_client = GraphClient(
            uri=settings.graph_database_url,
            username=settings.graph_database_username,
            password=settings.graph_database_password,
        )
    return _graph_client


def set_graph_client(client: GraphClient | None) -> None:
    """Set the global GraphClient instance (for testing)."""
    global _graph_client
    _graph_client = client
