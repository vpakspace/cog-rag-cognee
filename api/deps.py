"""Dependency injection for FastAPI."""
from __future__ import annotations

from cog_rag_cognee.service import PipelineService

_service: PipelineService | None = None


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
