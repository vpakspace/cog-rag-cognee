"""Domain models for the Cog-RAG Cognee pipeline."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result from retrieval."""

    content: str
    score: float = Field(ge=0.0, le=1.0)
    source: str = ""
    entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class QAResult(BaseModel):
    """Question-answering result with sources and confidence."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SearchResult] = Field(default_factory=list)
    mode: str = "default"


class IngestResult(BaseModel):
    """Result of ingesting a single document."""

    filename: str
    chunks_added: int = 0
    entities_extracted: int = 0
    status: str = "success"
    error: str | None = None


class GraphStats(BaseModel):
    """Knowledge graph statistics."""

    nodes: int = 0
    edges: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Service health status."""

    status: str = "ok"
    ollama: bool = False
    neo4j: bool = False
    lancedb: bool = False
