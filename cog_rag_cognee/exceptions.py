"""Custom exception hierarchy for Cog-RAG Cognee."""


class CogRagError(Exception):
    """Base exception for all Cog-RAG Cognee errors."""


class ConfigError(CogRagError):
    """Configuration-related error."""


class IngestionError(CogRagError):
    """Document ingestion error."""


class SearchError(CogRagError):
    """Search/retrieval error."""


class GraphError(CogRagError):
    """Knowledge graph operation error."""


class OllamaError(CogRagError):
    """Ollama LLM/embedding service error."""
