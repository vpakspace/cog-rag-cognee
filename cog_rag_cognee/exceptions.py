"""Custom exception hierarchy for Cog-RAG Cognee."""


class CogRagError(Exception):
    """Base exception for all Cog-RAG Cognee errors."""

    code: str = "ERR_INTERNAL"


class ConfigError(CogRagError):
    """Configuration-related error."""

    code: str = "ERR_CONFIG"


class IngestionError(CogRagError):
    """Document ingestion error."""

    code: str = "ERR_INGESTION"


class SearchError(CogRagError):
    """Search/retrieval error."""

    code: str = "ERR_SEARCH"


class GraphError(CogRagError):
    """Knowledge graph operation error."""

    code: str = "ERR_GRAPH"


class OllamaError(CogRagError):
    """Ollama LLM/embedding service error."""

    code: str = "ERR_OLLAMA"
