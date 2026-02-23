"""Application configuration via Pydantic Settings."""

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_endpoint: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"

    # Embeddings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_endpoint: str = "http://localhost:11434/api/embed"
    embedding_dimensions: int = 768

    # Graph DB
    graph_database_provider: str = "neo4j"
    graph_database_url: str = "neo4j://localhost:7687"
    graph_database_username: str = "neo4j"
    graph_database_password: str = "password"

    # Cognee SDK
    huggingface_tokenizer: str = "gpt2"
    enable_backend_access_control: bool = False

    # Vector DB
    vector_db_provider: str = "lancedb"

    # Storage
    storage_root_dir: str = "./cognee_data"

    # Cognee SDK
    cognee_timeout: int = 300  # seconds for cognee operations
    neo4j_timeout: int = 30  # seconds for individual Neo4j operations

    # Docling
    docling_use_gpu: bool = False

    # API
    api_key: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8508
    cors_origins: str = "http://localhost:8506"
    debug: bool = False
    allow_anonymous: bool = False
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB

    # UI
    ui_port: int = 8506

    @field_validator("cognee_timeout")
    @classmethod
    def validate_cognee_timeout(cls, v: int) -> int:
        if not (10 <= v <= 3600):
            raise ValueError("cognee_timeout must be between 10 and 3600 seconds")
        return v

    @field_validator("neo4j_timeout")
    @classmethod
    def validate_neo4j_timeout(cls, v: int) -> int:
        if not (1 <= v <= 300):
            raise ValueError("neo4j_timeout must be between 1 and 300 seconds")
        return v

    @field_validator("max_upload_bytes")
    @classmethod
    def validate_max_upload_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_bytes must be positive")
        return v

    @model_validator(mode="after")
    def validate_ports(self) -> "Settings":
        if self.api_port == self.ui_port:
            raise ValueError(f"api_port and ui_port must differ (both are {self.api_port})")
        return self

    @model_validator(mode="after")
    def validate_timeouts(self) -> "Settings":
        if self.cognee_timeout < self.neo4j_timeout:
            raise ValueError(
                f"cognee_timeout ({self.cognee_timeout}) must be "
                f">= neo4j_timeout ({self.neo4j_timeout})"
            )
        return self

    @property
    def ollama_base_url(self) -> str:
        """Return the base URL for Ollama API."""
        return self.llm_endpoint

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
