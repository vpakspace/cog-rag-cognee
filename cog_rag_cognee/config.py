"""Application configuration via Pydantic Settings."""

from functools import lru_cache

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

    # Docling
    docling_use_gpu: bool = False

    # API
    api_key: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8508
    cors_origins: str = "http://localhost:8506"
    debug: bool = False
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB

    # UI
    ui_port: int = 8506

    @property
    def ollama_base_url(self) -> str:
        """Return the base URL for Ollama API."""
        return self.llm_endpoint

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
