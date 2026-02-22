"""Shared test fixtures and environment defaults."""

import os

# Set default env vars before any Settings import
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.1:8b")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "neo4j")
os.environ.setdefault("GRAPH_DATABASE_URL", "neo4j://localhost:7687")
os.environ.setdefault("GRAPH_DATABASE_USERNAME", "neo4j")
os.environ.setdefault("GRAPH_DATABASE_PASSWORD", "password")
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
os.environ.setdefault("STORAGE_ROOT_DIR", "./cognee_data")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("API_HOST", "0.0.0.0")
os.environ.setdefault("API_PORT", "8508")
os.environ.setdefault("UI_PORT", "8506")
