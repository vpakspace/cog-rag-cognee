"""Shared test fixtures and environment defaults."""

import os
import sys
import types

# Inject a stub 'cognee' module so service.py can be imported without
# installing the real Cognee SDK. Tests mock it via unittest.mock.patch.
if "cognee" not in sys.modules:
    _cognee_stub = types.ModuleType("cognee")
    _cognee_stub.add = None
    _cognee_stub.cognify = None
    _cognee_stub.search = None
    _cognee_stub.prune = types.ModuleType("cognee.prune")
    _cognee_stub.prune.prune_data = None
    _cognee_stub.prune.prune_system = None
    # Stub deep submodule path for SearchType enum used by service.py
    import enum

    class _FakeSearchType(str, enum.Enum):
        SUMMARIES = "SUMMARIES"
        CHUNKS = "CHUNKS"
        INSIGHTS = "INSIGHTS"
        RAG_COMPLETION = "RAG_COMPLETION"
        GRAPH_COMPLETION = "GRAPH_COMPLETION"
        GRAPH_SUMMARY_COMPLETION = "GRAPH_SUMMARY_COMPLETION"
        TRIPLET_COMPLETION = "TRIPLET_COMPLETION"

    _search_types_mod = types.ModuleType("cognee.modules.search.types.SearchType")
    _search_types_mod.SearchType = _FakeSearchType  # type: ignore[attr-defined]

    _types_mod = types.ModuleType("cognee.modules.search.types")
    _types_mod.SearchType = _search_types_mod  # type: ignore[attr-defined]

    _search_mod = types.ModuleType("cognee.modules.search")
    _search_mod.types = _types_mod  # type: ignore[attr-defined]

    _modules_mod = types.ModuleType("cognee.modules")
    _modules_mod.search = _search_mod  # type: ignore[attr-defined]

    _cognee_stub.modules = _modules_mod  # type: ignore[attr-defined]

    sys.modules["cognee"] = _cognee_stub
    sys.modules["cognee.prune"] = _cognee_stub.prune
    sys.modules["cognee.modules"] = _modules_mod
    sys.modules["cognee.modules.search"] = _search_mod
    sys.modules["cognee.modules.search.types"] = _types_mod
    sys.modules["cognee.modules.search.types.SearchType"] = _search_types_mod

# Stub neo4j driver so graph_client.py can be imported without real neo4j package.
# Supports both sync GraphDatabase and async AsyncGraphDatabase.
if "neo4j" not in sys.modules:
    _neo4j_stub = types.ModuleType("neo4j")

    def _fake_driver(*_a, **_kw):
        return None

    _neo4j_stub.GraphDatabase = type(  # type: ignore[attr-defined]
        "GraphDatabase", (), {"driver": staticmethod(_fake_driver)}
    )
    _neo4j_stub.AsyncGraphDatabase = type(  # type: ignore[attr-defined]
        "AsyncGraphDatabase", (), {"driver": staticmethod(_fake_driver)}
    )
    # Provide neo4j.exceptions stub so tests can reference ServiceUnavailable etc.
    _neo4j_exc = types.ModuleType("neo4j.exceptions")

    class _ServiceUnavailable(Exception):
        pass

    _neo4j_exc.ServiceUnavailable = _ServiceUnavailable  # type: ignore[attr-defined]
    _neo4j_stub.exceptions = _neo4j_exc  # type: ignore[attr-defined]

    sys.modules["neo4j"] = _neo4j_stub
    sys.modules["neo4j.exceptions"] = _neo4j_exc

# Set default env vars before any Settings import
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "llama3.1:8b")
os.environ.setdefault("LLM_ENDPOINT", "http://localhost:11434/v1")
os.environ.setdefault("LLM_API_KEY", "ollama")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("EMBEDDING_ENDPOINT", "http://localhost:11434/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "neo4j")
os.environ.setdefault("GRAPH_DATABASE_URL", "neo4j://localhost:7687")
os.environ.setdefault("GRAPH_DATABASE_USERNAME", "neo4j")
os.environ.setdefault("GRAPH_DATABASE_PASSWORD", "password")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "gpt2")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("VECTOR_DB_PROVIDER", "lancedb")
os.environ.setdefault("STORAGE_ROOT_DIR", "./cognee_data")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("ALLOW_ANONYMOUS", "true")
os.environ.setdefault("API_HOST", "0.0.0.0")
os.environ.setdefault("API_PORT", "8508")
os.environ.setdefault("DOCLING_USE_GPU", "false")
os.environ.setdefault("UI_PORT", "8506")
