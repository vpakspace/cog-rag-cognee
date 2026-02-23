# Improvement Plan — cog-rag-cognee

**Created**: 2026-02-23
**Source**: Deep analysis by 3 agents (architecture, tests, security)

## Priority 1 — Security (CRITICAL/HIGH)

### 1.1 Auth middleware + CORS
- [ ] API key auth via FastAPI `Depends()` — check `config.api_key`
- [ ] CORS middleware with configurable origins
- [ ] slowapi rate limiting (already in requirements.txt)

### 1.2 File upload hardening
- [ ] Max file size 50MB (`routes.py`)
- [ ] Content-type validation (magic bytes, not just extension)
- [ ] `max_length` on `IngestRequest.text` (e.g. 500KB)

### 1.3 Input validation
- [ ] `mode` field → `Literal["CHUNKS", "GRAPH_COMPLETION", "RAG_COMPLETION", "SUMMARIES"]`
- [ ] `dataset_name` regex validation (alphanumeric + underscore)
- [ ] Filename sanitization in `ingest-file`

### 1.4 Secrets & config
- [ ] `reload=True` → gated by `DEBUG` env var
- [ ] Neo4j password: use env var in docker-compose healthcheck
- [ ] `api_host` default → `127.0.0.1` (not `0.0.0.0`)

## Priority 2 — Error Handling

### 2.1 Use custom exceptions
- [ ] Wire `exceptions.py` into service layer (wrap Cognee errors)
- [ ] FastAPI exception handlers for custom exceptions → proper HTTP codes
- [ ] Replace silent `except Exception` in routes with logging + error responses

### 2.2 UI error handling
- [ ] Add try/except to text ingestion path in Streamlit
- [ ] Sanitize exception messages shown to users

## Priority 3 — Code Quality

### 3.1 Dead code cleanup
- [ ] Remove unused models: `HealthStatus`, `IngestResult`, `GraphStats` (or wire them)
- [ ] Wire response_model to endpoints (use Pydantic models)

### 3.2 Graph client fixes
- [ ] `id(n)` → `elementId(n)` (Neo4j 5 deprecation)
- [ ] `close()` called in API lifespan shutdown
- [ ] Combine `get_stats()` into single Cypher query

### 3.3 Service layer
- [ ] DRY: deduplicate `_extract_result` branches
- [ ] Rename misleading `query()` or add actual LLM generation

## Priority 4 — Performance

### 4.1 Temp file cleanup in graph_viz.py
### 4.2 httpx Client reuse in Streamlit (connection pooling)
### 4.3 DoclingLoader singleton (cache in service)
### 4.4 `cognify()` as background task (future)

## Priority 5 — Tests

### 5.1 Cover error paths in service.py (reset_system, no-results, fallback)
### 5.2 Graph entities exception fallback test
### 5.3 Input validation tests (422 responses)

## Priority 6 — Architecture (future)

### 6.1 Unify UI data path (all via API or all via svc)
### 6.2 Async Neo4j driver
### 6.3 Response envelope pattern
