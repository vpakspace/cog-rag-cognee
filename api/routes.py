"""REST API endpoints."""
from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from api.deps import get_graph_client, get_service, verify_api_key
from cog_rag_cognee.config import get_settings
from cog_rag_cognee.graph_client import GraphClient
from cog_rag_cognee.health import check_ollama
from cog_rag_cognee.models import (
    GraphEntitiesResponse,
    GraphStats,
    HealthStatus,
    IngestResponse,
    QAResult,
    SearchResult,
)
from cog_rag_cognee.service import PipelineService

logger = logging.getLogger(__name__)

SearchMode = Literal["CHUNKS", "GRAPH_COMPLETION", "RAG_COMPLETION", "SUMMARIES"]

router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],
)

_DATASET_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class QueryRequest(BaseModel):
    """Request body for /query and /search endpoints."""

    text: str = Field(..., min_length=1, max_length=500_000)
    mode: SearchMode = "CHUNKS"
    limit: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    """Request body for /ingest endpoint."""

    text: str = Field(..., min_length=1, max_length=500_000)
    dataset_name: str = Field(default="main", min_length=1, max_length=64)

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        if not _DATASET_RE.match(v):
            raise ValueError("dataset_name must be alphanumeric, hyphens, or underscores")
        return v


@router.get("/health", response_model=HealthStatus)
async def health(gc: GraphClient = Depends(get_graph_client)):
    """Health check — verifies Neo4j and Ollama connectivity."""
    try:
        neo4j_ok = await gc.health_check()
    except Exception:
        neo4j_ok = False

    settings = get_settings()
    ollama_ok = await check_ollama(settings.llm_endpoint)

    status = "ok" if (neo4j_ok and ollama_ok) else "degraded"
    return HealthStatus(status=status, neo4j=neo4j_ok, ollama=ollama_ok)


@router.post("/query", response_model=QAResult)
async def query(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    """Full RAG query: search + answer generation."""
    return await svc.query(req.text, search_type=req.mode, limit=req.limit)


@router.post("/search", response_model=list[SearchResult])
async def search(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    """Search the knowledge graph without answer generation."""
    return await svc.search(req.text, search_type=req.mode, limit=req.limit)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, svc: PipelineService = Depends(get_service)):
    """Ingest text and run Cognee ECL pipeline."""
    result = await svc.add_text(req.text, dataset_name=req.dataset_name)
    cognify_result = await svc.cognify(dataset_name=req.dataset_name)
    return IngestResponse(ingest=result, cognify=str(cognify_result))


@router.post("/ingest-file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    dataset_name: str = Form("main"),
    svc: PipelineService = Depends(get_service),
):
    """Ingest an uploaded file (PDF, DOCX, TXT, etc.) and run Cognee pipeline."""
    # Validate dataset_name
    if not _DATASET_RE.match(dataset_name):
        raise HTTPException(status_code=422, detail="Invalid dataset_name")

    # Enforce file size limit
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large (max {mb} MB)")

    # Sanitize filename: strip path components, remove unsafe chars
    raw_name = (file.filename or "upload.txt").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    filename = re.sub(r"[^\w.\-]", "_", raw_name).lstrip(".")

    result = await svc.add_bytes(data, filename, dataset_name)
    cognify_result = await svc.cognify(dataset_name=dataset_name)
    return IngestResponse(ingest=result, cognify=str(cognify_result))


@router.get("/graph/stats", response_model=GraphStats)
async def graph_stats(gc: GraphClient = Depends(get_graph_client)):
    """Return knowledge graph statistics from Neo4j."""
    try:
        return await gc.get_stats()
    except Exception:
        logger.warning("Failed to fetch graph stats", exc_info=True)
        return GraphStats()


@router.get("/graph/entities", response_model=GraphEntitiesResponse)
async def graph_entities(
    limit: int = Query(default=200, ge=1, le=1000),
    entity_types: str | None = Query(default=None),
    gc: GraphClient = Depends(get_graph_client),
):
    """Return graph nodes and edges for visualization."""
    types_list = (
        [t.strip() for t in entity_types.split(",") if t.strip()]
        if entity_types
        else None
    )
    try:
        nodes = await gc.get_entities(limit=limit, entity_types=types_list)
        edges = await gc.get_relationships(limit=limit * 2, entity_types=types_list)
        return GraphEntitiesResponse(nodes=nodes, edges=edges)
    except Exception:
        logger.warning("Failed to fetch graph entities", exc_info=True)
        return GraphEntitiesResponse()


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""

    confirm: bool = False


@router.post("/reset")
async def reset(req: ResetRequest, svc: PipelineService = Depends(get_service)):
    """Reset all Cognee data. Requires confirm=true."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to reset all data")
    logger.warning("DATA RESET triggered via API")
    await svc.reset()
    return {"status": "ok"}
