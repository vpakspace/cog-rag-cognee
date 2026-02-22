"""REST API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from api.deps import get_graph_client, get_service
from cog_rag_cognee.graph_client import GraphClient
from cog_rag_cognee.service import PipelineService

router = APIRouter(prefix="/api/v1")


class QueryRequest(BaseModel):
    """Request body for /query and /search endpoints."""

    text: str = Field(..., min_length=1)
    mode: str = "GRAPH_COMPLETION"
    limit: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    """Request body for /ingest endpoint."""

    text: str = Field(..., min_length=1)
    dataset_name: str = "main"


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/query")
async def query(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    """Full RAG query: search + answer generation."""
    result = await svc.query(req.text, search_type=req.mode, limit=req.limit)
    return result.model_dump()


@router.post("/search")
async def search(req: QueryRequest, svc: PipelineService = Depends(get_service)):
    """Search the knowledge graph without answer generation."""
    results = await svc.search(req.text, search_type=req.mode, limit=req.limit)
    return [r.model_dump() for r in results]


@router.post("/ingest")
async def ingest(req: IngestRequest, svc: PipelineService = Depends(get_service)):
    """Ingest text and run Cognee ECL pipeline."""
    result = await svc.add_text(req.text, dataset_name=req.dataset_name)
    cognify_result = await svc.cognify(dataset_name=req.dataset_name)
    return {"ingest": result, "cognify": str(cognify_result)}


@router.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    dataset_name: str = Form("main"),
    svc: PipelineService = Depends(get_service),
):
    """Ingest an uploaded file (PDF, DOCX, TXT, etc.) and run Cognee pipeline."""
    data = await file.read()
    result = await svc.add_bytes(data, file.filename or "upload.txt", dataset_name)
    cognify_result = await svc.cognify(dataset_name=dataset_name)
    return {"ingest": result, "cognify": str(cognify_result)}


@router.get("/graph/stats")
async def graph_stats(gc: GraphClient = Depends(get_graph_client)):
    """Return knowledge graph statistics from Neo4j."""
    try:
        return gc.get_stats()
    except Exception:
        return {"nodes": 0, "edges": 0, "entity_types": {}}


@router.get("/graph/entities")
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
        nodes = gc.get_entities(limit=limit, entity_types=types_list)
        edges = gc.get_relationships(limit=limit * 2, entity_types=types_list)
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {"nodes": [], "edges": []}
