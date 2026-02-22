"""REST API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_service
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


@router.get("/graph/stats")
async def graph_stats():
    """Return knowledge graph statistics."""
    return {"nodes": 0, "edges": 0, "entity_types": {}}
