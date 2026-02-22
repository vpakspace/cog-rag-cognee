"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    apply_cognee_env(settings)
    logger.info("cog-rag-cognee API started on port %d", settings.api_port)
    yield
    logger.info("cog-rag-cognee API shutting down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="cog-rag-cognee",
        description="Semantic memory layer with Cognee SDK — 100% local stack",
        version="0.1.0",
        lifespan=lifespan,
    )

    from api.routes import router

    app.include_router(router)

    return app
