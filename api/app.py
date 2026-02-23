"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import Settings, get_settings
from cog_rag_cognee.exceptions import CogRagError, IngestionError, SearchError

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    apply_cognee_env(settings)
    if not settings.api_key:
        logger.warning("API_KEY is not set — all endpoints are unauthenticated!")

    # Check dependencies (non-blocking — log warnings only)
    await _check_startup_deps(settings)

    logger.info("cog-rag-cognee API started on port %d", settings.api_port)
    yield
    # Shutdown: close GraphClient connection pool
    from api.deps import get_graph_client, set_graph_client

    try:
        gc = get_graph_client()
        await gc.close()
        set_graph_client(None)
        logger.info("GraphClient closed")
    except Exception:
        pass
    logger.info("cog-rag-cognee API shutting down")


async def _check_startup_deps(settings: Settings) -> None:
    """Check Neo4j and Ollama at startup. Log warnings if unreachable."""
    import httpx

    # Neo4j
    try:
        from api.deps import get_graph_client

        gc = get_graph_client()
        if await gc.health_check():
            logger.info("Neo4j: connected (%s)", settings.graph_database_url)
        else:
            logger.warning("Neo4j: unreachable at %s", settings.graph_database_url)
    except Exception:
        logger.warning("Neo4j: unreachable at %s", settings.graph_database_url)

    # Ollama
    base = settings.llm_endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{base}/api/tags")
            if resp.status_code == 200:
                logger.info("Ollama: connected (%s)", base)
            else:
                logger.warning("Ollama: unreachable at %s", base)
    except Exception:
        logger.warning("Ollama: unreachable at %s", base)


async def cograg_error_handler(request: Request, exc: CogRagError) -> JSONResponse:
    """Map custom exceptions to appropriate HTTP status codes."""
    if isinstance(exc, (IngestionError, SearchError)):
        status_code = 502
    else:
        status_code = 500
    logger.warning("CogRagError: %s", exc, exc_info=True)

    settings = get_settings()
    if settings.debug:
        detail = str(exc)
    else:
        detail = "An internal error occurred. Check server logs for details."

    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": detail},
    )


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="cog-rag-cognee",
        description="Semantic memory layer with Cognee SDK — 100% local stack",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Custom exception handler
    app.add_exception_handler(CogRagError, cograg_error_handler)  # type: ignore[arg-type]

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    from api.routes import router

    app.include_router(router)

    return app
