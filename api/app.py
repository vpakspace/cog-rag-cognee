"""FastAPI application factory."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from cog_rag_cognee.cognee_setup import apply_cognee_env
from cog_rag_cognee.config import Settings, get_settings
from cog_rag_cognee.exceptions import CogRagError, IngestionError, SearchError
from cog_rag_cognee.health import check_ollama

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    apply_cognee_env(settings)
    if not settings.api_key:
        if not settings.debug and not settings.allow_anonymous:
            raise RuntimeError(
                "API_KEY is required in production. "
                "Set API_KEY, enable DEBUG, or set ALLOW_ANONYMOUS=true."
            )
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
    if await check_ollama(settings.llm_endpoint):
        logger.info("Ollama: connected (%s)", settings.llm_endpoint)
    else:
        logger.warning("Ollama: unreachable at %s", settings.llm_endpoint)


async def cograg_error_handler(request: Request, exc: CogRagError) -> JSONResponse:
    """Map custom exceptions to appropriate HTTP status codes."""
    status_code = 502 if isinstance(exc, (IngestionError, SearchError)) else 500
    logger.warning("CogRagError: %s", exc, exc_info=True)

    settings = get_settings()
    if settings.debug:
        detail = str(exc)
    else:
        detail = "An internal error occurred. Check server logs for details."

    return JSONResponse(
        status_code=status_code,
        content={"code": exc.code, "error": type(exc).__name__, "detail": detail},
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize Pydantic validation errors to consistent {code, error, detail} shape."""
    return JSONResponse(
        status_code=422,
        content={
            "code": "ERR_VALIDATION",
            "error": "ValidationError",
            "detail": jsonable_encoder(exc.errors()),
        },
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
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Custom exception handlers
    app.add_exception_handler(CogRagError, cograg_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    if not settings.debug and "*" in origins:
        logger.warning("CORS wildcard '*' is not allowed in production — removing it")
        origins = [o for o in origins if o != "*"]
        if not origins:
            origins = ["http://localhost:8506"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    from api.routes import router

    app.include_router(router)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add X-Request-ID, security headers, and log request metrics."""
        from cog_rag_cognee.request_context import request_id_var

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        start = time.monotonic()
        try:
            try:
                response = await call_next(request)
            except RequestValidationError as exc:
                # BaseHTTPMiddleware may propagate validation errors before
                # ExceptionMiddleware catches them — handle here as fallback.
                response = await validation_error_handler(request, exc)
            duration_ms = round((time.monotonic() - start) * 1000, 1)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Cache-Control"] = "no-store"
            logger.info(
                "%s %s %s %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        finally:
            request_id_var.reset(token)

    return app
