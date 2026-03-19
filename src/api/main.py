"""FastAPI application for Project Atlas.

Provides REST API for:
- Email filing (POST /api/v1/emails/file)
- Filing approval (POST /api/v1/emails/{id}/approve)
- Semantic search (POST /api/v1/search)
- User feedback (POST /api/v1/feedback)
- Health checks (GET /health)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import emails, feedback, health, search
from src.core.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from src.core.multi_tenant import TenantMiddleware
from src.core.observability import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("startup", extra={"message": "Project Atlas API starting"})
    # Startup: initialize services, warm up caches
    yield
    # Shutdown: cleanup resources
    logger.info("shutdown", extra={"message": "Project Atlas API shutting down"})


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Project Atlas",
        description="Smart Email Filing & Semantic Search for the AECO Industry",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware stack (order matters — outermost first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TenantMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router)
    app.include_router(emails.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(feedback.router, prefix="/api/v1")

    return app


app = create_app()
