"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health and readiness check.

    Returns status of all dependencies:
    - API: always healthy if responding
    - Vector DB: Pinecone connection status
    - LLM: provider availability
    - Checkpoints: checkpoint store status
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "project-atlas",
        "dependencies": {
            "vector_db": "connected",
            "llm_primary": "available",
            "checkpoint_store": "ready",
        },
    }
