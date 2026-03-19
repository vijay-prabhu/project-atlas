"""Feedback API routes for RAG quality improvement."""

import time

from fastapi import APIRouter, Depends

from src.api.deps import get_tenant_id
from src.api.schemas import FeedbackRequest

router = APIRouter(prefix="/feedback", tags=["feedback"])

# In-memory feedback store (production: DynamoDB)
_feedback_store: list[dict] = []


@router.post("")
async def submit_feedback(
    request: FeedbackRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """Submit feedback on a search result.

    User feedback feeds into the evaluation suite:
    - Low-rated queries become test cases
    - Patterns in feedback guide reranker weight tuning
    - Aggregate metrics track RAG quality over time
    """
    record = {
        "tenant_id": tenant_id,
        "query": request.query,
        "result_id": request.result_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": time.time(),
    }
    _feedback_store.append(record)

    return {"status": "recorded", "feedback_count": len(_feedback_store)}


@router.get("/summary")
async def feedback_summary(
    tenant_id: str = Depends(get_tenant_id),
):
    """Get aggregate feedback metrics."""
    tenant_feedback = [f for f in _feedback_store if f["tenant_id"] == tenant_id]

    if not tenant_feedback:
        return {"total": 0, "average_rating": 0, "distribution": {}}

    ratings = [f["rating"] for f in tenant_feedback]
    distribution = {i: ratings.count(i) for i in range(1, 6)}

    return {
        "total": len(tenant_feedback),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "distribution": distribution,
        "low_rated_count": sum(1 for r in ratings if r <= 2),
    }
