"""User feedback collection and analysis.

In-memory store for now. Collects ratings on search results so we can
identify bad queries for the eval suite and export feedback for reranker
weight tuning or fine-tuning.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackRecord:
    """A single piece of user feedback on a search result."""

    query: str
    result_id: str
    rating: int  # 1-5
    comment: str = ""
    tenant_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FeedbackStore:
    """In-memory feedback storage with thread-safe access.

    Designed so the interface stays the same when we swap in DynamoDB
    or Postgres later. The public methods don't expose storage details.
    """

    def __init__(self) -> None:
        self._records: list[FeedbackRecord] = []
        self._lock = threading.Lock()

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        """Store a feedback record."""
        # Clamp rating to valid range
        if feedback.rating < 1:
            feedback.rating = 1
        elif feedback.rating > 5:
            feedback.rating = 5

        with self._lock:
            self._records.append(feedback)

        logger.info(
            "Feedback recorded",
            extra={
                "result_id": feedback.result_id,
                "rating": feedback.rating,
                "tenant_id": feedback.tenant_id,
            },
        )

    def get_feedback_summary(
        self, tenant_id: Optional[str] = None
    ) -> dict:
        """Get aggregate feedback stats.

        Args:
            tenant_id: If provided, filter to this tenant only.

        Returns:
            {
                "total_count": int,
                "average_rating": float,
                "distribution": {1: count, 2: count, ...},
                "tenant_id": str or "all",
            }
        """
        with self._lock:
            records = list(self._records)

        if tenant_id:
            records = [r for r in records if r.tenant_id == tenant_id]

        if not records:
            return {
                "total_count": 0,
                "average_rating": 0.0,
                "distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "tenant_id": tenant_id or "all",
            }

        distribution = {i: 0 for i in range(1, 6)}
        for r in records:
            distribution[r.rating] = distribution.get(r.rating, 0) + 1

        avg = sum(r.rating for r in records) / len(records)

        return {
            "total_count": len(records),
            "average_rating": round(avg, 2),
            "distribution": distribution,
            "tenant_id": tenant_id or "all",
        }

    def get_low_rated_queries(
        self, threshold: float = 2.0
    ) -> list[FeedbackRecord]:
        """Return queries rated below the threshold.

        These are candidates for the eval suite — they represent cases
        where the system performed poorly and we want to track improvement.
        """
        with self._lock:
            records = list(self._records)

        low_rated = [r for r in records if r.rating <= threshold]
        low_rated.sort(key=lambda r: r.rating)

        logger.info(
            "Low-rated queries retrieved",
            extra={
                "threshold": threshold,
                "count": len(low_rated),
            },
        )

        return low_rated

    def export_for_training(self) -> list[dict]:
        """Export feedback in a format suitable for fine-tuning or reranker tuning.

        Returns a list of dicts with query, result_id, rating, and a
        normalized score (0.0-1.0) derived from the 1-5 rating. This
        format works for both reranker weight adjustment and LLM
        fine-tuning preference pairs.
        """
        with self._lock:
            records = list(self._records)

        exported = []
        for r in records:
            # Normalize 1-5 rating to 0.0-1.0 score
            normalized_score = (r.rating - 1) / 4.0

            exported.append({
                "query": r.query,
                "result_id": r.result_id,
                "rating": r.rating,
                "normalized_score": round(normalized_score, 3),
                "comment": r.comment,
                "tenant_id": r.tenant_id,
                "timestamp": r.timestamp.isoformat(),
            })

        logger.info(
            "Feedback exported for training",
            extra={"record_count": len(exported)},
        )

        return exported
