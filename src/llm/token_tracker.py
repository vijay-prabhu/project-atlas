"""Per-tenant, per-model, per-operation cost tracking.

In-memory for now. Designed to swap in DynamoDB later without changing the
public interface.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)

# Pricing per 1M tokens: (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost for a single LLM call."""
    input_rate, output_rate = _MODEL_PRICING.get(model, (0.0, 0.0))
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return round(cost, 8)


@dataclass
class UsageRecord:
    """A single tracked LLM call."""

    tenant_id: str
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float


class TokenTracker:
    """Singleton that accumulates token usage across the process.

    Thread-safe via a lock around the shared records list. Call
    ``TokenTracker()`` from anywhere and you'll get the same instance.
    """

    _instance: Optional["TokenTracker"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "TokenTracker":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._records: list[UsageRecord] = []
                    instance._lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        tenant_id: str,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record a single LLM call's token usage and cost."""
        cost = _calculate_cost(model, input_tokens, output_tokens)
        record = UsageRecord(
            tenant_id=tenant_id,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=cost,
        )
        with self._lock:
            self._records.append(record)

        logger.info(
            "Token usage tracked",
            extra={
                "tenant_id": tenant_id,
                "model": model,
                "operation": operation,
                "tokens": input_tokens + output_tokens,
                "cost": cost,
            },
        )

    def get_tenant_usage(self, tenant_id: str) -> dict:
        """Aggregate usage for a specific tenant.

        Returns:
            {
                "tenant_id": str,
                "total_input_tokens": int,
                "total_output_tokens": int,
                "total_tokens": int,
                "total_cost": float,
                "by_model": {
                    "gpt-4o-mini": {"input_tokens": ..., "output_tokens": ..., "cost": ...},
                    ...
                },
            }
        """
        with self._lock:
            tenant_records = [r for r in self._records if r.tenant_id == tenant_id]

        by_model: dict[str, dict] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
        )
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for r in tenant_records:
            total_input += r.input_tokens
            total_output += r.output_tokens
            total_cost += r.cost
            entry = by_model[r.model]
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["cost"] += r.cost

        return {
            "tenant_id": tenant_id,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(total_cost, 8),
            "by_model": dict(by_model),
        }

    def get_operation_costs(self) -> dict:
        """Average cost per operation type across all tenants.

        Returns:
            {
                "classification": {"avg_cost": 0.0001, "call_count": 42},
                ...
            }
        """
        with self._lock:
            records = list(self._records)

        by_op: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_op[r.operation].append(r.cost)

        return {
            op: {
                "avg_cost": round(sum(costs) / len(costs), 8),
                "call_count": len(costs),
            }
            for op, costs in by_op.items()
        }

    def reset(self) -> None:
        """Clear all records. Useful for testing."""
        with self._lock:
            self._records.clear()
