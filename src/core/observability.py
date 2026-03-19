"""Structured logging and observability for multi-agent systems.

Provides:
- JSON-structured logging with correlation IDs
- Agent step timing and tracing
- Per-operation latency tracking
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Outputs logs as structured JSON for production observability."""

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "tenant_id"):
            log_data["tenant_id"] = record.tenant_id
        # Merge any extra fields passed via logger.info(..., extra={})
        for key in ("request_id", "method", "path", "status", "duration_ms",
                     "agent_name", "step", "model", "tokens", "cost"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class AgentTrace:
    """Tracks execution of a multi-agent workflow for observability."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    steps: list[dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)

    def add_step(
        self,
        agent_name: str,
        action: str,
        duration_ms: float,
        metadata: Optional[dict] = None,
    ):
        self.steps.append({
            "agent": agent_name,
            "action": action,
            "duration_ms": round(duration_ms, 2),
            "step_number": len(self.steps) + 1,
            **(metadata or {}),
        })

    @property
    def total_duration_ms(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": self.total_duration_ms,
            "total_steps": len(self.steps),
            "steps": self.steps,
        }


@contextmanager
def trace_step(trace: AgentTrace, agent_name: str, action: str):
    """Context manager to time an agent step and add it to the trace."""
    start = time.perf_counter()
    metadata = {}
    try:
        yield metadata
    finally:
        duration = (time.perf_counter() - start) * 1000
        trace.add_step(agent_name, action, duration, metadata)


@dataclass
class LatencyBudget:
    """Tracks latency against a budget for each component.

    Used to communicate SLAs between agentic AI and backend teams.
    """

    budgets: dict[str, float] = field(default_factory=lambda: {
        "retrieval": 200.0,     # ms
        "llm_call": 2000.0,     # ms
        "reranking": 100.0,     # ms
        "total": 3000.0,        # ms
    })
    actuals: dict[str, float] = field(default_factory=dict)

    def record(self, component: str, duration_ms: float):
        self.actuals[component] = duration_ms

    def check_budgets(self) -> list[str]:
        """Returns list of components that exceeded their budget."""
        violations = []
        for component, budget in self.budgets.items():
            actual = self.actuals.get(component, 0)
            if actual > budget:
                violations.append(
                    f"{component}: {actual:.0f}ms > {budget:.0f}ms budget"
                )
        return violations
