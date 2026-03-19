"""Checkpoint persistence and human-in-the-loop (HITL) support.

Provides:
- DynamoDB-backed checkpointing for agent state persistence
- HITL pause/resume flow for low-confidence filing decisions
- State serialization/deserialization

When the filing agent's confidence falls between 0.5 and 0.85,
the graph pauses at the human_review node. The state is checkpointed
to DynamoDB, and the system waits for a human to approve or correct
the filing decision via the API.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointRecord:
    """A persisted checkpoint of agent state."""

    thread_id: str
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    state: dict = field(default_factory=dict)
    status: str = "paused"  # paused / resumed / completed
    created_at: float = field(default_factory=time.time)
    resumed_at: Optional[float] = None
    human_decision: Optional[str] = None


class CheckpointStore:
    """In-memory checkpoint store.

    In production, this would be backed by DynamoDB:
    - PK: TENANT#{tenant_id}
    - SK: CHECKPOINT#{thread_id}
    - TTL: 7 days (auto-cleanup of old checkpoints)

    The DynamoDB table structure:
    {
        "pk": "TENANT#tenant_a",
        "sk": "CHECKPOINT#email_001",
        "state": <serialized agent state>,
        "status": "paused",
        "created_at": 1711036800,
        "ttl": 1711641600
    }
    """

    def __init__(self):
        self._store: dict[str, CheckpointRecord] = {}

    def save(self, thread_id: str, state: dict, tenant_id: str = "demo") -> CheckpointRecord:
        """Save agent state to checkpoint store.

        Called when the graph pauses for human review.
        """
        record = CheckpointRecord(
            thread_id=thread_id,
            state=_serialize_state(state),
            status="paused",
        )
        key = f"{tenant_id}:{thread_id}"
        self._store[key] = record

        logger.info(
            "checkpoint_saved",
            extra={
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "checkpoint_id": record.checkpoint_id,
                "status": "paused",
            },
        )
        return record

    def load(self, thread_id: str, tenant_id: str = "demo") -> Optional[CheckpointRecord]:
        """Load a checkpointed state.

        Called when resuming after human review.
        """
        key = f"{tenant_id}:{thread_id}"
        record = self._store.get(key)
        if record:
            logger.info(
                "checkpoint_loaded",
                extra={
                    "thread_id": thread_id,
                    "tenant_id": tenant_id,
                    "status": record.status,
                },
            )
        return record

    def resume(
        self,
        thread_id: str,
        human_decision: str,
        corrected_project_id: Optional[str] = None,
        tenant_id: str = "demo",
    ) -> Optional[dict]:
        """Resume a paused agent with human feedback.

        The human's decision is written back to state, and the
        graph can resume from the interrupt point.

        Args:
            thread_id: The filing thread to resume
            human_decision: "approve" or "reject" or "correct"
            corrected_project_id: If human corrects the project assignment

        Returns:
            Updated state dict ready for graph resumption
        """
        record = self.load(thread_id, tenant_id)
        if not record:
            logger.warning("checkpoint_not_found", extra={"thread_id": thread_id})
            return None

        if record.status != "paused":
            logger.warning(
                "checkpoint_not_paused",
                extra={"thread_id": thread_id, "status": record.status},
            )
            return None

        # Update the state with human decision
        state = record.state
        state["human_feedback"] = human_decision
        state["needs_human_review"] = False

        if human_decision == "approve":
            state["filing_action"] = "auto_file"
            state["filing_confidence"] = 1.0  # Human confirmed
        elif human_decision == "correct" and corrected_project_id:
            state["filing_action"] = "auto_file"
            state["filing_project_id"] = corrected_project_id
            state["filing_confidence"] = 1.0
        elif human_decision == "reject":
            state["filing_action"] = "flagged"
            state["filing_confidence"] = 0.0

        # Update record
        record.status = "resumed"
        record.resumed_at = time.time()
        record.human_decision = human_decision
        record.state = state

        key = f"{tenant_id}:{thread_id}"
        self._store[key] = record

        logger.info(
            "checkpoint_resumed",
            extra={
                "thread_id": thread_id,
                "human_decision": human_decision,
                "corrected_project_id": corrected_project_id,
            },
        )

        return state

    def list_pending(self, tenant_id: str = None) -> list[CheckpointRecord]:
        """List all checkpoints waiting for human review."""
        results = []
        for key, record in self._store.items():
            if record.status == "paused":
                if tenant_id is None or key.startswith(f"{tenant_id}:"):
                    results.append(record)
        return sorted(results, key=lambda r: r.created_at)


def should_request_human_review(state: dict) -> bool:
    """Check if the current state warrants human review.

    Called by the graph's conditional edge to decide routing.
    """
    action = state.get("filing_action")
    confidence = state.get("filing_confidence", 0.0)

    if action == "needs_review":
        return True
    if 0.5 <= confidence < 0.85:
        return True
    return False


def _serialize_state(state: dict) -> dict:
    """Serialize agent state for storage.

    Handles non-JSON-serializable types.
    """
    serialized = {}
    for key, value in state.items():
        if key == "messages":
            # Skip message objects — they're LangGraph internal
            serialized[key] = []
        else:
            try:
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                serialized[key] = str(value)
    return serialized


# Global checkpoint store instance
_checkpoint_store = CheckpointStore()


def get_checkpoint_store() -> CheckpointStore:
    return _checkpoint_store
