"""Tool for looking up a sender's past filing patterns.

Maps a sender email to their known projects so the filing agent can use
prior context when deciding where to file a new email.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.observability import get_logger

logger = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_projects.json"


# ─── Input / Output ─────────────────────────────────────


class SenderHistoryInput(BaseModel):
    """Input for sender history lookup."""

    sender_email: str = Field(..., description="Email address of the sender")


class SenderHistoryOutput(BaseModel):
    """Filing patterns for a known sender."""

    filing_patterns: list[dict] = Field(
        default_factory=list,
        description=(
            "List of {project_id, project_name, count, last_filed} dicts "
            "sorted by count desc"
        ),
    )


# ─── Tool Schema ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "sender_history",
    "description": (
        "Look up a sender's past filing patterns. Returns projects the sender "
        "is associated with, sorted by frequency. Helps the filing agent use "
        "prior context when deciding where to file."
    ),
    "input_schema": SenderHistoryInput.model_json_schema(),
    "output_schema": SenderHistoryOutput.model_json_schema(),
}


# ─── Helpers ─────────────────────────────────────────────


def _load_data() -> tuple[list[dict], list[dict]]:
    """Load contacts and projects from the sample data file."""
    try:
        with open(DATA_PATH) as f:
            data = json.load(f)
        return data.get("contacts", []), data.get("projects", [])
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load sample data for sender history")
        return [], []


def _build_project_index(projects: list[dict]) -> dict[str, dict]:
    """Map project ID -> project dict for fast lookups."""
    return {p["id"]: p for p in projects}


# ─── Execute ─────────────────────────────────────────────


def execute(input_data: SenderHistoryInput) -> SenderHistoryOutput:
    """Look up projects associated with a sender email."""
    contacts, projects = _load_data()
    project_index = _build_project_index(projects)

    sender = input_data.sender_email.lower().strip()

    # Find all contacts matching this email
    matching_contacts = [c for c in contacts if c.get("email", "").lower() == sender]

    if not matching_contacts:
        logger.info("No sender history found", extra={"sender": sender})
        return SenderHistoryOutput(filing_patterns=[])

    # Collect project associations across all matching contacts.
    # In a real system, count and last_filed would come from a filing log.
    # Here we derive a synthetic count from the number of contact entries
    # that reference each project.
    project_counts: dict[str, int] = {}
    for contact in matching_contacts:
        for pid in contact.get("project_ids", []):
            project_counts[pid] = project_counts.get(pid, 0) + 1

    patterns = []
    for pid, count in project_counts.items():
        project = project_index.get(pid)
        if not project:
            continue
        patterns.append({
            "project_id": pid,
            "project_name": project.get("name", ""),
            "count": count,
            # In production this comes from the filing log. Using a
            # placeholder here since we don't have real filing history.
            "last_filed": None,
        })

    patterns.sort(key=lambda p: p["count"], reverse=True)

    logger.info(
        "Sender history lookup completed",
        extra={"sender": sender, "projects_found": len(patterns)},
    )

    return SenderHistoryOutput(filing_patterns=patterns)
