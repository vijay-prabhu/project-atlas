"""Tool that files an email to a project folder.

In this portfolio version the action is simulated — it logs the filing
and returns a success record. A production implementation would write to
a document management system or cloud storage.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.core.observability import get_logger

logger = get_logger(__name__)


# ─── Input / Output ─────────────────────────────────────


class FilingActionInput(BaseModel):
    """Input for the filing action tool."""

    email_id: str = Field(..., description="ID of the email being filed")
    project_id: str = Field(..., description="Target project ID")
    folder_path: str = Field(
        ...,
        description="Folder path within the project (e.g. 'RFIs/RFI-247')",
    )
    category: str = Field(
        ...,
        description="Email category (rfi, submittal, change_order, etc.)",
    )


class FilingActionOutput(BaseModel):
    """Result of a filing action."""

    success: bool = Field(..., description="Whether the filing succeeded")
    filed_at: datetime = Field(..., description="Timestamp of the filing")
    filing_record_id: str = Field(..., description="Unique ID for this filing record")


# ─── Tool Schema ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "filing_action",
    "description": (
        "File an email to a specific project and folder. Returns a filing "
        "record with a unique ID for audit trail purposes."
    ),
    "input_schema": FilingActionInput.model_json_schema(),
    "output_schema": FilingActionOutput.model_json_schema(),
}


# ─── Execute ─────────────────────────────────────────────


def execute(input_data: FilingActionInput) -> FilingActionOutput:
    """Simulate filing an email and return a success record."""
    filing_record_id = str(uuid.uuid4())
    filed_at = datetime.now(timezone.utc)

    logger.info(
        "Email filed",
        extra={
            "email_id": input_data.email_id,
            "project_id": input_data.project_id,
            "folder_path": input_data.folder_path,
            "category": input_data.category,
            "filing_record_id": filing_record_id,
        },
    )

    return FilingActionOutput(
        success=True,
        filed_at=filed_at,
        filing_record_id=filing_record_id,
    )
