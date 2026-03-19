"""Tool for matching email content to existing RFIs.

Checks for explicit RFI number references first (e.g. "RFI-247", "RFI #247").
Falls back to fuzzy subject matching if no number is found.
"""

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.core.observability import get_logger

logger = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_projects.json"

# Matches "RFI-247", "RFI #247", "RFI 247" (case-insensitive)
_RFI_NUMBER_PATTERN = re.compile(r"RFI[\s\-#]*(\d+)", re.IGNORECASE)


# ─── Input / Output ─────────────────────────────────────


class RFIMatcherInput(BaseModel):
    """Input for RFI matching tool."""

    email_subject: str = Field(..., description="Email subject line")
    email_body: str = Field(..., description="Email body text")
    project_id: Optional[str] = Field(default=None, description="Filter to a specific project")


class RFIMatcherOutput(BaseModel):
    """Top RFI matches with confidence scores."""

    matches: list[dict] = Field(
        default_factory=list,
        description="List of {rfi_id, number, subject, score} dicts sorted by score desc",
    )


# ─── Tool Schema ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "rfi_matcher",
    "description": (
        "Match email content to existing RFIs. Looks for explicit RFI number "
        "references first, then falls back to fuzzy subject matching. "
        "Returns up to 3 matches."
    ),
    "input_schema": RFIMatcherInput.model_json_schema(),
    "output_schema": RFIMatcherOutput.model_json_schema(),
}


# ─── Helpers ─────────────────────────────────────────────


def _load_rfis() -> list[dict]:
    """Load RFIs from the sample data file."""
    try:
        with open(DATA_PATH) as f:
            data = json.load(f)
        return data.get("rfis", [])
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load RFI data")
        return []


def _extract_rfi_numbers(text: str) -> list[str]:
    """Pull all RFI numbers from a string. Returns them as zero-padded strings."""
    return [match.group(0).upper().replace(" ", "-").replace("#", "-")
            for match in _RFI_NUMBER_PATTERN.finditer(text)]


def _normalize_rfi_number(raw: str) -> str:
    """Turn any RFI reference into the canonical 'RFI-NNN' format."""
    match = _RFI_NUMBER_PATTERN.search(raw)
    if match:
        return f"RFI-{match.group(1)}"
    return raw.upper()


def _tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def _fuzzy_subject_score(email_subject: str, rfi_subject: str) -> float:
    """Token overlap between email subject and RFI subject."""
    email_tokens = _tokenize(email_subject)
    rfi_tokens = _tokenize(rfi_subject)
    if not email_tokens or not rfi_tokens:
        return 0.0
    overlap = email_tokens & rfi_tokens
    # Use the smaller set as denominator so short subjects can still score high
    return len(overlap) / min(len(email_tokens), len(rfi_tokens))


# ─── Execute ─────────────────────────────────────────────


def execute(input_data: RFIMatcherInput) -> RFIMatcherOutput:
    """Match an email to existing RFIs by number or subject similarity."""
    rfis = _load_rfis()

    # Filter by project if specified
    if input_data.project_id:
        rfis = [r for r in rfis if r.get("project_id") == input_data.project_id]

    combined_text = f"{input_data.email_subject} {input_data.email_body}"
    referenced_numbers = [_normalize_rfi_number(n) for n in _extract_rfi_numbers(combined_text)]

    scored: list[dict] = []

    for rfi in rfis:
        rfi_number = rfi.get("number", "")

        # Exact number match gets a perfect score
        if rfi_number in referenced_numbers:
            scored.append({
                "rfi_id": rfi["id"],
                "number": rfi_number,
                "subject": rfi.get("subject", ""),
                "score": 1.0,
            })
            continue

        # Fall back to fuzzy subject matching
        subject_score = _fuzzy_subject_score(input_data.email_subject, rfi.get("subject", ""))
        if subject_score > 0.0:
            scored.append({
                "rfi_id": rfi["id"],
                "number": rfi_number,
                "subject": rfi.get("subject", ""),
                "score": round(subject_score, 3),
            })

    scored.sort(key=lambda m: m["score"], reverse=True)
    top_matches = scored[:3]

    logger.info(
        "RFI matching completed",
        extra={
            "exact_matches": len(referenced_numbers),
            "total_matches": len(top_matches),
            "project_filter": input_data.project_id,
        },
    )

    return RFIMatcherOutput(matches=top_matches)
