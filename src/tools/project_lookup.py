"""Tool for fuzzy-matching a query to a project in the tenant's project list."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.observability import get_logger

logger = get_logger(__name__)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_projects.json"


# ─── Input / Output ─────────────────────────────────────


class ProjectLookupInput(BaseModel):
    """Input for project lookup tool."""

    query: str = Field(..., description="Free-text query — project name, number, or description fragment")
    tenant_id: str = Field(..., description="Tenant ID for namespace isolation")


class ProjectLookupOutput(BaseModel):
    """Top project matches with confidence scores."""

    matches: list[dict] = Field(
        default_factory=list,
        description="List of {project_id, name, number, score} dicts sorted by score desc",
    )


# ─── Tool Schema ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "project_lookup",
    "description": (
        "Fuzzy-match a query string against the tenant's project list. "
        "Returns up to 3 projects ranked by token-overlap score."
    ),
    "input_schema": ProjectLookupInput.model_json_schema(),
    "output_schema": ProjectLookupOutput.model_json_schema(),
}


# ─── Helpers ─────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Lowercase and split into alphanumeric tokens."""
    return {t for t in text.lower().split() if t.isalnum() or "-" in t}


def _score_project(query_tokens: set[str], project: dict) -> float:
    """Score a project based on token overlap with the query.

    Checks project name, number, description, and client. Returns the
    ratio of query tokens found in any of those fields (0.0 to 1.0).
    """
    if not query_tokens:
        return 0.0

    searchable = " ".join([
        project.get("name", ""),
        project.get("number", ""),
        project.get("description", ""),
        project.get("client", ""),
    ])
    project_tokens = _tokenize(searchable)

    matched = query_tokens & project_tokens
    return len(matched) / len(query_tokens)


def _load_projects() -> list[dict]:
    """Load projects from the sample data file."""
    try:
        with open(DATA_PATH) as f:
            data = json.load(f)
        return data.get("projects", [])
    except FileNotFoundError:
        logger.warning("Sample projects file not found", extra={"path": str(DATA_PATH)})
        return []
    except json.JSONDecodeError:
        logger.error("Failed to parse sample projects JSON")
        return []


# ─── Execute ─────────────────────────────────────────────


def execute(input_data: ProjectLookupInput) -> ProjectLookupOutput:
    """Run fuzzy project lookup and return the top 3 matches."""
    projects = _load_projects()
    query_tokens = _tokenize(input_data.query)

    scored = []
    for project in projects:
        score = _score_project(query_tokens, project)
        if score > 0.0:
            scored.append({
                "project_id": project["id"],
                "name": project["name"],
                "number": project["number"],
                "score": round(score, 3),
            })

    scored.sort(key=lambda m: m["score"], reverse=True)
    top_matches = scored[:3]

    logger.info(
        "Project lookup completed",
        extra={
            "query": input_data.query,
            "tenant_id": input_data.tenant_id,
            "matches_found": len(top_matches),
        },
    )

    return ProjectLookupOutput(matches=top_matches)
