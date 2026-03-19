"""Filing decision agent — decides where to file an email.

This is Node 3 in the filing graph. It takes the classification and
extracted metadata, calls tools (project_lookup, rfi_matcher, sender_history),
and makes a filing decision with a confidence score.

The confidence score drives the routing:
- >= 0.85: auto-file (no human needed)
- 0.5 - 0.85: needs human review (HITL)
- < 0.5: flagged (too uncertain to file)

Priority hierarchy for conflicts: flagged > needs_review > auto_file
"""

import time
from typing import Optional

from src.agents.state import EmailFilingState
from src.core.observability import get_logger
from src.tools.project_lookup import execute as lookup_project
from src.tools.project_lookup import ProjectLookupInput
from src.tools.rfi_matcher import execute as match_rfi
from src.tools.rfi_matcher import RFIMatcherInput
from src.tools.sender_history import execute as get_sender_history
from src.tools.sender_history import SenderHistoryInput

logger = get_logger(__name__)


def make_filing_decision(state: EmailFilingState) -> dict:
    """Decide where to file the email based on all gathered evidence.

    Calls tools to verify and enrich the extraction results,
    then combines signals to make a confident filing decision.
    """
    start = time.perf_counter()
    trace_entries = []

    tenant_id = state.get("tenant_id", "demo")
    email_subject = state.get("email_subject", "")
    email_body = state.get("email_body", "")
    email_sender = state.get("email_sender", "")

    project_number = state.get("extracted_project_number")
    project_name = state.get("extracted_project_name")
    rfi_number = state.get("extracted_rfi_number")
    classification = state.get("classification", "general")
    classification_confidence = state.get("classification_confidence", 0.0)

    # ── Tool 1: Project Lookup ──────────────────────────────
    query = project_number or project_name or email_subject
    project_results = lookup_project(
        ProjectLookupInput(query=query, tenant_id=tenant_id)
    )
    project_matches = project_results.matches
    trace_entries.append({
        "tool": "project_lookup",
        "query": query,
        "matches_found": len(project_matches),
    })

    # ── Tool 2: RFI Matcher ─────────────────────────────────
    rfi_matches = []
    if rfi_number or classification == "rfi":
        best_project_id = project_matches[0]["project_id"] if project_matches else None
        rfi_results = match_rfi(
            RFIMatcherInput(
                email_subject=email_subject,
                email_body=email_body,
                project_id=best_project_id,
            )
        )
        rfi_matches = rfi_results.matches
        trace_entries.append({
            "tool": "rfi_matcher",
            "rfi_number": rfi_number,
            "matches_found": len(rfi_matches),
        })

    # ── Tool 3: Sender History ──────────────────────────────
    sender_results = get_sender_history(
        SenderHistoryInput(sender_email=email_sender)
    )
    sender_patterns = sender_results.filing_patterns
    trace_entries.append({
        "tool": "sender_history",
        "sender": email_sender,
        "known_projects": len(sender_patterns),
    })

    # ── Combine signals into filing decision ────────────────
    decision = _compute_decision(
        project_matches=project_matches,
        rfi_matches=rfi_matches,
        sender_patterns=sender_patterns,
        classification=classification,
        classification_confidence=classification_confidence,
        project_number=project_number,
    )

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "filing_decision_complete",
        extra={
            "agent_name": "filer",
            "action": decision["filing_action"],
            "confidence": decision["filing_confidence"],
            "project_id": decision.get("filing_project_id"),
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {
        "project_lookup_results": project_matches,
        "rfi_match_results": rfi_matches,
        "sender_history_results": sender_patterns,
        "filing_action": decision["filing_action"],
        "filing_project_id": decision.get("filing_project_id"),
        "filing_folder_path": decision.get("filing_folder_path"),
        "filing_confidence": decision["filing_confidence"],
        "filing_reasoning": decision["filing_reasoning"],
        "current_agent": "filer",
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "filer",
            "action": "make_filing_decision",
            "duration_ms": round(duration_ms, 2),
            "tools_called": trace_entries,
            "result": decision["filing_action"],
            "confidence": decision["filing_confidence"],
        }],
    }


def _compute_decision(
    project_matches: list[dict],
    rfi_matches: list[dict],
    sender_patterns: list[dict],
    classification: str,
    classification_confidence: float,
    project_number: Optional[str],
) -> dict:
    """Combine all signals into a filing decision with confidence.

    Signal weights:
    - Project number match: +0.3
    - RFI/submittal number match: +0.2
    - Sender history match: +0.15
    - Classification confidence: +0.2
    - Top project match score: +0.15
    """
    confidence = 0.0
    reasoning_parts = []
    best_project_id = None
    folder_path = None

    # Signal 1: Direct project match
    if project_matches and project_matches[0]["score"] > 0.5:
        best_match = project_matches[0]
        best_project_id = best_match["project_id"]
        confidence += min(best_match["score"] * 0.3, 0.3)
        reasoning_parts.append(
            f"Project match: '{best_match['name']}' (score: {best_match['score']:.2f})"
        )

    # Signal 2: RFI/submittal number match
    if rfi_matches and rfi_matches[0]["score"] > 0.5:
        confidence += 0.2
        rfi_match = rfi_matches[0]
        # If RFI match gives us a project, prefer it
        if not best_project_id:
            best_project_id = rfi_match.get("project_id")
        reasoning_parts.append(
            f"RFI match: '{rfi_match['number']}' (score: {rfi_match['score']:.2f})"
        )

    # Signal 3: Sender history
    if sender_patterns:
        for pattern in sender_patterns:
            if pattern["project_id"] == best_project_id:
                confidence += 0.15
                reasoning_parts.append(
                    f"Sender history confirms: {pattern['count']} previous emails to this project"
                )
                break
        else:
            # Sender known but for different projects
            if not best_project_id and sender_patterns:
                best_project_id = sender_patterns[0]["project_id"]
                confidence += 0.1
                reasoning_parts.append(
                    f"Sender typically files to: '{sender_patterns[0]['project_name']}'"
                )

    # Signal 4: Classification confidence
    confidence += classification_confidence * 0.2
    reasoning_parts.append(
        f"Classification '{classification}' (confidence: {classification_confidence:.2f})"
    )

    # Signal 5: Explicit project number in email
    if project_number:
        confidence += 0.15
        reasoning_parts.append(f"Explicit project number found: {project_number}")

    # Cap confidence at 1.0
    confidence = min(confidence, 1.0)

    # Build folder path
    if best_project_id and classification:
        folder_path = f"/{best_project_id}/{classification}"

    # Determine action based on confidence thresholds
    # Priority hierarchy: flagged > needs_review > auto_file
    if confidence >= 0.85:
        action = "auto_file"
    elif confidence >= 0.5:
        action = "needs_review"
    else:
        action = "flagged"

    reasoning = " | ".join(reasoning_parts)

    return {
        "filing_action": action,
        "filing_project_id": best_project_id,
        "filing_folder_path": folder_path,
        "filing_confidence": round(confidence, 3),
        "filing_reasoning": reasoning,
    }
