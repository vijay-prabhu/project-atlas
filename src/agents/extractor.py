"""Metadata extraction agent — pulls structured data from AECO emails.

This is Node 2 in the filing graph. It takes the classified email and
extracts: project number, discipline, RFI/submittal numbers, spec sections,
urgency level, and action items.

Uses chain-of-thought reasoning for transparent, auditable extraction.
"""

import re
import time
from typing import Optional

from src.agents.state import EmailFilingState
from src.core.observability import get_logger

logger = get_logger(__name__)

# Chain-of-thought extraction prompt
EXTRACTION_PROMPT = """You are an AECO document metadata extractor. Given an email that has been classified as '{category}', extract structured metadata.

Think step by step:
1. First, identify the project — look for project numbers (P-XXXX-XXXX), project names, or client references
2. Then, identify the discipline — architectural, structural, mechanical, electrical, plumbing, civil, or general
3. Look for document references — RFI numbers (RFI-XXX), submittal numbers (SUB-XXX), spec sections (XX XX XX)
4. Assess urgency — look for deadline mentions, "ASAP", "critical path", "urgent"
5. List any action items — things someone needs to do

Email Subject: {subject}
Email From: {sender}
Email Body:
{body}

Respond in JSON:
{{
    "project_number": "<P-XXXX-XXXX or null>",
    "project_name": "<name or null>",
    "discipline": "<discipline or general>",
    "rfi_number": "<RFI-XXX or null>",
    "submittal_number": "<SUB-XXX or null>",
    "spec_section": "<XX XX XX or null>",
    "urgency": "<urgent|high|normal|low>",
    "action_items": ["<action 1>", "<action 2>"],
    "reasoning": "<your step-by-step reasoning>"
}}"""


def extract_metadata(state: EmailFilingState) -> dict:
    """Extract structured metadata from a classified email.

    Reads classification from state, applies extraction rules,
    and writes extracted fields back to state.
    """
    start = time.perf_counter()

    subject = state.get("email_subject", "")
    body = state.get("email_body", "")
    sender = state.get("email_sender", "")
    category = state.get("classification", "general")

    # Run extraction
    result = _extract(subject, body, sender, category)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "extraction_complete",
        extra={
            "agent_name": "extractor",
            "project_number": result.get("project_number"),
            "rfi_number": result.get("rfi_number"),
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {
        "extracted_project_number": result.get("project_number"),
        "extracted_project_name": result.get("project_name"),
        "extracted_rfi_number": result.get("rfi_number"),
        "extracted_submittal_number": result.get("submittal_number"),
        "extracted_discipline": result.get("discipline"),
        "extracted_urgency": result.get("urgency", "normal"),
        "extraction_reasoning": result.get("reasoning", ""),
        "current_agent": "extractor",
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "extractor",
            "action": "extract_metadata",
            "duration_ms": round(duration_ms, 2),
            "project_number": result.get("project_number"),
            "rfi_number": result.get("rfi_number"),
            "discipline": result.get("discipline"),
        }],
    }


def _extract(subject: str, body: str, sender: str, category: str) -> dict:
    """Extract metadata using regex patterns and heuristics.

    In production, this calls the LLM with the chain-of-thought
    EXTRACTION_PROMPT. For the portfolio version, we use robust
    regex extraction that demonstrates the same logic.
    """
    full_text = f"{subject}\n{body}"
    reasoning_steps = []

    # Step 1: Extract project number
    project_number = _extract_project_number(full_text)
    reasoning_steps.append(
        f"Step 1 - Project number: Found '{project_number}'" if project_number
        else "Step 1 - Project number: No explicit project number found"
    )

    # Step 2: Extract project name
    project_name = _extract_project_name(full_text)
    reasoning_steps.append(
        f"Step 2 - Project name: Inferred '{project_name}'" if project_name
        else "Step 2 - Project name: Could not determine project name"
    )

    # Step 3: Identify discipline
    discipline = _extract_discipline(full_text)
    reasoning_steps.append(f"Step 3 - Discipline: Identified as '{discipline}'")

    # Step 4: Extract document references
    rfi_number = _extract_rfi_number(full_text)
    submittal_number = _extract_submittal_number(full_text)
    spec_section = _extract_spec_section(full_text)
    reasoning_steps.append(
        f"Step 4 - References: RFI={rfi_number}, Submittal={submittal_number}, Spec={spec_section}"
    )

    # Step 5: Assess urgency
    urgency = _assess_urgency(full_text)
    reasoning_steps.append(f"Step 5 - Urgency: Assessed as '{urgency}'")

    return {
        "project_number": project_number,
        "project_name": project_name,
        "discipline": discipline,
        "rfi_number": rfi_number,
        "submittal_number": submittal_number,
        "spec_section": spec_section,
        "urgency": urgency,
        "reasoning": " → ".join(reasoning_steps),
    }


def _extract_project_number(text: str) -> Optional[str]:
    """Look for project number patterns like P-2024-0847."""
    match = re.search(r'P-\d{4}-\d{3,4}', text)
    return match.group(0) if match else None


def _extract_project_name(text: str) -> Optional[str]:
    """Infer project name from known project references in the text."""
    text_lower = text.lower()
    known_projects = {
        "waterfront": "Waterfront Mixed-Use Tower",
        "city hall": "City Hall Renovation Phase 2",
        "highway 401": "Highway 401 Bridge Replacement",
        "westfield": "Westfield Medical Center Expansion",
        "oakridge": "Oakridge Elementary School",
        "distribution center": "Industrial Distribution Center",
        "industrial distribution": "Industrial Distribution Center",
        "medical center": "Westfield Medical Center Expansion",
        "bridge replacement": "Highway 401 Bridge Replacement",
        "elementary school": "Oakridge Elementary School",
    }
    for keyword, name in known_projects.items():
        if keyword in text_lower:
            return name
    return None


def _extract_discipline(text: str) -> str:
    """Identify the engineering discipline from content."""
    text_lower = text.lower()
    discipline_signals = {
        "structural": ["structural", "steel", "concrete", "foundation", "beam", "column",
                        "rebar", "shoring", "seismic", "framing"],
        "mechanical": ["hvac", "mechanical", "diffuser", "ductwork", "air handling",
                        "chiller", "boiler", "ventilation", "cooling"],
        "electrical": ["electrical", "panelboard", "conduit", "switchgear", "transformer",
                        "lighting", "power", "480v", "277v"],
        "architectural": ["curtain wall", "facade", "window", "door", "ceiling",
                           "flooring", "partition", "finish"],
        "plumbing": ["plumbing", "piping", "drainage", "water", "sanitary",
                      "medical gas", "sprinkler"],
        "civil": ["grading", "excavation", "soil", "geotechnical", "paving",
                   "stormwater", "utility", "site work"],
        "landscape": ["landscape", "planting", "irrigation", "topsoil", "tree"],
    }

    scores = {}
    for discipline, keywords in discipline_signals.items():
        scores[discipline] = sum(1 for kw in keywords if kw in text_lower)

    if not any(scores.values()):
        return "general"

    return max(scores, key=scores.get)


def _extract_rfi_number(text: str) -> Optional[str]:
    """Extract RFI number from text."""
    patterns = [
        r'RFI[- #]?(\d{2,4})',
        r'Request for Information[- #]?(\d{2,4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"RFI-{match.group(1)}"
    return None


def _extract_submittal_number(text: str) -> Optional[str]:
    """Extract submittal number from text."""
    patterns = [
        r'SUB[- #]?(\d{2,4})',
        r'Submittal[- #]?(\d{2,4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"SUB-{match.group(1)}"
    return None


def _extract_spec_section(text: str) -> Optional[str]:
    """Extract CSI spec section number (XX XX XX format)."""
    match = re.search(r'\b(\d{2}\s\d{2}\s\d{2})\b', text)
    return match.group(1) if match else None


def _assess_urgency(text: str) -> str:
    """Assess urgency from content signals."""
    text_lower = text.lower()

    urgent_signals = ["asap", "urgent", "immediately", "critical path", "today",
                       "by end of day", "emergency"]
    high_signals = ["by end of next week", "time sensitive", "deadline",
                     "schedule impact", "please prioritize", "outstanding for"]
    normal_signals = ["please review", "when you get a chance", "for your review"]

    if any(s in text_lower for s in urgent_signals):
        return "urgent"
    if any(s in text_lower for s in high_signals):
        return "high"
    return "normal"
