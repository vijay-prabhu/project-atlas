"""Classification agent — determines the type of an AECO email.

This is Node 1 in the filing graph. It reads the email subject and body,
then classifies it into one of the AECO document types (RFI, submittal,
change order, transmittal, meeting minutes, daily report, general).

Uses chain-of-thought prompting for transparent reasoning.
"""

import json
import time
from typing import Optional

from src.agents.state import EmailFilingState
from src.core.observability import get_logger

logger = get_logger(__name__)

CLASSIFICATION_PROMPT = """You are an expert at classifying construction industry emails.

Given an email from an AECO (Architecture, Engineering, Construction, Owner) project, classify it into exactly one category:

- rfi: Request for Information — questions about design, specs, or construction details
- submittal: Shop drawings, product data, or samples submitted for review
- change_order: Changes to scope, cost, or schedule
- transmittal: Document transmissions (drawings, specs, reports)
- meeting_minutes: Minutes from OAC meetings, progress meetings, etc.
- daily_report: Daily construction activity reports
- shop_drawing: Shop drawing submissions or reviews
- general: General correspondence that doesn't fit other categories

Think step by step:
1. Read the subject line for category keywords (RFI, submittal, CO, transmittal, minutes)
2. Read the body for confirmation of the category
3. Check for document numbers (RFI-XXX, SUB-XXX, CO #XX)
4. Determine your confidence (0.0 to 1.0)

Respond in JSON format:
{{
    "category": "<category>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation of why>"
}}"""


def classify_email(state: EmailFilingState) -> dict:
    """Classify an email into an AECO document category.

    This node function reads the email from state, calls the LLM,
    and writes the classification back to state.
    """
    start = time.perf_counter()

    subject = state.get("email_subject", "")
    body = state.get("email_body", "")
    sender = state.get("email_sender", "")

    # Build the user message
    user_message = f"Subject: {subject}\nFrom: {sender}\n\nBody:\n{body[:2000]}"

    # Try rule-based classification first for high-confidence cases
    rule_result = _rule_based_classify(subject, body)
    if rule_result and rule_result["confidence"] >= 0.95:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "classification_complete",
            extra={
                "agent_name": "classifier",
                "method": "rule_based",
                "category": rule_result["category"],
                "confidence": rule_result["confidence"],
                "duration_ms": round(duration_ms, 2),
            },
        )
        return {
            "classification": rule_result["category"],
            "classification_confidence": rule_result["confidence"],
            "classification_reasoning": rule_result["reasoning"],
            "current_agent": "classifier",
            "agent_trace": state.get("agent_trace", []) + [{
                "agent": "classifier",
                "action": "classify_email",
                "method": "rule_based",
                "duration_ms": round(duration_ms, 2),
                "result": rule_result["category"],
                "confidence": rule_result["confidence"],
            }],
        }

    # Fall back to LLM classification
    # In production, this calls the LLM via the model router
    # For the portfolio version, we use the rule-based result or a simulated response
    llm_result = _llm_classify(subject, body, sender)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "classification_complete",
        extra={
            "agent_name": "classifier",
            "method": "llm",
            "category": llm_result["category"],
            "confidence": llm_result["confidence"],
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {
        "classification": llm_result["category"],
        "classification_confidence": llm_result["confidence"],
        "classification_reasoning": llm_result["reasoning"],
        "current_agent": "classifier",
        "agent_trace": state.get("agent_trace", []) + [{
            "agent": "classifier",
            "action": "classify_email",
            "method": "llm",
            "duration_ms": round(duration_ms, 2),
            "result": llm_result["category"],
            "confidence": llm_result["confidence"],
        }],
    }


def _rule_based_classify(subject: str, body: str) -> Optional[dict]:
    """Fast rule-based classification for obvious cases.

    Checks subject line for unambiguous category indicators.
    Returns None if uncertain — lets the LLM handle ambiguous cases.
    """
    subject_lower = subject.lower()
    body_lower = body.lower()

    # RFI patterns
    if any(p in subject_lower for p in ["rfi-", "rfi #", "rfi ", "request for information"]):
        return {
            "category": "rfi",
            "confidence": 0.95,
            "reasoning": f"Subject contains RFI reference: '{subject}'",
        }

    # Submittal patterns
    if any(p in subject_lower for p in ["submittal", "sub-", "shop drawing submittal"]):
        return {
            "category": "submittal",
            "confidence": 0.95,
            "reasoning": f"Subject contains submittal reference: '{subject}'",
        }

    # Change order patterns
    if any(p in subject_lower for p in ["change order", "co #", "co#", "change notice"]):
        return {
            "category": "change_order",
            "confidence": 0.95,
            "reasoning": f"Subject contains change order reference: '{subject}'",
        }

    # Transmittal patterns
    if any(p in subject_lower for p in ["transmittal", "drawing transmittal", "revised drawings"]):
        return {
            "category": "transmittal",
            "confidence": 0.95,
            "reasoning": f"Subject contains transmittal reference: '{subject}'",
        }

    # Meeting minutes patterns
    if any(p in subject_lower for p in ["meeting minutes", "mtg minutes", "minutes -", "oac meeting"]):
        return {
            "category": "meeting_minutes",
            "confidence": 0.95,
            "reasoning": f"Subject contains meeting minutes reference: '{subject}'",
        }

    # Daily report patterns
    if any(p in subject_lower for p in ["daily report", "daily log", "field report"]):
        return {
            "category": "daily_report",
            "confidence": 0.95,
            "reasoning": f"Subject contains daily report reference: '{subject}'",
        }

    return None


def _llm_classify(subject: str, body: str, sender: str) -> dict:
    """LLM-based classification for ambiguous emails.

    In production, this calls the actual LLM via the model router.
    For testing/demo, it uses enhanced heuristics as a simulation.
    """
    subject_lower = subject.lower()
    body_lower = body.lower()

    # Check body for category signals when subject is ambiguous
    rfi_signals = sum(1 for p in ["rfi", "clarification", "please advise", "design intent",
                                   "please confirm", "conflict", "discrepancy"]
                      if p in body_lower)
    submittal_signals = sum(1 for p in ["submittal", "shop drawing", "product data",
                                         "spec section", "shop drawings", "sample"]
                           if p in body_lower)
    co_signals = sum(1 for p in ["change order", "additional cost", "unforeseen",
                                  "scope change", "price", "change notice"]
                     if p in body_lower)
    general_signals = sum(1 for p in ["schedule", "update", "fyi", "reminder", "coordination"]
                         if p in body_lower)

    scores = {
        "rfi": rfi_signals * 0.2,
        "submittal": submittal_signals * 0.2,
        "change_order": co_signals * 0.2,
        "general": general_signals * 0.15 + 0.1,  # slight bias toward general
    }

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    confidence = min(0.5 + best_score, 0.9)  # LLM classifications cap at 0.9 for humility

    return {
        "category": best_category,
        "confidence": round(confidence, 2),
        "reasoning": f"Body analysis: strongest signals for '{best_category}' "
                     f"(score: {best_score:.2f}). Subject: '{subject}'",
    }
