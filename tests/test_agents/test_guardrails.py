"""Tests for hallucination guardrails.

Covers Q12: Architectural guardrails for Smart Search.
"""

import pytest

from src.agents.guardrails import (
    check_retrieval_quality,
    extract_claims,
    run_guardrails,
    verify_claim_against_source,
)


class TestRetrievalQualityGating:
    """Guardrail 1: Don't generate answers from poor context."""

    def test_good_retrieval_passes(self):
        scores = [0.85, 0.72, 0.65, 0.45, 0.30]
        passes, quality = check_retrieval_quality(scores)
        assert passes is True
        assert quality == 0.85

    def test_poor_retrieval_fails(self):
        scores = [0.15, 0.10, 0.08]
        passes, quality = check_retrieval_quality(scores)
        assert passes is False
        assert quality < 0.3

    def test_empty_scores_fails(self):
        passes, quality = check_retrieval_quality([])
        assert passes is False
        assert quality == 0.0

    def test_custom_threshold(self):
        scores = [0.4, 0.35, 0.25]
        passes, _ = check_retrieval_quality(scores, min_score_threshold=0.5)
        assert passes is False


class TestClaimExtraction:
    """Break answers into verifiable claims."""

    def test_extracts_multiple_claims(self):
        answer = (
            "The steel connection at Grid J-7 uses a bolted end plate detail. "
            "The RFI was submitted on March 15, 2026. "
            "Pacific Steel Erectors is the contractor."
        )
        claims = extract_claims(answer)
        assert len(claims) == 3

    def test_filters_filler_sentences(self):
        answer = (
            "Based on the documents, here is the answer. "
            "The RFI number is 247. "
            "I hope this helps."
        )
        claims = extract_claims(answer)
        assert len(claims) == 1
        assert "247" in claims[0]

    def test_handles_empty_answer(self):
        claims = extract_claims("")
        assert len(claims) == 0


class TestClaimVerification:
    """Verify claims against source material."""

    def test_supported_claim_verified(self):
        claim = "The steel connection at Grid J-7 uses a bolted end plate detail."
        sources = [
            "RFI-247 regarding the steel connection detail at Grid J-7. "
            "The current detail shows a bolted end plate connection."
        ]
        result = verify_claim_against_source(claim, sources)
        assert result.is_supported is True
        assert result.confidence > 0.3

    def test_unsupported_claim_rejected(self):
        claim = "The project was completed in January 2025."
        sources = [
            "The curtain wall mock-up testing is scheduled for March 25, 2026."
        ]
        result = verify_claim_against_source(claim, sources)
        assert result.is_supported is False

    def test_no_sources_returns_unsupported(self):
        claim = "Some claim about the project."
        result = verify_claim_against_source(claim, [])
        assert result.is_supported is False
        assert result.confidence == 0.0


class TestFullGuardrails:
    """End-to-end guardrail tests."""

    def test_safe_answer_passes(self):
        answer = "The steel connection at Grid J-7 uses a bolted end plate."
        sources = ["The detail shows a bolted end plate connection at Grid J-7."]
        scores = [0.85]

        result = run_guardrails(answer, sources, scores)
        assert result.is_safe is True
        assert result.confidence > 0.0
        assert len(result.warnings) == 0

    def test_hallucinated_answer_flagged(self):
        answer = (
            "The project was cancelled in 2024. "
            "All contractors were dismissed. "
            "The budget was $500 million."
        )
        sources = ["RFI-247 steel connection detail for Waterfront Tower."]
        scores = [0.85]

        result = run_guardrails(answer, sources, scores)
        assert result.confidence < 0.5
        assert len(result.warnings) > 0

    def test_poor_retrieval_adds_warning(self):
        answer = "The answer based on weak context."
        sources = ["Some barely relevant text."]
        scores = [0.15]

        result = run_guardrails(answer, sources, scores)
        assert result.is_safe is False
        assert any("retrieval quality" in w.lower() for w in result.warnings)
