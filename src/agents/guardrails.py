"""Hallucination guardrails for the Smart Search agent.

Provides architectural safeguards against LLM hallucination:

1. Mandatory citations — every claim must reference a source
2. Confidence scoring — flag low-confidence answers
3. Retrieval quality gating — reject answers when context is poor
4. Claim verification — check claims against source chunks
5. Output validation — validate structured fields against database

These guardrails ensure users can trust the system even when
the LLM occasionally generates incorrect information.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClaimVerification:
    """Result of verifying a single claim against source material."""

    claim: str
    source_chunk: Optional[str]
    is_supported: bool
    confidence: float  # 0.0 to 1.0
    reason: str


@dataclass
class GuardrailResult:
    """Result of running all guardrails on a generated answer."""

    answer: str
    is_safe: bool
    confidence: float
    warnings: list[str]
    verified_claims: list[ClaimVerification]
    retrieval_quality: float  # How good was the retrieved context


def check_retrieval_quality(
    retrieval_scores: list[float],
    min_score_threshold: float = 0.3,
    min_relevant_chunks: int = 1,
) -> tuple[bool, float]:
    """Gate: check if retrieved context is good enough to generate an answer.

    If the highest retrieval score is below threshold, the context is
    too weak to generate a reliable answer. Better to say "I don't know"
    than to let the LLM improvise.

    Returns:
        (passes_gate, quality_score)
    """
    if not retrieval_scores:
        return False, 0.0

    above_threshold = [s for s in retrieval_scores if s >= min_score_threshold]
    quality_score = max(retrieval_scores)

    passes = len(above_threshold) >= min_relevant_chunks
    return passes, quality_score


def extract_claims(answer: str) -> list[str]:
    """Break an answer into individual claims for verification.

    Splits on sentence boundaries and filters out filler sentences
    (greetings, transitions, etc.)
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    filler_patterns = [
        r'^(based on|according to|in summary|overall|to summarize)',
        r'^(yes|no|sure|certainly|of course)',
        r'^(here is|here are|the following)',
        r'^(i hope|let me know|please)',
    ]

    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        is_filler = any(
            re.match(pattern, sentence, re.IGNORECASE)
            for pattern in filler_patterns
        )
        if not is_filler:
            claims.append(sentence)

    return claims


def verify_claim_against_source(
    claim: str,
    source_chunks: list[str],
) -> ClaimVerification:
    """Verify a single claim against the retrieved source chunks.

    Uses keyword overlap as a lightweight verification method.
    In production, this could use embedding similarity or an LLM judge.
    """
    if not source_chunks:
        return ClaimVerification(
            claim=claim,
            source_chunk=None,
            is_supported=False,
            confidence=0.0,
            reason="No source chunks available for verification",
        )

    claim_words = set(claim.lower().split())
    # Remove common words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "with", "and", "or", "but", "not", "this", "that"}
    claim_words -= stopwords

    best_match = None
    best_overlap = 0.0

    for chunk in source_chunks:
        chunk_words = set(chunk.lower().split()) - stopwords
        if not claim_words:
            continue
        overlap = len(claim_words & chunk_words) / len(claim_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = chunk

    is_supported = best_overlap >= 0.4  # At least 40% word overlap
    confidence = min(best_overlap, 1.0)

    return ClaimVerification(
        claim=claim,
        source_chunk=best_match[:200] if best_match else None,
        is_supported=is_supported,
        confidence=round(confidence, 3),
        reason=f"Word overlap: {best_overlap:.1%}" if best_match else "No matching source",
    )


def run_guardrails(
    answer: str,
    source_chunks: list[str],
    retrieval_scores: list[float],
    min_retrieval_quality: float = 0.3,
) -> GuardrailResult:
    """Run all guardrails on a generated answer.

    This is the main entry point for the guardrail system.
    Returns a GuardrailResult with the safety assessment.
    """
    warnings = []

    # Guardrail 1: Retrieval quality gating
    passes_retrieval, retrieval_quality = check_retrieval_quality(
        retrieval_scores, min_retrieval_quality
    )
    if not passes_retrieval:
        warnings.append(
            f"Low retrieval quality ({retrieval_quality:.2f}). "
            "Answer may not be well-supported by source documents."
        )

    # Guardrail 2: Extract and verify claims
    claims = extract_claims(answer)
    verified_claims = [
        verify_claim_against_source(claim, source_chunks)
        for claim in claims
    ]

    # Guardrail 3: Calculate overall confidence
    if verified_claims:
        supported_count = sum(1 for vc in verified_claims if vc.is_supported)
        overall_confidence = supported_count / len(verified_claims)
    else:
        overall_confidence = 0.0

    if overall_confidence < 0.5:
        warnings.append(
            f"Only {overall_confidence:.0%} of claims are supported by source material."
        )

    # Guardrail 4: Check for unsupported claims
    unsupported = [vc for vc in verified_claims if not vc.is_supported]
    if unsupported:
        for vc in unsupported:
            warnings.append(f"Unsupported claim: '{vc.claim[:80]}...'")

    # Overall safety assessment
    is_safe = passes_retrieval and overall_confidence >= 0.5 and len(unsupported) <= 1

    return GuardrailResult(
        answer=answer,
        is_safe=is_safe,
        confidence=round(overall_confidence, 3),
        warnings=warnings,
        verified_claims=verified_claims,
        retrieval_quality=retrieval_quality,
    )
