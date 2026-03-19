"""Citation extraction and source verification.

Parses [Source: ...] references from LLM answers, matches them to actual
retrieved chunks, and verifies that the claims are actually supported by
the source text.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)

# Pattern to match [Source: document_name, section] or [Source N: ...]
_CITATION_PATTERN = re.compile(
    r"\[Source(?:\s+\d+)?:\s*([^\]]+)\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    """A citation linking a claim in the answer to a source document."""

    claim: str
    source_document: str
    source_chunk: str = ""
    relevance_score: float = 0.0
    verified: bool = False


def extract_citations(answer: str, source_chunks: list) -> list[Citation]:
    """Parse [Source: ...] references from an LLM answer and match to chunks.

    For each citation found in the answer:
    1. Extract the source reference text
    2. Find the sentence(s) around the citation (the "claim")
    3. Match the reference to the best source chunk
    4. Create a Citation object

    Args:
        answer: The LLM-generated answer text containing [Source: ...] refs.
        source_chunks: List of SourceChunk objects from retrieval.

    Returns:
        List of Citation objects with source_chunk text filled in.
    """
    citations: list[Citation] = []
    matches = list(_CITATION_PATTERN.finditer(answer))

    if not matches:
        return citations

    for match in matches:
        ref_text = match.group(1).strip()
        claim = _extract_claim(answer, match.start(), match.end())

        # Parse document name and section from the reference
        doc_name, section = _parse_reference(ref_text)

        # Find the best matching source chunk
        best_chunk_text = ""
        best_score = 0.0

        for chunk in source_chunks:
            # Get source_document and source_section — handles both
            # SourceChunk dataclass and plain dict
            chunk_doc = _get_attr(chunk, "source_document", "")
            chunk_section = _get_attr(chunk, "source_section", "")
            chunk_text = _get_attr(chunk, "text", "")

            score = _match_score(doc_name, section, chunk_doc, chunk_section)
            if score > best_score:
                best_score = score
                best_chunk_text = chunk_text

        citations.append(Citation(
            claim=claim,
            source_document=doc_name,
            source_chunk=best_chunk_text,
            relevance_score=round(best_score, 3),
        ))

    logger.info(
        "Citations extracted",
        extra={"citation_count": len(citations)},
    )
    return citations


def verify_claim(claim: str, source_chunk: str) -> tuple[bool, float]:
    """Check if a claim is supported by the source chunk.

    Uses keyword overlap as a baseline check. Counts how many significant
    words from the claim appear in the source text. A higher overlap ratio
    means the source more likely supports the claim.

    Args:
        claim: The factual claim from the answer.
        source_chunk: The source text to verify against.

    Returns:
        (verified, confidence) — verified is True if confidence >= 0.3
    """
    if not claim or not source_chunk:
        return False, 0.0

    claim_tokens = _tokenize(claim)
    source_tokens = _tokenize(source_chunk)

    if not claim_tokens:
        return False, 0.0

    # Count how many claim tokens appear in the source
    matched = claim_tokens & source_tokens
    overlap_ratio = len(matched) / len(claim_tokens)

    # Also check for exact phrase matches (numbers, spec refs, etc.)
    # These are strong signals that the source supports the claim
    numbers_in_claim = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
    numbers_in_source = set(re.findall(r"\b\d+(?:\.\d+)?\b", source_chunk))
    if numbers_in_claim:
        number_match = len(numbers_in_claim & numbers_in_source) / len(numbers_in_claim)
        # Boost confidence if numbers match
        overlap_ratio = (overlap_ratio + number_match) / 2

    confidence = round(min(overlap_ratio, 1.0), 3)
    verified = confidence >= 0.3

    return verified, confidence


def verify_all_claims(citations: list[Citation]) -> list[Citation]:
    """Run verify_claim on each citation and set the verified flag.

    Returns the same citation objects with updated verified and
    relevance_score fields.
    """
    for citation in citations:
        verified, confidence = verify_claim(
            citation.claim, citation.source_chunk,
        )
        citation.verified = verified
        citation.relevance_score = confidence

    verified_count = sum(1 for c in citations if c.verified)
    logger.info(
        "Claims verified",
        extra={
            "total": len(citations),
            "verified": verified_count,
            "unverified": len(citations) - verified_count,
        },
    )

    return citations


# ─── Helpers ─────────────────────────────────────────────


# Common stop words to exclude from token matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "this", "that", "these", "those", "it", "its",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase, split, and filter stop words."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _extract_claim(text: str, cite_start: int, cite_end: int) -> str:
    """Extract the sentence containing the citation reference.

    Looks backward from the citation to find the sentence start, and
    forward to find the sentence end.
    """
    # Find the start of the sentence (look for period, newline, or start of text)
    sent_start = max(
        text.rfind(". ", 0, cite_start) + 2,
        text.rfind("\n", 0, cite_start) + 1,
        0,
    )

    # Find the end of the sentence
    next_period = text.find(". ", cite_end)
    next_newline = text.find("\n", cite_end)

    candidates = [pos for pos in [next_period, next_newline] if pos >= 0]
    if candidates:
        sent_end = min(candidates) + 1
    else:
        sent_end = len(text)

    claim = text[sent_start:sent_end].strip()
    # Remove the citation reference itself from the claim text
    claim = _CITATION_PATTERN.sub("", claim).strip()
    return claim


def _parse_reference(ref_text: str) -> tuple[str, str]:
    """Parse 'document_name, section' from a citation reference string."""
    parts = [p.strip() for p in ref_text.split(",", 1)]
    doc_name = parts[0] if parts else ref_text
    section = parts[1] if len(parts) > 1 else ""
    return doc_name, section


def _match_score(
    doc_name: str,
    section: str,
    chunk_doc: str,
    chunk_section: str,
) -> float:
    """Score how well a citation reference matches a source chunk.

    Returns 0.0 to 1.0. Exact document match is weighted heavily.
    Section match adds a bonus.
    """
    if not doc_name:
        return 0.0

    score = 0.0

    # Document name matching — use token overlap
    doc_tokens = _tokenize(doc_name)
    chunk_doc_tokens = _tokenize(chunk_doc)
    if doc_tokens and chunk_doc_tokens:
        overlap = len(doc_tokens & chunk_doc_tokens)
        score = overlap / max(len(doc_tokens), 1)
    elif doc_name.lower() in chunk_doc.lower():
        score = 0.8

    # Section matching adds a bonus
    if section and chunk_section:
        if section.lower() in chunk_section.lower():
            score = min(score + 0.3, 1.0)
        elif chunk_section.lower() in section.lower():
            score = min(score + 0.2, 1.0)

    return score


def _get_attr(obj: object, attr: str, default: str = "") -> str:
    """Get an attribute from an object or dict, with a default."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
