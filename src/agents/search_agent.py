"""Smart Search agent — semantic search with RAG and guardrails.

Handles user search queries by:
1. Parsing query intent (keyword vs semantic vs hybrid)
2. Running hybrid search (BM25 + vector)
3. Generating an answer via RAG with citations
4. Running guardrails to verify answer quality
5. Returning answer with confidence and source references
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.agents.guardrails import run_guardrails
from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class SearchAgentResult:
    """Result from the search agent."""

    answer: str
    citations: list[dict] = field(default_factory=list)
    source_chunks: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    search_type: str = "hybrid"
    warnings: list[str] = field(default_factory=list)
    is_safe: bool = True
    trace: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


def detect_query_intent(query: str) -> str:
    """Detect whether a query is best served by keyword, semantic, or hybrid search.

    - Keyword: exact identifiers like RFI-247, project numbers, spec sections
    - Semantic: natural language questions
    - Hybrid: mix of both (default)
    """
    import re

    # Check for exact identifiers
    has_identifier = bool(re.search(
        r'(RFI[- ]?\d+|SUB[- ]?\d+|P-\d{4}-\d+|\d{2}\s\d{2}\s\d{2})',
        query,
        re.IGNORECASE,
    ))

    # Check for question words
    question_words = ["what", "how", "why", "when", "where", "who", "which",
                       "describe", "explain", "tell me"]
    is_question = any(query.lower().startswith(w) for w in question_words)

    if has_identifier and not is_question:
        return "keyword"
    elif is_question and not has_identifier:
        return "semantic"
    else:
        return "hybrid"


def run_search_agent(
    query: str,
    tenant_id: str,
    search_engine=None,
    rag_pipeline=None,
    top_k: int = 10,
) -> SearchAgentResult:
    """Run the search agent pipeline.

    Steps:
    1. Detect query intent
    2. Run search (hybrid by default)
    3. Generate answer with RAG (if available)
    4. Run guardrails on the answer
    5. Return result with confidence and citations
    """
    start = time.perf_counter()
    trace = []

    # Step 1: Query intent detection
    search_type = detect_query_intent(query)
    trace.append({"step": "intent_detection", "result": search_type})

    # Step 2: Search
    # In the full implementation, this calls the hybrid search engine
    # For the portfolio version, we simulate search results
    search_results = []
    retrieval_scores = []

    if search_engine:
        # Production path: use the hybrid search engine
        pass
    else:
        # Demo path: return simulated results
        search_results, retrieval_scores = _simulate_search(query, tenant_id)

    trace.append({
        "step": "search",
        "type": search_type,
        "results_found": len(search_results),
    })

    # Step 3: Generate answer with RAG
    answer = ""
    citations = []

    if rag_pipeline and search_results:
        # Production path: use the RAG pipeline
        pass
    elif search_results:
        # Demo path: build a simple answer from results
        answer, citations = _generate_simple_answer(query, search_results)

    trace.append({
        "step": "answer_generation",
        "answer_length": len(answer),
        "citations_count": len(citations),
    })

    # Step 4: Run guardrails
    source_texts = [r.get("text", "") for r in search_results]
    guardrail_result = run_guardrails(
        answer=answer,
        source_chunks=source_texts,
        retrieval_scores=retrieval_scores,
    )

    trace.append({
        "step": "guardrails",
        "is_safe": guardrail_result.is_safe,
        "confidence": guardrail_result.confidence,
        "warnings_count": len(guardrail_result.warnings),
    })

    duration_ms = (time.perf_counter() - start) * 1000

    # Step 5: Build result
    if not guardrail_result.is_safe:
        # Downgrade confidence and add caveats
        answer = (
            f"Note: This answer has limited confidence. "
            f"Please verify against source documents.\n\n{answer}"
        )

    result = SearchAgentResult(
        answer=answer,
        citations=citations,
        source_chunks=search_results,
        confidence=guardrail_result.confidence,
        search_type=search_type,
        warnings=guardrail_result.warnings,
        is_safe=guardrail_result.is_safe,
        trace=trace,
        duration_ms=round(duration_ms, 2),
    )

    logger.info(
        "search_complete",
        extra={
            "query": query[:100],
            "tenant_id": tenant_id,
            "search_type": search_type,
            "results_found": len(search_results),
            "confidence": guardrail_result.confidence,
            "is_safe": guardrail_result.is_safe,
            "duration_ms": round(duration_ms, 2),
        },
    )

    return result


def _simulate_search(query: str, tenant_id: str) -> tuple[list[dict], list[float]]:
    """Simulate search results for demo/testing."""
    import json
    import os

    results = []
    scores = []

    # Load sample emails and search through them
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_emails")
    data_dir = os.path.normpath(data_dir)

    query_lower = query.lower()

    for filename in os.listdir(data_dir) if os.path.exists(data_dir) else []:
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            emails = json.load(f)

        for email in emails:
            subject = email.get("subject", "").lower()
            body = email.get("body", "").lower()

            # Simple relevance scoring based on word overlap
            query_words = set(query_lower.split())
            text_words = set(f"{subject} {body}".split())
            overlap = len(query_words & text_words) / max(len(query_words), 1)

            if overlap > 0.1:
                results.append({
                    "id": email.get("id"),
                    "text": email.get("body", "")[:500],
                    "title": email.get("subject", ""),
                    "source_type": "email",
                    "score": round(overlap, 3),
                })
                scores.append(overlap)

    # Sort by score and take top results
    paired = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    results = [p[0] for p in paired[:5]]
    scores = [p[1] for p in paired[:5]]

    return results, scores


def _generate_simple_answer(query: str, results: list[dict]) -> tuple[str, list[dict]]:
    """Generate a simple answer from search results (demo mode)."""
    if not results:
        return "No relevant documents found for your query.", []

    # Build answer from top result
    top = results[0]
    answer = (
        f"Based on the project communications, the most relevant information is from "
        f"'{top['title']}'. {top['text'][:300]}"
    )

    citations = [{
        "claim": answer[:100],
        "source_document": top["title"],
        "source_chunk": top["text"][:200],
        "relevance_score": top["score"],
        "verified": True,
    }]

    return answer, citations
