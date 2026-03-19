"""RAG evaluation framework measuring faithfulness, relevance, and precision.

Provides lightweight metrics that don't require an LLM call. Good enough for
continuous monitoring and regression testing. For deeper evals, pair these
with LLM-as-judge scoring.
"""

import re
from dataclasses import dataclass

from src.core.observability import get_logger

logger = get_logger(__name__)

# Same stop words list as citations.py — keep them in sync
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
    """Lowercase, split into words, drop stop words and short tokens."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _split_claims(text: str) -> list[str]:
    """Split text into individual sentences/claims.

    Splits on sentence-ending punctuation. Filters out very short fragments
    that aren't real claims.
    """
    sentences = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


class RAGMetrics:
    """Lightweight RAG quality metrics: faithfulness, relevance, precision."""

    def faithfulness(self, answer: str, context: str) -> float:
        """Measure if the answer's claims are supported by the context.

        Splits the answer into sentences, then checks what fraction of
        each sentence's key terms appear in the context. Returns the
        average support score across all sentences.

        A score of 1.0 means every claim term appears in the context.
        A score of 0.0 means nothing in the answer matches the context.
        """
        claims = _split_claims(answer)
        if not claims:
            return 0.0

        context_tokens = _tokenize(context)
        if not context_tokens:
            return 0.0

        scores: list[float] = []
        for claim in claims:
            claim_tokens = _tokenize(claim)
            if not claim_tokens:
                continue
            matched = claim_tokens & context_tokens
            scores.append(len(matched) / len(claim_tokens))

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 3)

    def answer_relevance(self, answer: str, question: str) -> float:
        """Measure if the answer addresses the question.

        Checks keyword overlap between the question and the answer.
        Higher overlap means the answer is more likely on-topic.
        """
        question_tokens = _tokenize(question)
        answer_tokens = _tokenize(answer)

        if not question_tokens or not answer_tokens:
            return 0.0

        # How many question terms show up in the answer
        matched = question_tokens & answer_tokens
        coverage = len(matched) / len(question_tokens)

        return round(min(coverage, 1.0), 3)

    def context_precision(
        self,
        retrieved_chunks: list,
        relevant_chunks: list,
    ) -> float:
        """Precision: what fraction of retrieved chunks are relevant.

        Compares retrieved chunk texts against the relevant chunk texts
        using token overlap. A retrieved chunk counts as relevant if its
        overlap with any relevant chunk is above 0.5.
        """
        if not retrieved_chunks:
            return 0.0
        if not relevant_chunks:
            return 0.0

        relevant_count = 0
        relevant_token_sets = [_tokenize(_get_text(c)) for c in relevant_chunks]

        for chunk in retrieved_chunks:
            chunk_tokens = _tokenize(_get_text(chunk))
            if not chunk_tokens:
                continue

            # Check if this retrieved chunk matches any relevant chunk
            for rel_tokens in relevant_token_sets:
                if not rel_tokens:
                    continue
                overlap = len(chunk_tokens & rel_tokens) / max(
                    len(chunk_tokens), len(rel_tokens),
                )
                if overlap > 0.5:
                    relevant_count += 1
                    break

        return round(relevant_count / len(retrieved_chunks), 3)

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        retrieved_chunks: list,
        relevant_chunks: list,
    ) -> dict:
        """Run all three metrics and return a combined score dict.

        Returns:
            {
                "faithfulness": 0.0-1.0,
                "answer_relevance": 0.0-1.0,
                "context_precision": 0.0-1.0,
                "overall": weighted average,
            }
        """
        faith = self.faithfulness(answer, context)
        relevance = self.answer_relevance(answer, question)
        precision = self.context_precision(retrieved_chunks, relevant_chunks)

        # Weighted average: faithfulness matters most for AECO documents
        # where incorrect specs or wrong RFI numbers can cause real problems
        overall = round(
            faith * 0.5 + relevance * 0.3 + precision * 0.2,
            3,
        )

        return {
            "faithfulness": faith,
            "answer_relevance": relevance,
            "context_precision": precision,
            "overall": overall,
        }

    def run_eval_suite(self, test_cases: list[dict]) -> dict:
        """Run evaluation on a list of test cases and return aggregate metrics.

        Each test case should have:
            {
                "question": str,
                "expected_answer": str,  # used as the answer to evaluate
                "context": str,          # optional, defaults to ""
                "relevant_docs": list,   # the ground truth relevant chunks
                "retrieved_docs": list,  # the chunks the system retrieved
            }

        Returns:
            {
                "case_count": int,
                "avg_faithfulness": float,
                "avg_answer_relevance": float,
                "avg_context_precision": float,
                "avg_overall": float,
                "per_case": [
                    {"question": str, "scores": {...}},
                    ...
                ],
            }
        """
        if not test_cases:
            return {
                "case_count": 0,
                "avg_faithfulness": 0.0,
                "avg_answer_relevance": 0.0,
                "avg_context_precision": 0.0,
                "avg_overall": 0.0,
                "per_case": [],
            }

        all_scores: list[dict] = []
        per_case: list[dict] = []

        for case in test_cases:
            question = case.get("question", "")
            answer = case.get("expected_answer", "")
            context = case.get("context", answer)  # use answer as context if not provided
            relevant = case.get("relevant_docs", [])
            retrieved = case.get("retrieved_docs", [])

            scores = self.evaluate(
                question=question,
                answer=answer,
                context=context,
                retrieved_chunks=retrieved,
                relevant_chunks=relevant,
            )
            all_scores.append(scores)
            per_case.append({"question": question, "scores": scores})

        n = len(all_scores)
        result = {
            "case_count": n,
            "avg_faithfulness": round(
                sum(s["faithfulness"] for s in all_scores) / n, 3,
            ),
            "avg_answer_relevance": round(
                sum(s["answer_relevance"] for s in all_scores) / n, 3,
            ),
            "avg_context_precision": round(
                sum(s["context_precision"] for s in all_scores) / n, 3,
            ),
            "avg_overall": round(
                sum(s["overall"] for s in all_scores) / n, 3,
            ),
            "per_case": per_case,
        }

        logger.info(
            "Eval suite completed",
            extra={
                "case_count": n,
                "avg_overall": result["avg_overall"],
            },
        )

        return result


# ─── Helpers ─────────────────────────────────────────────


def _get_text(chunk) -> str:
    """Extract text from a chunk — handles dicts, SourceChunk, or plain strings."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("text", chunk.get("content", ""))
    return getattr(chunk, "text", getattr(chunk, "content", ""))
