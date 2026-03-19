"""Cross-encoder reranking of search results using an LLM.

After the initial retrieval (BM25 + vector), a reranker re-scores each
result against the query using a more expensive but more accurate model.
This is the standard retrieve-then-rerank pattern.

Falls back to original scores if the LLM call fails, so search still
works even when the reranking step has issues.
"""

import json

from src.core.observability import get_logger
from src.llm.client import LLMClient

logger = get_logger(__name__)

_RERANK_PROMPT = """Score how relevant this document is to the query.
Return ONLY a JSON object: {{"score": <float between 0.0 and 1.0>}}

Query: {query}

Document: {document}
"""


class Reranker:
    """LLM-based reranker for search results.

    Uses the LLM as a cross-encoder — it sees both the query and each
    candidate document, then scores relevance from 0 to 1. More accurate
    than embedding similarity alone, but slower and more expensive.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def _score_single(self, query: str, document_text: str) -> float | None:
        """Ask the LLM to score a single query-document pair.

        Returns a float between 0.0 and 1.0, or None if scoring fails.
        """
        prompt = _RERANK_PROMPT.format(query=query, document=document_text)

        try:
            response = self._llm.generate(
                prompt=prompt,
                system_prompt="You are a relevance scoring system. Return only valid JSON.",
                temperature=0.0,
                max_tokens=32,
            )
            parsed = json.loads(response.content)
            score = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Reranker score parse failed: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"Reranker LLM call failed: {exc}")
            return None

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank search results using LLM-based relevance scoring.

        Each result dict should have at least an 'id' and either 'text',
        'chunk_text', or a 'metadata.text' field to score against.

        Falls back to original ordering if all LLM calls fail.
        """
        if not results:
            return []

        scored_results = []
        fallback_needed = True

        for result in results:
            # Try to find the text content in the result
            text = (
                result.get("text")
                or result.get("chunk_text")
                or result.get("metadata", {}).get("text", "")
            )

            if not text:
                scored_results.append(result)
                continue

            llm_score = self._score_single(query, text)

            reranked = dict(result)
            if llm_score is not None:
                reranked["rerank_score"] = llm_score
                fallback_needed = False
            scored_results.append(reranked)

        if fallback_needed:
            logger.warning("All reranker scores failed — keeping original order")
            return results[:top_k]

        # Sort by rerank_score (results without a score go to the end)
        scored_results.sort(
            key=lambda r: r.get("rerank_score", -1.0),
            reverse=True,
        )

        logger.info(
            "Reranking completed",
            extra={
                "input_count": len(results),
                "output_count": min(len(scored_results), top_k),
            },
        )

        return scored_results[:top_k]
