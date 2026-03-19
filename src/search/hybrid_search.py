"""BM25 + vector search with Reciprocal Rank Fusion (RRF).

Combines keyword search (BM25) with semantic search (vector similarity)
to get the best of both worlds. BM25 catches exact keyword matches that
embeddings might miss, while vectors handle semantic similarity.

The two result lists are merged using RRF, which is rank-based rather
than score-based — so it works even though BM25 scores and cosine
similarity scores are on different scales.
"""

from typing import Optional

from src.core.observability import get_logger
from src.search.vector_store import VectorStore

logger = get_logger(__name__)

# Standard RRF constant from the original paper (Cormack et al., 2009).
# Higher k smooths out rank differences. 60 is the widely used default.
RRF_K = 60


class HybridSearchEngine:
    """Combines BM25 keyword search with vector similarity search.

    Args:
        vector_store: VectorStore instance for semantic search.
        documents: List of document dicts with at least 'id' and 'text' fields.
            Used to build the BM25 index for keyword search.
    """

    def __init__(self, vector_store: VectorStore, documents: list[dict]) -> None:
        self._vector_store = vector_store
        self._documents = documents
        self._doc_index = {doc["id"]: doc for doc in documents}
        self._bm25 = None

        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build the BM25 index from document texts."""
        if not self._documents:
            return

        try:
            from rank_bm25 import BM25Okapi

            tokenized = [
                doc.get("text", "").lower().split()
                for doc in self._documents
            ]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed — keyword search disabled")
        except Exception as exc:
            logger.warning(f"BM25 index build failed: {exc}")

    # ─── Search Methods ──────────────────────────────────

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Keyword search using BM25.

        Returns list of {id, score, rank} dicts.
        """
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with document IDs and sort
        scored = [
            {"id": doc["id"], "score": float(score)}
            for doc, score in zip(self._documents, scores)
            if score > 0.0
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Add rank (1-based)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank

        return scored[:top_k]

    def _vector_search(
        self,
        query_embedding: list[float],
        namespace: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Vector similarity search via the VectorStore.

        Returns list of {id, score, rank, metadata} dicts.
        """
        results = self._vector_store.query(
            query_vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            filters=filters,
        )

        # Add rank (1-based, already sorted by score from Pinecone)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results

    # ─── Reciprocal Rank Fusion ──────────────────────────

    def _rrf_merge(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        alpha: float,
    ) -> list[dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all lists where the doc appears.
        Alpha controls the weight: alpha applies to vector results,
        (1 - alpha) applies to BM25 results.
        """
        fused_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        # Score BM25 results
        for item in bm25_results:
            doc_id = item["id"]
            rrf_score = (1 - alpha) * (1.0 / (RRF_K + item["rank"]))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
            doc_data[doc_id] = item

        # Score vector results
        for item in vector_results:
            doc_id = item["id"]
            rrf_score = alpha * (1.0 / (RRF_K + item["rank"]))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
            # Vector results may have richer metadata — prefer them
            doc_data[doc_id] = item

        # Sort by fused score
        merged = []
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
            result = dict(doc_data[doc_id])
            result["rrf_score"] = round(score, 6)
            merged.append(result)

        return merged

    # ─── Public API ──────────────────────────────────────

    def search(
        self,
        query: str,
        query_embedding: list[float],
        namespace: str,
        top_k: int = 10,
        alpha: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Run hybrid search: BM25 + vector with RRF fusion.

        Args:
            query: Raw text query for BM25 search.
            query_embedding: Pre-computed embedding vector for semantic search.
            namespace: Tenant namespace for vector store isolation.
            top_k: Number of results to return.
            alpha: Weight for semantic vs keyword. 0.7 = 70% semantic, 30% keyword.
            filters: Optional metadata filters for vector search.

        Returns:
            Deduplicated, RRF-ranked results with scores.
        """
        # Fetch more candidates than needed so RRF has enough to work with
        candidate_k = min(top_k * 3, 50)

        bm25_results = self._bm25_search(query, top_k=candidate_k)
        vector_results = self._vector_search(
            query_embedding, namespace, top_k=candidate_k, filters=filters,
        )

        merged = self._rrf_merge(bm25_results, vector_results, alpha)

        logger.info(
            "Hybrid search completed",
            extra={
                "bm25_hits": len(bm25_results),
                "vector_hits": len(vector_results),
                "merged_hits": len(merged),
                "alpha": alpha,
                "namespace": namespace,
            },
        )

        return merged[:top_k]
