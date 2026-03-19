"""Pinecone vector store client with namespace-per-tenant isolation.

Every operation requires a namespace parameter. This maps directly to a
Pinecone namespace, giving each tenant full data isolation without
needing separate indexes. One shared index, many namespaces.
"""

from typing import Optional

from src.core.config import get_settings
from src.core.exceptions import SearchError
from src.core.observability import get_logger

logger = get_logger(__name__)

UPSERT_BATCH_SIZE = 100


class VectorStore:
    """Thin wrapper around the Pinecone client.

    Handles connection setup, batching, and graceful degradation when
    Pinecone is unavailable (e.g. during local testing).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._index = None
        self._available = False

        if not settings.pinecone_api_key:
            logger.warning("Pinecone API key not set — vector store running in stub mode")
            return

        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=settings.pinecone_api_key)
            self._index = pc.Index(settings.pinecone_index_name)
            self._available = True
            logger.info(
                "Pinecone connected",
                extra={"index": settings.pinecone_index_name},
            )
        except Exception as exc:
            logger.warning(
                "Pinecone unavailable — falling back to stub mode",
                extra={"error": str(exc)},
            )

    @property
    def available(self) -> bool:
        return self._available

    # ─── Core Operations ─────────────────────────────────

    def upsert(
        self,
        vectors: list[dict],
        namespace: str,
    ) -> int:
        """Batch upsert vectors into a tenant namespace.

        Each vector dict should have: {id, values, metadata}.
        Upserts in batches of 100 to stay within Pinecone limits.

        Returns the total number of vectors upserted.
        """
        if not self._available:
            logger.info("Stub upsert", extra={"count": len(vectors), "namespace": namespace})
            return len(vectors)

        total_upserted = 0
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            try:
                self._index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
            except Exception as exc:
                raise SearchError(
                    f"Pinecone upsert failed at batch {i // UPSERT_BATCH_SIZE}: {exc}"
                ) from exc

        logger.info(
            "Vectors upserted",
            extra={"count": total_upserted, "namespace": namespace},
        )
        return total_upserted

    def query(
        self,
        query_vector: list[float],
        namespace: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Run a similarity search within a tenant namespace.

        Returns a list of {id, score, metadata} dicts sorted by score desc.
        """
        if not self._available:
            logger.info("Stub query", extra={"namespace": namespace, "top_k": top_k})
            return []

        try:
            kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
            }
            if filters:
                kwargs["filter"] = filters

            response = self._index.query(**kwargs)

            results = []
            for match in response.get("matches", []):
                results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                })
            return results

        except Exception as exc:
            raise SearchError(f"Pinecone query failed: {exc}") from exc

    def delete(self, ids: list[str], namespace: str) -> None:
        """Remove vectors by ID from a tenant namespace."""
        if not self._available:
            logger.info("Stub delete", extra={"count": len(ids), "namespace": namespace})
            return

        try:
            self._index.delete(ids=ids, namespace=namespace)
            logger.info(
                "Vectors deleted",
                extra={"count": len(ids), "namespace": namespace},
            )
        except Exception as exc:
            raise SearchError(f"Pinecone delete failed: {exc}") from exc

    def get_index_stats(self, namespace: str) -> dict:
        """Return vector count and other stats for a namespace.

        Returns: {vector_count: int, dimension: int}
        """
        if not self._available:
            return {"vector_count": 0, "dimension": 0}

        try:
            stats = self._index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            return {
                "vector_count": ns_stats.get("vector_count", 0),
                "dimension": stats.get("dimension", 0),
            }
        except Exception as exc:
            raise SearchError(f"Failed to get index stats: {exc}") from exc
