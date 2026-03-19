"""Document processing orchestrator with status transitions.

Takes an incoming email, runs it through chunking and embedding, then upserts
the vectors to the store. Tracks status transitions so callers can monitor
progress and handle failures.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.api.schemas import AECOEmail
from src.core.observability import get_logger
from src.pipeline.chunking import chunk_email
from src.pipeline.embedding import EmbeddingService

logger = get_logger(__name__)


class ProcessingStatus(str, Enum):
    """Status transitions: RECEIVED → CHUNKING → EMBEDDING → INDEXING → COMPLETED.

    Any step can fail and move to FAILED instead.
    """

    RECEIVED = "received"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Outcome of processing a single document."""

    email_id: str
    status: ProcessingStatus
    chunk_count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


class DocumentProcessor:
    """Orchestrates the chunking → embedding → indexing pipeline for emails.

    Takes a vector store that implements upsert(vectors, namespace) and
    an optional EmbeddingService (creates one if not provided).
    """

    def __init__(
        self,
        vector_store: Any = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._status = ProcessingStatus.RECEIVED

    def _set_status(self, status: ProcessingStatus, email_id: str) -> None:
        self._status = status
        logger.info(
            "Processing status changed",
            extra={"email_id": email_id, "status": status.value},
        )

    def process_email(
        self,
        email: AECOEmail,
        tenant_id: str,
    ) -> ProcessingResult:
        """Run the full processing pipeline on an email.

        Steps:
            1. Chunk the email body
            2. Generate embeddings for all chunks
            3. Upsert vectors to the store under the tenant namespace
            4. Return the result with chunk count and timing

        If any step fails, status moves to FAILED and the error is captured.
        """
        email_id = email.id or str(uuid.uuid4())[:12]
        start = time.perf_counter()

        try:
            # 1. Chunking
            self._set_status(ProcessingStatus.CHUNKING, email_id)
            chunks = chunk_email(email.body, subject=email.subject)

            if not chunks:
                return ProcessingResult(
                    email_id=email_id,
                    status=ProcessingStatus.COMPLETED,
                    chunk_count=0,
                    duration_ms=_elapsed_ms(start),
                )

            # 2. Embedding
            self._set_status(ProcessingStatus.EMBEDDING, email_id)
            embedding_svc = self._embedding_service or EmbeddingService(
                tenant_id=tenant_id,
            )
            texts = [chunk.content for chunk in chunks]
            embeddings = embedding_svc.embed_batch(texts)

            # 3. Indexing
            self._set_status(ProcessingStatus.INDEXING, email_id)
            if self._vector_store is not None:
                vectors = []
                for chunk, embedding in zip(chunks, embeddings):
                    vectors.append({
                        "id": f"{email_id}_{chunk.start_index}",
                        "values": embedding,
                        "metadata": {
                            "email_id": email_id,
                            "tenant_id": tenant_id,
                            "chunk_type": chunk.chunk_type,
                            "text": chunk.content,
                            "subject": email.subject,
                            "sender": email.sender,
                            **chunk.metadata,
                        },
                    })
                self._vector_store.upsert(
                    vectors=vectors,
                    namespace=tenant_id,
                )

            # 4. Done
            self._set_status(ProcessingStatus.COMPLETED, email_id)
            duration = _elapsed_ms(start)

            logger.info(
                "Email processed",
                extra={
                    "email_id": email_id,
                    "tenant_id": tenant_id,
                    "chunk_count": len(chunks),
                    "duration_ms": duration,
                },
            )

            return ProcessingResult(
                email_id=email_id,
                status=ProcessingStatus.COMPLETED,
                chunk_count=len(chunks),
                duration_ms=duration,
            )

        except Exception as exc:
            self._set_status(ProcessingStatus.FAILED, email_id)
            duration = _elapsed_ms(start)
            error_msg = f"{type(exc).__name__}: {exc}"

            logger.error(
                "Email processing failed",
                extra={
                    "email_id": email_id,
                    "tenant_id": tenant_id,
                    "error": error_msg,
                    "duration_ms": duration,
                },
            )

            return ProcessingResult(
                email_id=email_id,
                status=ProcessingStatus.FAILED,
                chunk_count=0,
                duration_ms=duration,
                error=error_msg,
            )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
