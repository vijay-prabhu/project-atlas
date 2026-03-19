"""CDC (Change Data Capture) event processing for DynamoDB streams and SQS.

Listens for INSERT/MODIFY/REMOVE events and keeps the vector index in sync
with the source of truth in DynamoDB. Tracks processed event IDs to avoid
double-processing.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.core.observability import get_logger
from src.pipeline.chunking import chunk_auto
from src.pipeline.embedding import EmbeddingService

logger = get_logger(__name__)


class EventType(str, Enum):
    INSERT = "INSERT"
    MODIFY = "MODIFY"
    REMOVE = "REMOVE"


@dataclass
class CDCEvent:
    """A single change data capture event."""

    event_type: EventType
    table_name: str
    record: dict = field(default_factory=dict)
    old_record: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = ""


class CDCHandler:
    """Handles CDC events by syncing document changes to the vector index.

    INSERT  → chunk the document, generate embeddings, upsert to index
    MODIFY  → re-embed changed fields, update the index
    REMOVE  → delete vectors from the index

    Idempotency: processed event IDs are tracked in a set so duplicate
    events from DynamoDB streams or SQS retries don't cause double work.
    The set is in-memory for now — swap in Redis or DynamoDB for production.
    """

    def __init__(
        self,
        vector_store: Any = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._processed_ids: set[str] = set()
        self._lock = threading.Lock()

    def _already_processed(self, event_id: str) -> bool:
        """Check if an event has already been handled. Thread-safe."""
        if not event_id:
            return False
        with self._lock:
            if event_id in self._processed_ids:
                return True
            self._processed_ids.add(event_id)
            return False

    def handle_event(self, event: CDCEvent) -> None:
        """Route a CDC event to the right handler based on event type."""
        if self._already_processed(event.event_id):
            logger.info(
                "Skipping duplicate event",
                extra={"event_id": event.event_id, "table": event.table_name},
            )
            return

        logger.info(
            "Handling CDC event",
            extra={
                "event_type": event.event_type.value,
                "table": event.table_name,
                "event_id": event.event_id,
            },
        )

        if event.event_type == EventType.INSERT:
            self._handle_insert(event)
        elif event.event_type == EventType.MODIFY:
            self._handle_modify(event)
        elif event.event_type == EventType.REMOVE:
            self._handle_remove(event)

    def _handle_insert(self, event: CDCEvent) -> None:
        """Process a new document: chunk it, embed it, index it."""
        record = event.record
        doc_id = record.get("id", "")
        tenant_id = record.get("tenant_id", "")
        doc_type = record.get("doc_type", "email")
        text = record.get("body", "") or record.get("content", "")

        if not text:
            logger.warning(
                "INSERT event has no text content, skipping",
                extra={"doc_id": doc_id},
            )
            return

        chunks = chunk_auto(text, doc_type)
        if not chunks:
            return

        embedding_svc = self._embedding_service or EmbeddingService(
            tenant_id=tenant_id,
        )
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_svc.embed_batch(texts)

        if self._vector_store is not None:
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": f"{doc_id}_{chunk.start_index}",
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "tenant_id": tenant_id,
                        "doc_type": doc_type,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.content,
                        **chunk.metadata,
                    },
                })
            self._vector_store.upsert(vectors=vectors, namespace=tenant_id)

        logger.info(
            "INSERT processed",
            extra={"doc_id": doc_id, "chunk_count": len(chunks)},
        )

    def _handle_modify(self, event: CDCEvent) -> None:
        """Re-embed changed fields and update the index.

        Compares old_record and new record to find which text fields changed.
        Only re-embeds if text content actually changed.
        """
        record = event.record
        old_record = event.old_record or {}
        doc_id = record.get("id", "")
        tenant_id = record.get("tenant_id", "")
        doc_type = record.get("doc_type", "email")

        # Check which text fields changed
        text_fields = ["body", "content", "subject", "notes"]
        new_text = ""
        changed = False

        for f in text_fields:
            new_val = record.get(f, "")
            old_val = old_record.get(f, "")
            if new_val != old_val and new_val:
                changed = True
                new_text += new_val + "\n"

        if not changed:
            logger.info(
                "MODIFY event has no text changes, skipping re-embedding",
                extra={"doc_id": doc_id},
            )
            return

        # Delete old vectors, then re-index like an insert
        if self._vector_store is not None:
            self._vector_store.delete(
                filter={"doc_id": doc_id},
                namespace=tenant_id,
            )

        chunks = chunk_auto(new_text.strip(), doc_type)
        if not chunks:
            return

        embedding_svc = self._embedding_service or EmbeddingService(
            tenant_id=tenant_id,
        )
        texts = [chunk.content for chunk in chunks]
        embeddings = embedding_svc.embed_batch(texts)

        if self._vector_store is not None:
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": f"{doc_id}_{chunk.start_index}",
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "tenant_id": tenant_id,
                        "doc_type": doc_type,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.content,
                        **chunk.metadata,
                    },
                })
            self._vector_store.upsert(vectors=vectors, namespace=tenant_id)

        logger.info(
            "MODIFY processed",
            extra={"doc_id": doc_id, "chunk_count": len(chunks)},
        )

    def _handle_remove(self, event: CDCEvent) -> None:
        """Delete all vectors for the removed document from the index."""
        record = event.record
        doc_id = record.get("id", "")
        tenant_id = record.get("tenant_id", "")

        if self._vector_store is not None:
            self._vector_store.delete(
                filter={"doc_id": doc_id},
                namespace=tenant_id,
            )

        logger.info("REMOVE processed", extra={"doc_id": doc_id})

    # ─── Event Parsers ───────────────────────────────────

    def handle_dynamodb_stream_event(
        self, raw_event: dict
    ) -> list[CDCEvent]:
        """Parse a DynamoDB Streams event into CDCEvent objects.

        DynamoDB stream records look like:
        {
            "Records": [
                {
                    "eventID": "...",
                    "eventName": "INSERT" | "MODIFY" | "REMOVE",
                    "eventSourceARN": "arn:aws:dynamodb:...:table/MyTable/stream/...",
                    "dynamodb": {
                        "NewImage": {...},
                        "OldImage": {...},
                    }
                }
            ]
        }
        """
        events: list[CDCEvent] = []
        records = raw_event.get("Records", [])

        for record in records:
            event_name = record.get("eventName", "")
            try:
                event_type = EventType(event_name)
            except ValueError:
                logger.warning(
                    "Unknown DynamoDB event type",
                    extra={"event_name": event_name},
                )
                continue

            # Extract table name from the ARN
            arn = record.get("eventSourceARN", "")
            table_name = _table_name_from_arn(arn)

            dynamodb_data = record.get("dynamodb", {})
            new_image = _deserialize_dynamodb_item(
                dynamodb_data.get("NewImage", {})
            )
            old_image = _deserialize_dynamodb_item(
                dynamodb_data.get("OldImage", {})
            )

            events.append(CDCEvent(
                event_type=event_type,
                table_name=table_name,
                record=new_image if new_image else old_image,
                old_record=old_image if event_type == EventType.MODIFY else None,
                event_id=record.get("eventID", ""),
            ))

        return events

    def handle_sqs_event(self, raw_event: dict) -> list[CDCEvent]:
        """Parse an SQS event into CDCEvent objects.

        Expects the SQS message body to contain a JSON object with:
        {
            "event_type": "INSERT" | "MODIFY" | "REMOVE",
            "table_name": "...",
            "record": {...},
            "old_record": {...},
            "event_id": "..."
        }
        """
        import json

        events: list[CDCEvent] = []
        records = raw_event.get("Records", [])

        for sqs_record in records:
            body_str = sqs_record.get("body", "{}")
            try:
                body = json.loads(body_str)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse SQS message body",
                    extra={"message_id": sqs_record.get("messageId", "")},
                )
                continue

            try:
                event_type = EventType(body.get("event_type", ""))
            except ValueError:
                logger.warning(
                    "Unknown event type in SQS message",
                    extra={"event_type": body.get("event_type")},
                )
                continue

            events.append(CDCEvent(
                event_type=event_type,
                table_name=body.get("table_name", ""),
                record=body.get("record", {}),
                old_record=body.get("old_record"),
                event_id=body.get("event_id", sqs_record.get("messageId", "")),
            ))

        return events


# ─── Helpers ─────────────────────────────────────────────


def _table_name_from_arn(arn: str) -> str:
    """Extract table name from a DynamoDB stream ARN.

    ARN format: arn:aws:dynamodb:region:account:table/TABLE_NAME/stream/...
    """
    if "/table/" in arn:
        parts = arn.split("/table/")[1]
        return parts.split("/")[0]
    return ""


def _deserialize_dynamodb_item(item: dict) -> dict:
    """Convert DynamoDB typed attribute map to plain dict.

    Handles common DynamoDB types: S (string), N (number), BOOL, NULL,
    L (list), M (map). Nested structures are handled recursively.
    """
    if not item:
        return {}

    result: dict = {}
    for key, type_value in item.items():
        result[key] = _deserialize_value(type_value)
    return result


def _deserialize_value(type_value: dict) -> Any:
    """Deserialize a single DynamoDB typed value."""
    if "S" in type_value:
        return type_value["S"]
    if "N" in type_value:
        num_str = type_value["N"]
        return int(num_str) if "." not in num_str else float(num_str)
    if "BOOL" in type_value:
        return type_value["BOOL"]
    if "NULL" in type_value:
        return None
    if "L" in type_value:
        return [_deserialize_value(v) for v in type_value["L"]]
    if "M" in type_value:
        return _deserialize_dynamodb_item(type_value["M"])
    if "SS" in type_value:
        return list(type_value["SS"])
    if "NS" in type_value:
        return [int(n) if "." not in n else float(n) for n in type_value["NS"]]
    # Unknown type — return the raw value
    return type_value
