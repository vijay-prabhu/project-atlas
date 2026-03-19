"""Embedding service with batch support and rate limit handling.

Uses OpenAI's text-embedding-3-small model. Tracks token usage through the
shared TokenTracker so embedding costs show up in tenant usage reports.
"""

import math
import time
from typing import Optional

import openai

from src.core.config import get_settings
from src.core.observability import get_logger
from src.llm.token_tracker import TokenTracker

logger = get_logger(__name__)

# Max texts per single OpenAI embeddings API call
_MAX_BATCH_SIZE = 100

# Retry config for rate limits
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0. Returns 0.0 if either vector
    has zero magnitude.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


class EmbeddingService:
    """Generates text embeddings via OpenAI with batch support and retry logic."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        tenant_id: str = "",
    ) -> None:
        settings = get_settings()
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = model
        self._dimensions = dimensions
        self._tenant_id = tenant_id
        self._tracker = TokenTracker()

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        result = self._call_api([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Splits into batches of 100 (OpenAI's per-call limit) and
        concatenates the results. Order is preserved.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[i : i + _MAX_BATCH_SIZE]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API with retry on rate limit errors.

        Retries up to 3 times with exponential backoff for rate limit (429)
        and server (5xx) errors.
        """
        last_error: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )

                # Track token usage
                if response.usage:
                    self._tracker.track(
                        tenant_id=self._tenant_id,
                        model=self._model,
                        operation="embedding",
                        input_tokens=response.usage.total_tokens,
                        output_tokens=0,
                        latency_ms=0.0,
                    )

                # Sort by index to guarantee order matches input
                sorted_data = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in sorted_data]

            except openai.RateLimitError as exc:
                last_error = exc
                delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    "Rate limited, retrying",
                    extra={
                        "attempt": attempt + 1,
                        "delay_s": delay,
                        "batch_size": len(texts),
                    },
                )
                time.sleep(delay)

            except openai.APIStatusError as exc:
                # Retry on server errors (5xx), fail fast on client errors
                if exc.status_code >= 500:
                    last_error = exc
                    delay = _BASE_DELAY_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Server error, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "status": exc.status_code,
                            "delay_s": delay,
                        },
                    )
                    time.sleep(delay)
                else:
                    raise

        # All retries exhausted
        raise RuntimeError(
            f"Embedding API call failed after {_MAX_RETRIES} retries: {last_error}"
        )
