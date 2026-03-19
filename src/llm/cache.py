"""Semantic prompt cache to avoid duplicate LLM calls.

Uses exact hash matching with in-memory storage and TTL-based expiry.
Hit/miss rates are tracked for monitoring.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class _CacheEntry:
    """A cached LLM response with expiration metadata."""

    response: str
    expires_at: float
    created_at: float = field(default_factory=time.time)


class PromptCache:
    """Hash-based prompt cache with TTL expiry and hit-rate tracking.

    Usage::

        cache = PromptCache()
        prompt_hash = PromptCache.hash_prompt(prompt, system_prompt, model)

        cached = cache.get(prompt_hash)
        if cached is not None:
            return cached

        response = llm_client.generate(prompt, ...)
        cache.set(prompt_hash, response.content)
    """

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prompt(
        prompt: str,
        system_prompt: str = "",
        model: str = "",
    ) -> str:
        """Create a deterministic hash for a prompt + context combination."""
        raw = f"{model}::{system_prompt}::{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt_hash: str) -> Optional[str]:
        """Look up a cached response by hash. Returns None on miss or expiry."""
        with self._lock:
            entry = self._store.get(prompt_hash)

            if entry is None:
                self._misses += 1
                return None

            if time.time() > entry.expires_at:
                del self._store[prompt_hash]
                self._misses += 1
                logger.info(
                    "Cache entry expired",
                    extra={"prompt_hash": prompt_hash[:12]},
                )
                return None

            self._hits += 1
            logger.info(
                "Cache hit",
                extra={"prompt_hash": prompt_hash[:12]},
            )
            return entry.response

    def set(
        self,
        prompt_hash: str,
        response: str,
        ttl_seconds: int = 3600,
    ) -> None:
        """Store a response in the cache with a TTL."""
        now = time.time()
        entry = _CacheEntry(
            response=response,
            expires_at=now + ttl_seconds,
            created_at=now,
        )
        with self._lock:
            self._store[prompt_hash] = entry

        logger.info(
            "Cache set",
            extra={
                "prompt_hash": prompt_hash[:12],
                "ttl_seconds": ttl_seconds,
            },
        )

    def stats(self) -> dict:
        """Return cache hit/miss statistics for monitoring."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_lookups": total,
                "hit_rate_pct": round(hit_rate, 2),
                "cached_entries": len(self._store),
            }

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count of entries evicted."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._store.items() if now > v.expires_at
            ]
            for key in expired_keys:
                del self._store[key]
                evicted += 1

        if evicted:
            logger.info("Cache eviction", extra={"evicted": evicted})
        return evicted

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
