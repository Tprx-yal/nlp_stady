"""Thin wrapper around ``redisvl.extensions.cache.llm.SemanticCache``.

Caches VLM responses keyed by semantic similarity. A paraphrased prompt that
falls within ``distance_threshold`` of a cached prompt will return the cached
response, eliminating a redundant model call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from redisvl.extensions.cache.llm import SemanticCache as _RedisVLSemanticCache

import config


class SemanticCache:
    """VLM-app semantic cache for prompt / response pairs."""

    def __init__(
        self,
        name: str = "vlm_llm_cache",
        distance_threshold: float = config.DEFAULT_DISTANCE_THRESHOLD,
        ttl: Optional[int] = config.DEFAULT_TTL,
        redis_url: Optional[str] = None,
        vectorizer=None,
        overwrite: bool = False,
    ) -> None:
        self.name = name
        self.vectorizer = config.resolve_vectorizer(vectorizer)

        kwargs: Dict[str, Any] = {
            "name": name,
            "distance_threshold": distance_threshold,
            "vectorizer": self.vectorizer,
            "redis_url": config.resolve_redis_url(redis_url),
            "overwrite": overwrite,
        }
        if ttl is not None:
            kwargs["ttl"] = ttl

        self._cache = _RedisVLSemanticCache(**kwargs)

    # ------------------------------------------------------------------
    # Pass-through API
    # ------------------------------------------------------------------

    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Cache the (prompt, response) pair. Returns the Redis key."""
        return self._cache.store(
            prompt=prompt,
            response=response,
            metadata=metadata,
            filters=filters,
            ttl=ttl,
        )

    def check(
        self,
        prompt: str,
        top_k: int = 1,
        distance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return cached matches for ``prompt`` (up to ``top_k`` results).

        Empty list when no entry is within the distance threshold.
        """
        kwargs: Dict[str, Any] = {"prompt": prompt, "num_results": top_k}
        if distance_threshold is not None:
            kwargs["distance_threshold"] = distance_threshold
        return self._cache.check(**kwargs)

    def set_threshold(self, value: float) -> None:
        """Update the cache's distance threshold."""
        self._cache.set_threshold(value)

    def set_ttl(self, value: Optional[int]) -> None:
        """Update the default TTL for new entries."""
        self._cache.set_ttl(value)

    def clear(self) -> None:
        """Remove every entry stored under this cache's index."""
        self._cache.clear()
