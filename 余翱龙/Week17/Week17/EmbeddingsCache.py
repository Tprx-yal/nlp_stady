"""Thin wrapper around ``redisvl.extensions.cache.embeddings.EmbeddingsCache``.

Lets callers store and retrieve embedding vectors keyed by their source text
without having to remember the underlying ``(content, model_name, embedding)``
signature or the RedisVL import path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from redisvl.extensions.cache.embeddings import (
    EmbeddingsCache as _RedisVLEmbeddingsCache,
)

import config


class EmbeddingsCache:
    """VLM-app embeddings cache.

    Stores precomputed embedding vectors so repeated inputs do not have to be
    re-embedded. Uses the shared :func:`config.get_vectorizer` model name to
    namespace entries unless overridden.
    """

    def __init__(
        self,
        name: str = "vlm_embeddings_cache",
        redis_url: Optional[str] = None,
        vectorizer=None,
        ttl: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        self.name = name
        self.vectorizer = config.resolve_vectorizer(vectorizer)
        self.model_name = getattr(
            self.vectorizer, "model", config.DEFAULT_VECTORIZER_MODEL
        )

        kwargs: Dict[str, Any] = {
            "name": name,
            "redis_url": config.resolve_redis_url(redis_url),
        }
        if ttl is not None:
            kwargs["ttl"] = ttl
        if overwrite:
            kwargs["overwrite"] = True

        self._cache = _RedisVLEmbeddingsCache(**kwargs)

    # ------------------------------------------------------------------
    # Pass-through API (simplified: text -> embedding)
    # ------------------------------------------------------------------

    def set(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store ``embedding`` for ``text``. Returns the Redis key."""
        return self._cache.set(
            content=text,
            model_name=self.model_name,
            embedding=embedding,
            metadata=metadata,
            ttl=ttl,
        )

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Return the cached entry (with ``embedding`` key) for ``text``, or None."""
        return self._cache.get(content=text, model_name=self.model_name)

    def delete(self, text: str) -> None:
        """Remove the cached entry for ``text``."""
        self._cache.drop(content=text, model_name=self.model_name)

    def clear(self) -> None:
        """Remove every entry stored under this cache's index."""
        self._cache.clear()

    def __contains__(self, text: str) -> bool:
        return self._cache.exists(content=text, model_name=self.model_name)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def embed_and_set(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Embed ``text`` with the configured vectorizer, then cache it."""
        embedding = self.vectorizer.embed(text)
        return self.set(text, embedding, metadata=metadata, ttl=ttl)

    def get_many(self, texts: Iterable[str]) -> List[Optional[Dict[str, Any]]]:
        """Return cached entries for each text in ``texts`` (None if missing)."""
        return [self.get(t) for t in texts]
