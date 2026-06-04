"""Shared configuration for the VLM caching project.

Centralizes Redis connection settings, the default text vectorizer, and
threshold/TTL defaults so each wrapper module can be instantiated with no
arguments. Overrides may still be passed explicitly to each wrapper.
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

# Override by setting the REDIS_URL environment variable, e.g.
#   set REDIS_URL=redis://user:pass@my-host:6379/0
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Default thresholds / TTLs (tunable)
# ---------------------------------------------------------------------------

# Matches the RedisVL SemanticCache default. Lower => stricter match.
DEFAULT_DISTANCE_THRESHOLD: float = 0.1

# 1 hour, applied to cache entries unless overridden per-entry.
DEFAULT_TTL: int = 3600

# Matches the RedisVL SemanticMessageHistory default.
DEFAULT_HISTORY_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# Default vectorizer (lazy singleton)
# ---------------------------------------------------------------------------

# Default HuggingFace sentence-transformers model. Text-only (chosen per the
# project plan). The first import of the model downloads weights to
# ~/.cache/huggingface and is therefore deferred until first use.
DEFAULT_VECTORIZER_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

_vectorizer_singleton = None  # type: ignore[var-annotated]


def get_vectorizer():
    """Return the shared HFTextVectorizer instance, creating it on first call.

    Lazy initialization avoids loading the sentence-transformers model at
    import time (which is slow and may download weights).
    """
    global _vectorizer_singleton
    if _vectorizer_singleton is None:
        # Local import keeps `import config` cheap.
        from redisvl.utils.vectorize import HFTextVectorizer

        _vectorizer_singleton = HFTextVectorizer(model=DEFAULT_VECTORIZER_MODEL)
    return _vectorizer_singleton


def resolve_redis_url(redis_url: Optional[str]) -> str:
    """Return the caller's `redis_url` if given, otherwise the shared default."""
    return redis_url if redis_url is not None else REDIS_URL


def resolve_vectorizer(vectorizer):
    """Return the caller's `vectorizer` if given, otherwise the shared default."""
    return vectorizer if vectorizer is not None else get_vectorizer()
