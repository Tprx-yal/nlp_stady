"""Thin wrapper around ``redisvl.extensions.message_history.SemanticMessageHistory``.

Stores conversation messages and retrieves semantically relevant ones to feed
back to the VLM as context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from redisvl.extensions.message_history import (
    SemanticMessageHistory as _RedisVLSemanticMessageHistory,
)

import config


class SemanticMessageHistory:
    """VLM-app message history with semantic retrieval."""

    def __init__(
        self,
        name: str = "vlm_message_history",
        session_tag: Optional[str] = None,
        distance_threshold: float = config.DEFAULT_HISTORY_THRESHOLD,
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
        if session_tag is not None:
            kwargs["session_tag"] = session_tag

        self._history = _RedisVLSemanticMessageHistory(**kwargs)

    # ------------------------------------------------------------------
    # Pass-through API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Add a single message with explicit role."""
        self._history.add_message({"role": role, "content": content})

    def add_user_message(self, prompt: str) -> None:
        """Convenience: add a ``user`` message."""
        self.add_message("user", prompt)

    def add_llm_message(self, response: str) -> None:
        """Convenience: add an ``llm`` message."""
        self.add_message("llm", response)

    def add_system_message(self, content: str) -> None:
        """Convenience: add a ``system`` message."""
        self.add_message("system", content)

    def add_messages(self, messages: List[Dict[str, str]]) -> None:
        """Add a batch of ``{"role": ..., "content": ...}`` messages."""
        self._history.add_messages(messages)

    def get_relevant(
        self,
        prompt: str,
        top_k: int = 5,
        fall_back: bool = True,
        as_text: bool = False,
        distance_threshold: Optional[float] = None,
        role: Optional[Union[str, List[str]]] = None,
    ) -> List[Any]:
        """Return previously stored messages semantically related to ``prompt``."""
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "top_k": top_k,
            "fall_back": fall_back,
            "as_text": as_text,
        }
        if distance_threshold is not None:
            kwargs["distance_threshold"] = distance_threshold
        if role is not None:
            kwargs["role"] = role
        return self._history.get_relevant(**kwargs)

    def get_recent(self, top_k: int = 5, as_text: bool = False) -> List[Any]:
        """Return the most recent messages in the active session."""
        return self._history.get_recent(top_k=top_k, as_text=as_text)

    def clear(self) -> None:
        """Remove every message stored under this history's index."""
        self._history.clear()
