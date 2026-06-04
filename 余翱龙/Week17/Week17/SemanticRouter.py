"""Thin wrapper around ``redisvl.extensions.router.SemanticRouter``.

Routes incoming queries to the best matching named ``Route`` based on
semantic similarity to that route's reference utterances.

The RedisVL :class:`Route` type is re-exported for caller convenience so users
of this module never need to import from ``redisvl`` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from redisvl.extensions.router import Route  # re-exported
from redisvl.extensions.router import SemanticRouter as _RedisVLSemanticRouter

import config


class SemanticRouter:
    """VLM-app semantic intent router."""

    def __init__(
        self,
        name: str = "vlm_router",
        routes: Optional[List[Route]] = None,
        redis_url: Optional[str] = None,
        vectorizer=None,
        overwrite: bool = False,
    ) -> None:
        self.name = name
        self.vectorizer = config.resolve_vectorizer(vectorizer)

        self._router = _RedisVLSemanticRouter(
            name=name,
            routes=list(routes) if routes else [],
            vectorizer=self.vectorizer,
            redis_url=config.resolve_redis_url(redis_url),
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Pass-through API
    # ------------------------------------------------------------------

    def __call__(self, query: str) -> Any:
        """Route ``query`` to the single best matching ``Route``."""
        return self._router(query)

    def route_many(
        self,
        query: str,
        max_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Any]:
        """Return up to ``max_k`` matching routes for ``query``."""
        kwargs: Dict[str, Any] = {"statement": query}
        if max_k is not None:
            kwargs["max_k"] = max_k
        if distance_threshold is not None:
            kwargs["distance_threshold"] = distance_threshold
        return self._router.route_many(**kwargs)

    def add_route(self, route: Route) -> None:
        """Register a new ``Route``."""
        # RedisVL exposes ``add_route`` accepting a list of Route objects.
        self._router.add_route([route])

    def remove_route(self, name: str) -> None:
        """Remove the route registered under ``name``."""
        self._router.remove_route(name)

    def clear(self) -> None:
        """Remove every route stored under this router's index."""
        self._router.clear()
