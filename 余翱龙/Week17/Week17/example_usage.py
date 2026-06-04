"""Smoke test for the VLM caching project.

Run with::

    python example_usage.py

Prerequisites:
    * Redis listening on ``REDIS_URL`` (default ``redis://localhost:6379``).
    * ``pip install -r requirements.txt`` completed.
    * First run downloads the sentence-transformers model (~440MB) to
      ``~/.cache/huggingface``; subsequent runs reuse it.

The script exercises each wrapper end-to-end and prints results so the user
can eyeball that every module can read from and write to Redis.
"""

from __future__ import annotations

import os

# Quiet the HF tokenizer warning we'd otherwise hit from multi-threaded calls.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from EmbeddingsCache import EmbeddingsCache
from SemanticCache import SemanticCache
from SemanticMessageHistory import SemanticMessageHistory
from SemanticRouter import Route, SemanticRouter


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def demo_embeddings_cache() -> None:
    _section("EmbeddingsCache")
    cache = EmbeddingsCache(overwrite=True)

    text = "hello world"
    embedding = cache.vectorizer.embed(text)
    key = cache.set(text, embedding)
    print(f"stored embedding under key={key} (dim={len(embedding)})")

    fetched = cache.get(text)
    print(f"fetched entry contains keys: {sorted(fetched.keys()) if fetched else None}")

    print(f"'{text}' in cache? {text in cache}")
    cache.delete(text)
    print(f"after delete -> '{text}' in cache? {text in cache}")


def demo_semantic_cache() -> None:
    _section("SemanticCache")
    cache = SemanticCache(overwrite=True)
    cache.clear()

    cache.store(
        prompt="What is the capital of France?",
        response="Paris",
        metadata={"country": "France"},
    )
    print("stored: 'What is the capital of France?' -> 'Paris'")

    hits = cache.check("France's capital?", top_k=1)
    print(f"check('France's capital?') -> {len(hits)} hit(s)")
    if hits:
        print(f"   best match: response={hits[0].get('response')!r}, "
              f"distance={hits[0].get('vector_distance')}")
        assert "Paris" in hits[0].get("response", ""), (
            "expected cached 'Paris' response for paraphrased query"
        )
        print("   assertion passed: semantic match returned 'Paris'")


def demo_message_history() -> None:
    _section("SemanticMessageHistory")
    history = SemanticMessageHistory(overwrite=True)
    history.clear()

    history.add_user_message("What is the largest ocean on Earth?")
    history.add_llm_message("The Pacific Ocean is the largest.")
    history.add_user_message("Which planet has the most moons?")
    history.add_llm_message("Saturn has the most confirmed moons.")
    print("added 2 user / 2 llm messages")

    relevant = history.get_relevant("Tell me about oceans", top_k=3)
    print(f"get_relevant('Tell me about oceans') -> {len(relevant)} message(s)")
    for msg in relevant:
        print(f"   {msg}")


def demo_semantic_router() -> None:
    _section("SemanticRouter")
    router = SemanticRouter(overwrite=True)
    router.clear()

    router.add_route(
        Route(
            name="greet",
            references=["hello", "hi", "hey there", "good morning"],
            distance_threshold=0.7,
        )
    )
    router.add_route(
        Route(
            name="farewell",
            references=["bye", "goodbye", "see you later", "talk soon"],
            distance_threshold=0.7,
        )
    )
    print("registered routes: greet, farewell")

    match = router("hey, what's up?")
    print(f"router('hey, what's up?') -> {match}")

    matches = router.route_many("see ya tomorrow", max_k=2)
    print(f"route_many('see ya tomorrow') -> {matches}")


def main() -> None:
    demo_embeddings_cache()
    demo_semantic_cache()
    demo_message_history()
    demo_semantic_router()
    _section("done")
    print("All four wrappers executed against Redis successfully.")


if __name__ == "__main__":
    main()
