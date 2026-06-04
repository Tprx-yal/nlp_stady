# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python project that uses Redis to build a caching layer for a Vision Language Model (VLM) application. The system implements semantic caching patterns to reduce redundant VLM API calls and enable intelligent message history.

## Planned Components

The project consists of four core modules (described in `Task requirements.md`):

- `EmbeddingsCache.py` — Caches embedding vectors to avoid recomputing embeddings for repeated inputs.
- `SemanticCache.py` — Caches VLM responses keyed by semantic similarity rather than exact text match, so similar queries return cached results.
- `SemanticMessageHistory.py` — Stores conversation history with semantic retrieval, enabling context-aware responses across sessions.
- `SemanticRouter.py` — Routes incoming requests to appropriate handlers/models based on semantic intent matching.

## Current State

Only `Task requirements.md` is present. The four Python modules listed above have not yet been created. When implementing, follow the component split defined in the requirements doc — each module should be self-contained and focus on a single caching concern.

## Development Notes

- This is a Python + Redis project; ensure a Redis instance is available for development and testing.
- Semantic caching requires an embedding model — coordinate the embedding choice (model, dimension) across all four modules so cached vectors are interoperable.
- Similarity thresholds and TTLs are key tunable parameters for `SemanticCache.py` and `SemanticRouter.py`; keep them configurable rather than hardcoded.
