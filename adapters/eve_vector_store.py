"""EVE v3 round55 — deterministic EVE-specific vector store.

Round55 implements vector creation/storage only. It does not auto-observe
runtime text, does not integrate with ``EmbeddingWrapper``, and does not change
fastText/PMI+SVD routing.

Vector rule:
- source space: loaded fastText medium 30k, 300d float32
- input: EVE-specific word + deterministic context word sequence
- use only context words that exist in fastText primary
- vector = arithmetic mean of known context vectors
- repeated updates for the same word average old and new vectors

This keeps the self-learning path deterministic and aligned with the v3
External Seed Policy: seed = initial lexical map, learned vectors = EVE-specific
drift from deterministic observation/update rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np


class EveSpecificVectorStore:
    """Store deterministic 300d vectors for EVE-specific vocabulary."""

    implementation_phase = "round55_vector_update"

    def __init__(self, engine: Any | None = None, dimension: int = 300) -> None:
        if int(dimension) <= 0:
            raise ValueError("dimension_must_be_positive")
        self.engine = engine
        self.dimension = int(dimension)
        self._vectors: dict[str, np.ndarray] = {}
        self._update_counts: dict[str, int] = {}
        self._last_context: dict[str, list[str]] = {}

    def _normalize_word(self, word: str) -> str:
        return str(word or "").strip()

    def _resolve_engine(self, engine: Any | None = None) -> Any | None:
        return engine if engine is not None else self.engine

    def _fasttext(self, engine: Any | None = None) -> Any | None:
        resolved = self._resolve_engine(engine)
        if resolved is None:
            return None
        fasttext = getattr(resolved, "fasttext_embedding", None)
        if fasttext is None:
            return None
        is_loaded = getattr(fasttext, "is_loaded", None)
        if callable(is_loaded) and not is_loaded():
            return None
        return fasttext

    def _context_vectors(
        self,
        context_words: Iterable[str],
        *,
        engine: Any | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        fasttext = self._fasttext(engine)
        if fasttext is None:
            return [], []
        vectors: list[np.ndarray] = []
        used_words: list[str] = []
        for raw in context_words or []:
            token = self._normalize_word(raw)
            if not token:
                continue
            try:
                vec = fasttext.get_vector(token)
            except Exception:
                vec = None
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape != (self.dimension,):
                continue
            vectors.append(arr)
            used_words.append(token)
        return vectors, used_words

    def _mean_context_vector(
        self,
        context_words: Iterable[str],
        *,
        engine: Any | None = None,
    ) -> tuple[np.ndarray | None, list[str]]:
        vectors, used_words = self._context_vectors(context_words, engine=engine)
        if not vectors:
            return None, []
        return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32), used_words

    def add_vector(self, word: str, vector: np.ndarray) -> None:
        """Store a provided vector after deterministic validation."""
        token = self._normalize_word(word)
        if not token:
            raise ValueError("word_must_be_non_empty")
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimension,):
            raise ValueError("vector_dimension_mismatch")
        self._vectors[token] = arr.astype(np.float32, copy=True)
        self._update_counts[token] = self._update_counts.get(token, 0) + 1
        self._last_context[token] = []

    def add_or_update_vector(
        self,
        word: str,
        context_words: Iterable[str],
        engine: Any | None = None,
    ) -> bool:
        """Create/update ``word`` from known fastText context words.

        Unknown context words are ignored. If none remain, no mutation occurs.
        Repeated updates average the previous stored vector with the newly
        computed context vector.
        """
        token = self._normalize_word(word)
        if not token:
            return False
        new_vec, used_words = self._mean_context_vector(context_words, engine=engine)
        if new_vec is None:
            return False
        old_vec = self._vectors.get(token)
        if old_vec is None:
            merged = new_vec
        else:
            merged = ((old_vec.astype(np.float32) + new_vec) / 2.0).astype(np.float32)
        self._vectors[token] = merged.astype(np.float32, copy=True)
        self._update_counts[token] = self._update_counts.get(token, 0) + 1
        self._last_context[token] = list(used_words)
        return True

    def get_vector(self, word: str) -> np.ndarray | None:
        """Return a copy of a stored EVE-specific vector, or ``None``."""
        token = self._normalize_word(word)
        vec = self._vectors.get(token)
        if vec is None:
            return None
        return vec.astype(np.float32, copy=True)

    def is_eve_specific(self, word: str) -> bool:
        """Return whether ``word`` has a stored EVE-specific vector."""
        return self._normalize_word(word) in self._vectors

    def stored_vocab(self) -> list[str]:
        """Return stored vocabulary in deterministic sorted order."""
        return sorted(self._vectors.keys())

    def update_count(self, word: str) -> int:
        """Return deterministic update count for ``word``."""
        return int(self._update_counts.get(self._normalize_word(word), 0))

    def stats(self) -> dict[str, Any]:
        """Read-only vector store stats."""
        stored = self.stored_vocab()
        return {
            "adapter": self.__class__.__name__,
            "stored_count": len(stored),
            "stored_vocab_count": len(stored),
            "stored_vocab": stored,
            "stored_sample": stored[:50],
            "dimension": self.dimension,
            "implementation_phase": self.implementation_phase,
            "vector_update_implemented": True,
            "wrapper_integrated": bool(
                getattr(self.engine, "eve_specific_vector_store", None) is self
                and getattr(self.engine, "self_embedding", None).__class__.__name__ == "EmbeddingWrapper"
            ),
            "auto_observe_enabled": bool(
                getattr(self.engine, "eve_specific_vector_store", None) is self
                and getattr(self.engine, "eve_self_learning", None) is not None
            ),
            "auto_promotion_enabled": False,
            "round": 58,
            "read_only": True,
            "update_counts": deepcopy(self._update_counts),
            "last_context": deepcopy(self._last_context),
            "stored_snapshot": deepcopy({key: value.tolist() for key, value in self._vectors.items()}),
        }
