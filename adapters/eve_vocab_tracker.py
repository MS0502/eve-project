"""EVE v3 round54 — deterministic EVE-specific vocabulary observation.

Round54 implements manual observation only. It does not auto-observe from the
runtime, does not generate vectors, and does not integrate with the wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class EveVocabTracker:
    """Track EVE-specific lexical observations deterministically.

    Round54 policy:
    - manual ``observe_word`` only
    - deterministic counts and sorted vocab
    - no vector generation
    - no wrapper integration
    - no automatic vocabulary promotion
    """

    implementation_phase = "round54_observation"
    context_cap = 1000

    def __init__(self, engine: Any | None = None, context_cap: int = 1000) -> None:
        self.engine = engine
        self._observed: dict[str, int] = {}
        self._contexts: list[dict[str, Any]] = []
        self._context_cap = max(0, int(context_cap))

    def observe_word(self, word: str, context: dict[str, Any] | None = None) -> None:
        """Observe one word with a deterministic count increment.

        Empty/whitespace-only words are ignored to avoid polluting the future
        EVE-specific vector store with non-lexical artifacts.
        """
        token = str(word or "").strip()
        if not token:
            return
        self._observed[token] = self._observed.get(token, 0) + 1
        if context is not None and self._context_cap > 0:
            self._contexts.append({
                "word": token,
                "context": deepcopy(context),
                "observation_index": self._observed[token],
            })
            if len(self._contexts) > self._context_cap:
                del self._contexts[: len(self._contexts) - self._context_cap]

    def _fasttext_has_word(self, word: str) -> bool:
        if self.engine is None:
            return False
        fasttext = getattr(self.engine, "fasttext_embedding", None)
        if fasttext is None:
            return False
        is_loaded = getattr(fasttext, "is_loaded", None)
        if callable(is_loaded) and not is_loaded():
            return False
        try:
            return fasttext.get_vector(word) is not None
        except Exception:
            return False

    def is_eve_specific(self, word: str) -> bool:
        """Return True when a manually observed word is absent from fastText.

        A word must first be observed. This prevents unobserved arbitrary OOV
        queries from being promoted as EVE-specific candidates.
        """
        token = str(word or "").strip()
        if not token or token not in self._observed:
            return False
        return not self._fasttext_has_word(token)

    def get_observation_count(self, word: str) -> int:
        """Return the deterministic observation count for ``word``."""
        return int(self._observed.get(str(word or "").strip(), 0))

    def tracked_vocab(self) -> list[str]:
        """Return observed vocabulary in deterministic sorted order."""
        return sorted(self._observed.keys())

    def eve_specific_vocab(self) -> list[str]:
        """Return observed fastText-OOV vocabulary in deterministic order."""
        return [word for word in self.tracked_vocab() if self.is_eve_specific(word)]

    def stats(self) -> dict[str, Any]:
        """Read-only observation stats."""
        eve_specific = self.eve_specific_vocab()
        return {
            "adapter": self.__class__.__name__,
            "total_observed": len(self._observed),
            "observed_count": len(self._observed),
            "tracked_vocab_count": len(self._observed),
            "eve_specific_count": len(eve_specific),
            "eve_specific_sample": eve_specific[:50],
            "context_log_size": len(self._contexts),
            "context_cap": self._context_cap,
            "implementation_phase": self.implementation_phase,
            "observation_implemented": True,
            # Round71 role boundary: the tracker only records lexical
            # observations. Continuous observation and vector commits are owned
            # by EveSelfLearningAdapter; vector storage is owned by
            # EveSpecificVectorStore. Keep these flags explicit so the tracker
            # is not mistaken for a promotion/generation component.
            "role": "lexical_observation_only",
            "manual_observation_supported": True,
            "continuous_observation_owner": (
                "EveSelfLearningAdapter"
                if getattr(self.engine, "eve_vocab_tracker", None) is self
                and getattr(self.engine, "eve_self_learning", None) is not None
                else None
            ),
            "vector_commit_owner": (
                "EveSelfLearningAdapter"
                if getattr(self.engine, "eve_vocab_tracker", None) is self
                and getattr(self.engine, "eve_self_learning", None) is not None
                else None
            ),
            "auto_observe_enabled": bool(
                getattr(self.engine, "eve_vocab_tracker", None) is self
                and getattr(self.engine, "eve_self_learning", None) is not None
            ),
            "auto_promotion_enabled": False,
            "vector_generation_enabled": False,
            "wrapper_integrated": False,
            "wrapper_observation_side_effect": False,
            "round": 71,
            "read_only": True,
            "observed_snapshot": deepcopy(self._observed),
            "context_snapshot": deepcopy(self._contexts),
        }
