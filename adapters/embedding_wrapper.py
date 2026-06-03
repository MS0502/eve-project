"""EmbeddingWrapper — v3 round41 final self_embedding swap.

Round72 adds explicit lookup-route policy snapshots so the local PMI+SVD
precheck in ``get_embedding`` is distinguishable from the final legacy fallback
API boundary. This is observability only; routing priority is unchanged.

FastText is the primary lexical space, while the original PMI+SVD
SelfEmbeddingAdapter remains as a deterministic fallback/rollback boundary.
The wrapper intentionally preserves the public self_embedding API used by the
existing six migrated modules.
"""

from __future__ import annotations

from typing import Any, Optional


class EmbeddingWrapper:
    """Primary embedding adapter with fail-open fallback.

    ``primary`` is expected to be a loaded FasttextEmbeddingAdapter. ``fallback``
    is the previous SelfEmbeddingAdapter. The wrapper does not delete or mutate
    either adapter's internal data model; it only routes read calls to primary
    first and falls back on OOV/errors.
    """

    def __init__(self, primary: Any, fallback: Any, eve_specific: Optional[Any] = None):
        self.primary = primary
        self.fallback = fallback
        # Optional deterministic EVE-specific vector store. When provided, this
        # store is queried after the primary fastText adapter and before the
        # PMI+SVD fallback. It must expose a ``get_vector(word)`` method and
        # should be read-only in this wrapper. See EveSpecificVectorStore for
        # the reference implementation.
        self.eve_specific = eve_specific
        self._fallback_count = 0
        self._primary_count = 0
        self._eve_specific_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._call_count = 0
        self._oov_log: list[dict[str, Any]] = []
        self._oov_log_cap = 1000
        # Preserve the historical method name for compatibility. Although this
        # wrapper introduces an intermediate eve-specific layer, the method
        # remains "fasttext_primary_pmi_svd_fallback" for backward tests.
        self.method = "fasttext_primary_pmi_svd_fallback"
        self.dim = self.get_dimension()

    def _primary_loaded(self) -> bool:
        try:
            is_loaded = getattr(self.primary, "is_loaded", None)
            return bool(is_loaded()) if callable(is_loaded) else True
        except Exception as exc:
            self._record_error(exc)
            return False

    def _record_error(self, exc: Exception) -> None:
        self._error_count += 1
        self._last_error = exc.__class__.__name__

    def _record_oov(self, token: str, operation: str) -> None:
        self._oov_log.append({"word": str(token), "operation": str(operation)})
        if len(self._oov_log) > self._oov_log_cap:
            del self._oov_log[: len(self._oov_log) - self._oov_log_cap]

    def _primary_loaded_read_only(self) -> bool:
        try:
            is_loaded = getattr(self.primary, "is_loaded", None)
            return bool(is_loaded()) if callable(is_loaded) else True
        except Exception:
            return False

    def _fallback_get(self, method_name: str, *args: Any, **kwargs: Any):
        method = getattr(self.fallback, method_name, None)
        if not callable(method):
            return None
        self._fallback_count += 1
        return method(*args, **kwargs)

    def _primary_get(self, method_name: str, *args: Any, **kwargs: Any):
        if not self._primary_loaded():
            return None
        method = getattr(self.primary, method_name, None)
        if not callable(method):
            return None
        try:
            result = method(*args, **kwargs)
            if result is not None:
                self._primary_count += 1
            return result
        except Exception as exc:
            self._record_error(exc)
            return None


    def _eve_specific_get(self, word: str) -> Any:
        """Return a vector from the eve-specific store, incrementing hit count.

        This helper checks the optional eve-specific vector store for a stored
        embedding for ``word``. If a vector is found, the internal
        ``_eve_specific_count`` is incremented. Exceptions during lookup are
        silently ignored and treated as missing.
        """
        store = getattr(self, "eve_specific", None)
        if store is None:
            return None
        try:
            result = store.get_vector(word)
        except Exception:
            result = None
        if result is not None:
            self._eve_specific_count += 1
        return result

    def _fallback_local_embedding(self, word: str) -> tuple[bool, Any]:
        """Return ``(hit, vector)`` from the fallback embedding map only.

        Round72 makes the ``get_embedding`` route explicit: learned PMI+SVD
        local vectors are preserved by checking the fallback ``embeddings`` map
        before the 300d fastText/EveSpecific path. This helper is a direct map
        lookup, not a call back into ``fallback.get_embedding``.
        """
        fallback_embeddings = getattr(self.fallback, "embeddings", {}) or {}
        if word in fallback_embeddings:
            return True, fallback_embeddings.get(word)
        return False, None

    def lookup_route_policy_snapshot(self) -> dict[str, Any]:
        """Return a read-only description of wrapper lookup priority.

        ``get_embedding`` intentionally differs from ``get_vector``: it first
        preserves already-learned local PMI+SVD vectors so legacy composition
        paths do not mix low-rank local vectors with 300d seed vectors. If that
        direct map check misses, the final fallback API is called at most once
        as the legacy safety boundary.
        """
        return {
            "policy_version": "v3_round72_embedding_lookup_route_snapshot",
            "adapter": "EmbeddingWrapper",
            "get_vector_route": [
                "fastText.get_vector",
                "EveSpecificVectorStore.get_vector",
                "PMI+SVD.get_embedding",
            ],
            "get_embedding_route": [
                "PMI+SVD.local_embeddings_map_precheck",
                "fastText.get_embedding",
                "EveSpecificVectorStore.get_vector",
                "PMI+SVD.get_embedding_final_fallback_once",
            ],
            "pmi_svd_local_precheck_enabled": True,
            "pmi_svd_local_precheck_calls_fallback_api": False,
            "fallback_api_final_boundary": True,
            "fallback_api_called_at_most_once_per_get_embedding_call": True,
            "duplicate_fallback_api_call_guarded": True,
            "routing_changed_in_round72": False,
            "read_only": True,
        }

    @property
    def embeddings(self):
        return getattr(self.fallback, "embeddings", {})

    @property
    def word_counts(self):
        return getattr(self.fallback, "word_counts", {})

    @property
    def cooc_counts(self):
        return getattr(self.fallback, "cooc_counts", {})

    # ===== Learning API: keep PMI+SVD fallback as the mutable learner =====

    def observe(self, text: str):
        method = getattr(self.fallback, "observe", None)
        if callable(method):
            return method(text)
        return None

    def bootstrap_from_engine(self):
        method = getattr(self.fallback, "bootstrap_from_engine", None)
        if callable(method):
            return method()
        return False

    def ensure_ready(self) -> bool:
        method = getattr(self.fallback, "ensure_ready", None)
        if callable(method):
            try:
                return bool(method())
            except Exception as exc:
                self._record_error(exc)
                return False
        return False

    def train(self):
        method = getattr(self.fallback, "train", None)
        if callable(method):
            return method()
        return False

    # ===== Embedding API: fastText primary, PMI+SVD fallback =====

    def get_vector(self, word: str):
        self._call_count += 1
        # Try primary fastText vector
        result = self._primary_get("get_vector", word)
        if result is not None:
            return result
        # Then try eve-specific vector store
        eve_vec = self._eve_specific_get(word)
        if eve_vec is not None:
            return eve_vec
        # Finally fallback to PMI+SVD embedding and record OOV
        self._record_oov(str(word), "get_vector")
        return self._fallback_get("get_embedding", word)

    def get_embedding(self, word: str):
        self._call_count += 1
        # Preserve coherent local PMI+SVD spaces when a concept has been learned
        # into the fallback. This prevents mixed 300d/low-rank vectors in legacy
        # composition paths. Otherwise use fastText as the primary source.
        local_hit, local_vec = self._fallback_local_embedding(word)
        if local_hit:
            self._fallback_count += 1
            return local_vec
        result = self._primary_get("get_embedding", word)
        if result is not None:
            return result
        # Try eve-specific store
        eve_vec = self._eve_specific_get(word)
        if eve_vec is not None:
            return eve_vec
        # Not found: record OOV and use fallback
        self._record_oov(str(word), "get_embedding")
        return self._fallback_get("get_embedding", word)

    def similarity(self, word_a: str, word_b: str):
        result = self._primary_get("similarity", word_a, word_b)
        # FasttextEmbeddingAdapter returns 0.0 for unknowns; treat that as a
        # valid deterministic primary result only when both vectors exist.
        try:
            if result not in (None, 0.0):
                return result
            if self._primary_loaded():
                va = getattr(self.primary, "get_vector", lambda _w: None)(word_a)
                vb = getattr(self.primary, "get_vector", lambda _w: None)(word_b)
                if va is not None and vb is not None:
                    return result
        except Exception as exc:
            self._record_error(exc)
        return self._fallback_get("similarity", word_a, word_b)

    def most_similar(self, word: str, top_n: int = 5, min_sim: float = 0.0) -> list:
        result = self._primary_get("most_similar", word, top_n, min_sim)
        if result:
            return result
        fallback_result = self._fallback_get("most_similar", word, top_n, min_sim)
        return fallback_result or []

    def text_embedding(self, text: str):
        result = self._primary_get("text_embedding", text)
        if result is not None:
            return result
        return self._fallback_get("text_embedding", text)

    def text_similarity(self, text_a: str, text_b: str):
        result = self._primary_get("text_similarity", text_a, text_b)
        if result is not None:
            return result
        return self._fallback_get("text_similarity", text_a, text_b)

    def get_dimension(self) -> int:
        method = getattr(self.primary, "get_dimension", None)
        if callable(method):
            try:
                return int(method())
            except Exception as exc:
                self._record_error(exc)
        return int(getattr(self.fallback, "dim", 50) or 50)

    def rollback(self):
        """Return the original fallback adapter for explicit rollback wiring."""
        return self.fallback

    def telemetry(self) -> dict[str, Any]:
        """Return read-only wrapper telemetry for round42 drift baselining."""
        total = int(self._call_count)
        primary_hits = int(self._primary_count)
        fallback_uses = int(self._fallback_count)
        eve_specific_hits = int(self._eve_specific_count)
        errors = int(self._error_count)
        return {
            "adapter": "EmbeddingWrapper",
            "telemetry_version": "v3_round42_wrapper_telemetry",
            "total_calls": total,
            "primary_hits": primary_hits,
            "fallback_uses": fallback_uses,
            "eve_specific_hits": eve_specific_hits,
            "errors": errors,
            "primary_hit_rate": (primary_hits / total) if total else 0.0,
            "fallback_rate": (fallback_uses / total) if total else 0.0,
            "eve_specific_rate": (eve_specific_hits / total) if total else 0.0,
            "error_rate": (errors / total) if total else 0.0,
            "oov_log_size": len(self._oov_log),
            "oov_log_cap": int(self._oov_log_cap),
            "recent_oov_sample": [dict(row) for row in self._oov_log[-10:]],
            "primary_loaded": self._primary_loaded_read_only(),
            "fallback_available": self.fallback is not None,
            "eve_specific_available": self.eve_specific is not None,
            "lookup_route_policy_version": "v3_round72_embedding_lookup_route_snapshot",
            "read_only": True,
        }

    def stats(self) -> dict[str, Any]:
        primary_stats = {}
        fallback_stats = {}
        try:
            if hasattr(self.primary, "stats"):
                primary_stats = dict(self.primary.stats())
        except Exception as exc:
            self._record_error(exc)
        try:
            if hasattr(self.fallback, "stats"):
                fallback_stats = dict(self.fallback.stats())
        except Exception as exc:
            self._record_error(exc)
        data = {
            "adapter": "EmbeddingWrapper",
            "method": self.method,
            "primary_class": self.primary.__class__.__name__,
            "fallback_class": self.fallback.__class__.__name__,
            "eve_specific_class": (self.eve_specific.__class__.__name__ if self.eve_specific is not None else None),
            "primary_loaded": self._primary_loaded(),
            "dimension": self.get_dimension(),
            "fallback_dimension": int(getattr(self.fallback, "dim", 50) or 50),
            "eve_specific_dimension": (int(getattr(self.eve_specific, "dimension", 0) or 0) if self.eve_specific is not None else 0),
            "fallback_count": self._fallback_count,
            "primary_count": self._primary_count,
            "eve_specific_count": self._eve_specific_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "rollback_available": self.fallback is not None,
            "telemetry": self.telemetry(),
            "lookup_route_policy": self.lookup_route_policy_snapshot(),
            "primary_stats": primary_stats,
            "fallback_stats": fallback_stats,
        }
        # Preserve legacy SelfEmbeddingAdapter stats keys for lazy bootstrap and
        # downstream diagnostics while still reporting wrapper identity.
        for key, value in fallback_stats.items():
            data.setdefault(key, value)
        return data
