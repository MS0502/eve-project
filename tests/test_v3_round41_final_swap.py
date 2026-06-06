from pathlib import Path

import pytest

from main import build_full_engine
from adapters.embedding_wrapper import EmbeddingWrapper
from adapters.external_seed_manifest import assess_main_py_post_swap_state
from adapters.self_embedding_adapter import SelfEmbeddingAdapter


class LoadedFasttextFixture:
    subset_name = "loaded.fasttext.fixture"
    dimension = 300

    def __init__(self):
        self._loaded = True
        self._vocab = ["known"]
        self._vector = [1.0] + [0.0] * 299

    def is_loaded(self):
        return self._loaded

    def get_dimension(self):
        return self.dimension

    def get_vector(self, word):
        return list(self._vector) if word == "known" else None

    def get_embedding(self, word):
        return self.get_vector(word)

    def stats(self):
        return {"loaded": self._loaded, "dimension": self.dimension, "vocab_size": len(self._vocab)}


def _attach_loaded_fasttext_fixture(engine):
    engine.fasttext_embedding = LoadedFasttextFixture()
    engine.self_embedding.primary = engine.fasttext_embedding
    return engine.fasttext_embedding


def test_round41_engine_self_embedding_is_wrapper():
    engine = build_full_engine()
    assert isinstance(engine.self_embedding, EmbeddingWrapper)
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert engine.self_embedding_backup.__class__.__name__ == "SelfEmbeddingAdapter"


def test_round41_fasttext_primary_is_no_load_until_operator_artifact_authorization():
    engine = build_full_engine()
    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.self_embedding.stats()["primary_loaded"] is False
    assert engine.medium30k_runtime_load_report["status"] == "default_no_load_explicit_authorization_required"


def test_round41_wrapper_primary_used_for_known_word_when_explicitly_loaded():
    engine = build_full_engine()
    loaded = _attach_loaded_fasttext_fixture(engine)
    word = loaded._vocab[0]
    vec = engine.self_embedding.get_vector(word)
    assert vec is not None
    assert len(vec) == 300
    assert engine.self_embedding.stats()["primary_count"] >= 1


def test_round41_wrapper_fallback_for_unknown_word():
    engine = build_full_engine()
    engine.self_embedding_backup.embeddings["fallback_only"] = [1.0, 0.0]
    vec = engine.self_embedding.get_embedding("fallback_only")
    assert vec == [1.0, 0.0]
    assert engine.self_embedding.stats()["fallback_count"] >= 1


def test_round41_wrapper_fallback_on_primary_exception():
    engine = build_full_engine()

    class BrokenPrimary:
        def is_loaded(self):
            return True
        def get_embedding(self, word):
            raise RuntimeError("boom")
        def get_vector(self, word):
            raise RuntimeError("boom")
        def get_dimension(self):
            return 300
        def stats(self):
            return {"loaded": True}

    engine.self_embedding.primary = BrokenPrimary()
    engine.self_embedding_backup.embeddings["safe"] = [0.5]
    assert engine.self_embedding.get_vector("safe") == [0.5]
    stats = engine.self_embedding.stats()
    assert stats["fallback_count"] >= 1
    assert stats["error_count"] >= 1


def test_round41_wrapper_api_compatibility_surface():
    engine = build_full_engine()
    wrapper = engine.self_embedding
    for name in [
        "observe", "bootstrap_from_engine", "ensure_ready", "train",
        "get_vector", "get_embedding", "similarity", "most_similar",
        "text_embedding", "text_similarity", "stats", "get_dimension",
    ]:
        assert callable(getattr(wrapper, name))


def test_round41_wrapper_dimension_reports_fasttext_primary():
    engine = build_full_engine()
    assert engine.self_embedding.get_dimension() == 300
    assert engine.self_embedding.dim == 300
    assert engine.self_embedding.stats()["fallback_dimension"] == 50


def test_round41_observe_keeps_pmi_svd_fallback_mutable():
    engine = build_full_engine()
    before = engine.self_embedding_backup.total_obs
    engine.self_embedding.observe("민석 이브 함께 공부")
    assert engine.self_embedding_backup.total_obs >= before


def test_round41_all_migrated_modules_report_wrapper():
    engine = build_full_engine()
    assert engine.attention.stats()["in_use_by_generation"] == "wrapper"
    assert engine.compositor.stats()["in_use_by_generation"] == "wrapper"
    assert engine.concept_memory.stats()["in_use_by_generation"] == "wrapper"
    assert engine.situation_responder.stats()["in_use_by_generation"] == "wrapper"
    assert engine.stats()["in_use_by_generation"] == "wrapper"


def test_round41_state_debug_reflects_wrapper_and_backup():
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    assert state["self_embedding"]["module"] == "embedding_wrapper"
    assert state["self_embedding"]["in_use_by_generation"] == "wrapper"
    assert state["self_embedding"]["primary_class"] == "FasttextEmbeddingAdapter"
    assert state["self_embedding"]["fallback_class"] == "SelfEmbeddingAdapter"
    assert state["main_engine"]["self_embedding_backup_available"] is True
    assert state["main_engine"]["in_use_by_generation"] == "wrapper"


def test_round41_rollback_possible():
    engine = build_full_engine()
    backup = engine.self_embedding_backup
    engine.self_embedding = engine.self_embedding.rollback()
    assert engine.self_embedding is backup
    assert isinstance(engine.self_embedding, SelfEmbeddingAdapter)


def test_round41_runtime_unload_falls_back_safely():
    engine = build_full_engine()
    engine.self_embedding_backup.embeddings["민석"] = [1.0]
    engine.fasttext_embedding._loaded = False
    result = engine.self_embedding.get_embedding("민석")
    assert result == [1.0]
    assert engine.self_embedding.stats()["fallback_count"] >= 1


def test_round41_post_swap_assessment_ready_without_default_vector_load():
    engine = build_full_engine()
    report = assess_main_py_post_swap_state(engine)
    assert report["swap_state"] == "active_no_load_default"
    assert report["migration_progress"] == "7/7"
    assert report["self_embedding_active"] == "EmbeddingWrapper"
    assert report["primary_loaded"] is False
    assert report["primary_load_required"] == "operator_authorized_artifact_load"
    assert report["concerns"] == []


def test_round41_self_embedding_adapter_file_preserved():
    assert Path("adapters/self_embedding_adapter.py").exists()


def test_round41_main_py_contains_wrapper_swap_and_backup():
    text = Path("main.py").read_text(encoding="utf-8")
    assert "engine.self_embedding_backup = engine.self_embedding" in text
    assert "EmbeddingWrapper" in text
    assert "apply_round194_guarded_medium30k_runtime_load" in text
    assert "operator_medium30k_load_authorized" in text
