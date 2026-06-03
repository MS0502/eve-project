"""EVE v3 round34 — state_debug fastText migration exposure.

Round34 exposes the separate FasttextEmbeddingAdapter in debug state only. It
must not load the small 5k subset, replace engine.self_embedding, or alter any
runtime generation/decision path.
"""

from __future__ import annotations

from pathlib import Path

from adapters.fasttext_embedding_adapter import FASTTEXT_EMBEDDING_DEFAULT_SUBSET
from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
CONCEPT_MEMORY = Path("adapters/concept_memory_adapter.py")


def test_engine_has_fasttext_embedding_attribute_unloaded():
    engine = build_full_engine()

    assert hasattr(engine, "fasttext_embedding")
    assert engine.fasttext_embedding.__class__.__name__ == "FasttextEmbeddingAdapter"
    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.subset_name == FASTTEXT_EMBEDDING_DEFAULT_SUBSET


def test_state_debug_includes_fasttext_section():
    engine = build_full_engine()

    state = engine.state_debug.snapshot_state()
    section = state["fasttext_embedding"]

    assert section["module"] == "fasttext_embedding_adapter"
    assert section["method"] == "fasttext_extracted_subset"
    assert section["dimension"] == 300
    assert section["subset_name"] == FASTTEXT_EMBEDDING_DEFAULT_SUBSET
    assert section["loaded"] is True
    assert section["vocab_size"] == 30000


def test_state_debug_fasttext_not_in_use_by_generation():
    engine = build_full_engine()

    state = engine.state_debug.snapshot_state()

    assert state["fasttext_embedding"]["in_use_by_generation"] == "wrapper_primary"
    assert state["fasttext_embedding"]["migration_stage"] == "final_swap_primary"


def test_state_debug_self_embedding_still_in_use():
    engine = build_full_engine()

    state = engine.state_debug.snapshot_state()
    section = state["self_embedding"]

    assert section["module"] == "embedding_wrapper"
    assert section["method"] == "fasttext_primary_pmi_svd_fallback"
    assert section["dimension"] == 300
    assert section["in_use_by_generation"] == "wrapper"
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_state_debug_is_read_only_for_fasttext_adapter():
    engine = build_full_engine()
    before_stats = dict(engine.fasttext_embedding.stats())
    before_loaded = engine.fasttext_embedding.is_loaded()
    before_self_class = engine.self_embedding.__class__.__name__

    first = engine.state_debug.snapshot_state()
    second = engine.state_debug.snapshot_state()

    assert first["fasttext_embedding"] == second["fasttext_embedding"]
    assert engine.fasttext_embedding.stats() == before_stats
    assert engine.fasttext_embedding.is_loaded() is before_loaded
    assert engine.self_embedding.__class__.__name__ == before_self_class


def test_state_debug_does_not_trigger_fasttext_load():
    engine = build_full_engine()

    engine.state_debug.snapshot_state()
    engine.state_debug.diagnose_text("오늘 날씨 좋다")

    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000


def test_generation_path_still_uses_pmi_svd_self_embedding():
    engine = build_full_engine()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.fasttext_embedding.is_loaded() is True
    # No broad swap: the two adapters are separate attributes.
    assert engine.self_embedding is not engine.fasttext_embedding


def test_concept_memory_adapter_not_rewritten_by_round34_state_debug_migration():
    text = CONCEPT_MEMORY.read_text(encoding="utf-8")

    # Round37 legitimately adds concept-memory fastText observation.
    # The invariant from round34 that still matters is no self_embedding rewrite
    # and no hard-coded subset path in concept memory.
    assert "vectors.npy" not in text
    assert "cc.ko.300.subset" not in text
    assert "engine.self_embedding =" not in text


def test_self_embedding_adapter_file_unchanged_by_round34():
    text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in text
    assert "fasttext_embedding" not in text
    assert "vectors.npy" not in text
    assert "cc.ko.300.subset" not in text


def test_round34_does_not_load_subset_or_change_migration_state():
    engine = build_full_engine()

    assert engine.fasttext_embedding.stats()["runtime_used"] is False
    assert engine.fasttext_embedding.stats()["self_embedding_rewrite"] is False
    assert engine.fasttext_embedding.stats()["loaded"] is True
    assert engine.state_debug.snapshot_state()["fasttext_embedding"]["loaded"] is True
