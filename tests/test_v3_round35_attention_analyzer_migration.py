"""EVE v3 round35 — attention_analyzer fastText parallel observation.

Round35 lets attention_analyzer observe FasttextEmbeddingAdapter results in
parallel when the small 5k subset has been explicitly loaded. The active
attention decision path remains the existing PMI+SVD self_embedding path.
"""

from __future__ import annotations

from pathlib import Path

from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
CONCEPT_MEMORY = Path("adapters/concept_memory_adapter.py")


def _sample_attention(engine):
    return engine.attention.analyze(
        "오늘 친구 좋아",
        ["오늘", "친구", "좋아"],
        emotions={"joy": 0.8},
    )


def test_attention_default_uses_self_embedding_and_no_fasttext_trace():
    engine = build_full_engine()

    result = _sample_attention(engine)

    assert result.top_entities
    assert engine.fasttext_embedding.is_loaded() is True
    assert len(engine.attention.fasttext_trace) >= 1
    assert engine.attention.stats()["in_use_by_generation"] == "wrapper"
    assert engine.attention.stats()["fasttext_parallel_observation"] == "fasttext"


def test_attention_does_not_auto_load_fasttext():
    engine = build_full_engine()

    for _ in range(3):
        _sample_attention(engine)

    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000
    assert len(engine.attention.fasttext_trace) >= 1


def test_attention_parallel_trace_when_fasttext_loaded():
    engine = build_full_engine()
    assert engine.fasttext_embedding.load() is True

    result = _sample_attention(engine)

    assert result.top_entities
    assert len(engine.attention.fasttext_trace) == 1
    trace = engine.attention.fasttext_trace[-1]
    assert trace["meaning_layer"] == "attention"
    assert trace["mode"] == "parallel_observation"
    assert trace["in_use_by_generation"] == "wrapper"
    assert "self_embedding_result" in trace
    assert "fasttext_result" in trace
    assert trace["fasttext_result"]["loaded"] is True
    assert set(trace["fasttext_result"]["scores"]) == {"오늘", "친구", "좋아"}


def test_attention_decision_unaffected_by_fasttext_observation():
    unloaded_engine = build_full_engine()
    loaded_engine = build_full_engine()
    loaded_engine.fasttext_embedding.load()

    unloaded = _sample_attention(unloaded_engine)
    loaded = _sample_attention(loaded_engine)

    assert loaded_engine.attention.fasttext_trace
    assert unloaded.primary_node == loaded.primary_node
    assert [entity for entity, _ in unloaded.top_entities] == [entity for entity, _ in loaded.top_entities]
    assert set(unloaded.scores) == set(loaded.scores)


def test_attention_trace_data_dict_and_trace_cap():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    engine.attention.fasttext_trace_cap = 3

    for idx in range(5):
        engine.attention.analyze(
            f"오늘 친구 좋아 {idx}",
            ["오늘", "친구", "좋아"],
            emotions={"joy": 0.8},
        )

    assert len(engine.attention.fasttext_trace) == 3
    assert engine.attention.fasttext_observation_count == 5
    assert all(isinstance(item, dict) for item in engine.attention.fasttext_trace)


def test_state_debug_attention_section_unloaded():
    engine = build_full_engine()

    state = engine.state_debug.snapshot_state()
    section = state["attention"]

    assert section["module"] == "attention_analyzer"
    assert section["in_use_by_generation"] == "wrapper"
    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 0
    assert section["migration_stage"] == "attention_parallel_observation_only"


def test_state_debug_attention_section_loaded_after_parallel_trace():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _sample_attention(engine)

    state = engine.state_debug.snapshot_state()
    section = state["attention"]

    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 1
    assert section["fasttext_observation_count"] == 1
    assert section["in_use_by_generation"] == "wrapper"


def test_state_debug_is_read_only_for_attention_migration():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _sample_attention(engine)
    before_trace = list(engine.attention.fasttext_trace)
    before_count = engine.attention.fasttext_observation_count
    before_loaded = engine.fasttext_embedding.is_loaded()

    first = engine.state_debug.snapshot_state()
    second = engine.state_debug.snapshot_state()

    assert first["attention"] == second["attention"]
    assert engine.attention.fasttext_trace == before_trace
    assert engine.attention.fasttext_observation_count == before_count
    assert engine.fasttext_embedding.is_loaded() is before_loaded


def test_generation_path_still_uses_pmi_svd_not_fasttext():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.attention.stats()["in_use_by_generation"] == "wrapper"


def test_round35_does_not_rewrite_core_self_embedding_file():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")
    concept_text = CONCEPT_MEMORY.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text
    # Round37 legitimately adds observation-only fastText hooks to concept memory.
    assert "vectors.npy" not in concept_text
    assert "engine.self_embedding =" not in concept_text
