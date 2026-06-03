"""EVE v3 round36 — compositor fastText parallel observation.

Round36 mirrors the AGP observation pattern for embedding migration: compositor
keeps using the existing PMI+SVD self_embedding path for output, while an
explicitly loaded FasttextEmbeddingAdapter can record diagnostics in a separate
trace. AGP traces/veto and fastText traces must remain independent.
"""

from pathlib import Path

from adapters.agp_adapter import AGP_MODE_OBSERVATION, AGP_MODE_VETO, AGP_FALLBACK_SURFACE_POOL
from adapters.compositor_adapter import CompositionResult
from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
CONCEPT_MEMORY = Path("adapters/concept_memory_adapter.py")
COMPOSITOR = Path("adapters/compositor_adapter.py")


def _sample_composition() -> CompositionResult:
    return CompositionResult(
        inputs=("오늘", "친구"),
        concepts=["좋아"],
        method="rule",
        confidence=0.5,
    )


def _phrase(engine):
    return engine.compositor.composition_phrase(_sample_composition())


def test_compositor_default_uses_self_embedding_and_no_fasttext_trace():
    engine = build_full_engine()

    output = _phrase(engine)

    assert output
    assert engine.fasttext_embedding.is_loaded() is True
    assert len(engine.compositor.fasttext_trace) >= 1
    assert engine.compositor.stats()["in_use_by_generation"] == "wrapper"
    assert engine.compositor.stats()["fasttext_parallel_observation"] == "fasttext"


def test_compositor_does_not_auto_load_fasttext():
    engine = build_full_engine()

    for _ in range(3):
        _phrase(engine)

    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000
    assert len(engine.compositor.fasttext_trace) >= 1


def test_compositor_parallel_trace_when_fasttext_loaded():
    engine = build_full_engine()
    assert engine.fasttext_embedding.load() is True

    output = _phrase(engine)

    assert output
    assert len(engine.compositor.fasttext_trace) == 1
    trace = engine.compositor.fasttext_trace[-1]
    assert trace["meaning_layer"] == "compositor"
    assert trace["mode"] == "parallel_observation"
    assert trace["in_use_by_generation"] == "wrapper"
    assert "self_embedding_result" in trace
    assert "fasttext_result" in trace
    assert trace["fasttext_result"]["loaded"] is True
    assert trace["fasttext_result"]["subset_name"] == "cc.ko.300.subset.medium.30k"


def test_compositor_output_unaffected_by_fasttext_observation():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    unloaded_output = _phrase(unloaded)
    loaded_output = _phrase(loaded)

    assert loaded.compositor.fasttext_trace
    assert unloaded_output == loaded_output


def test_compositor_agp_path_unaffected_by_fasttext():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding.load()

    unloaded_output = _phrase(unloaded)
    loaded_output = _phrase(loaded)

    assert unloaded_output == loaded_output
    assert len(unloaded.compositor.agp_trace) == 1
    assert len(loaded.compositor.agp_trace) == 1
    assert unloaded.compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert loaded.compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert len(unloaded.compositor.fasttext_trace) >= 1
    assert len(loaded.compositor.fasttext_trace) == 1


def test_compositor_agp_veto_and_fasttext_observation_coexist_safely():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.compositor.agp_mode = AGP_MODE_VETO

    output = _phrase(engine)

    assert output == "그렇구나."
    assert getattr(engine.compositor.agp_trace[-1]["result"], "passed", False) is True
    assert engine.compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert engine.compositor.agp_veto_count == 1
    assert len(engine.compositor.fasttext_trace) == 1
    ft_trace = engine.compositor.fasttext_trace[-1]
    assert ft_trace["mode"] == "parallel_observation"
    assert ft_trace["agp_mode"] == AGP_MODE_VETO
    assert ft_trace["agp_result_passed"] is True


def test_compositor_traces_are_separate_data_structures():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    _phrase(engine)

    assert engine.compositor.agp_trace
    assert engine.compositor.fasttext_trace
    assert engine.compositor.agp_trace is not engine.compositor.fasttext_trace
    assert "candidate_response" in engine.compositor.agp_trace[-1]
    assert "fasttext_result" in engine.compositor.fasttext_trace[-1]


def test_compositor_fasttext_trace_cap():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    engine.compositor.fasttext_trace_cap = 3

    for _ in range(5):
        _phrase(engine)

    assert len(engine.compositor.fasttext_trace) == 3
    assert engine.compositor.fasttext_observation_count == 5
    assert all(isinstance(item, dict) for item in engine.compositor.fasttext_trace)


def test_state_debug_compositor_section_unloaded():
    engine = build_full_engine()

    section = engine.state_debug.snapshot_state()["compositor"]

    assert section["module"] == "compositor_adapter"
    assert section["in_use_by_generation"] == "wrapper"
    assert section["parallel_observation"] == "fasttext"
    assert section["agp_mode"] == AGP_MODE_OBSERVATION
    assert section["migration_stage"] == "compositor_parallel_observation_only"


def test_state_debug_compositor_section_loaded_after_parallel_trace():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _phrase(engine)

    section = engine.state_debug.snapshot_state()["compositor"]

    assert section["parallel_observation"] == "fasttext"
    assert section["fasttext_trace_size"] == 1
    assert section["fasttext_observation_count"] == 1
    assert section["in_use_by_generation"] == "wrapper"


def test_state_debug_is_read_only_for_compositor_migration():
    engine = build_full_engine()
    engine.fasttext_embedding.load()
    _phrase(engine)
    before_fasttext_trace = list(engine.compositor.fasttext_trace)
    before_agp_trace = list(engine.compositor.agp_trace)
    before_fasttext_count = engine.compositor.fasttext_observation_count
    before_agp_count = engine.compositor.agp_observation_count

    first = engine.state_debug.snapshot_state()["compositor"]
    second = engine.state_debug.snapshot_state()["compositor"]

    assert first == second
    assert engine.compositor.fasttext_trace == before_fasttext_trace
    assert engine.compositor.agp_trace == before_agp_trace
    assert engine.compositor.fasttext_observation_count == before_fasttext_count
    assert engine.compositor.agp_observation_count == before_agp_count


def test_embedding_generation_path_still_uses_pmi_svd_not_fasttext():
    engine = build_full_engine()
    engine.fasttext_embedding.load()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.compositor.stats()["in_use_by_generation"] == "wrapper"


def test_round36_does_not_rewrite_core_self_embedding_file():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")
    concept_text = CONCEPT_MEMORY.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text
    # Round37 legitimately adds observation-only fastText hooks to concept memory.
    assert "vectors.npy" not in concept_text
    assert "engine.self_embedding =" not in concept_text


def test_round36_keeps_agp_and_fasttext_concerns_documented_as_independent():
    text = COMPOSITOR.read_text(encoding="utf-8")

    assert "fastText parallel observation" in text
    assert "independent from AGP" in text
    assert "This must never affect composition output" in text
