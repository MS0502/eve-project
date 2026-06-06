"""EVE v3 round40 — pre-swap audit before main.py self_embedding swap.

Round40 is an audit-only safety gate between 6/7 parallel-observation
migration and the round41 engine.self_embedding swap. It verifies invariants,
fail-open behavior, AGP/fastText/self_embedding coexistence, and readiness data
without changing main.py or swapping the active embedding owner.
"""

from __future__ import annotations

from pathlib import Path

from adapters.agp_adapter import AGP_MODE_VETO, AGP_FALLBACK_SURFACE_POOL
from adapters.compositor_adapter import CompositionResult
from adapters.external_seed_manifest import assess_main_py_swap_readiness
from main import build_full_engine
from utils.types import Meaning

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
MAIN = Path("main.py")
ROUND40_TEST = Path("tests/test_v3_round40_pre_swap_audit.py")
EXTERNAL_SEED_MANIFEST = Path("adapters/external_seed_manifest.py")

TRACE_MODULES = {
    "attention": lambda e: e.attention,
    "compositor": lambda e: e.compositor,
    "concept_memory": lambda e: e.concept_memory,
    "situation_responder": lambda e: e.situation_responder,
    "streaming": lambda e: e,
}
ALL_6_MODULES = tuple(TRACE_MODULES) + ("state_debug",)


def _sample_attention(engine):
    return engine.attention.analyze(
        "오늘 친구 좋아",
        ["오늘", "친구", "좋아"],
        emotions={"joy": 0.8},
    )


def _sample_composition() -> CompositionResult:
    return CompositionResult(
        inputs=("오늘", "친구"),
        concepts=["좋아"],
        method="rule",
        confidence=0.5,
    )


def _sample_compositor(engine):
    return engine.compositor.composition_phrase(_sample_composition())


def _sample_concept_memory(engine):
    learned = engine.concept_memory.learn("친구", "함께 시간을 보내는 사람")
    found = engine.concept_memory.get("친구")
    return learned, found


def _meaning(text="안녕"):
    m = Meaning(intent="statement", entities=[])
    m.raw_text = text
    return m


def _ctx():
    return {"user_presence": {"user_name": "민석", "intimacy": 0.8}}


def _sample_situation(engine):
    return engine.situation_responder.build_plan("greeting", _ctx(), _meaning("안녕"))


def _sample_streaming(engine, text="민석아 힘들다"):
    return list(engine.chat_stream(text))


def _exercise_trace_modules(engine):
    return {
        "attention": _sample_attention(engine),
        "compositor": _sample_compositor(engine),
        "concept_memory": _sample_concept_memory(engine),
        "situation_responder": _sample_situation(engine),
        "streaming": _sample_streaming(engine),
    }


def _trace_size(module):
    return len(getattr(module, "fasttext_trace", []) or [])


def _observation_count(module):
    return int(getattr(module, "fasttext_observation_count", 0) or 0)


class BrokenFasttextAdapter:
    """Loaded-looking adapter whose lookup methods fail.

    Used to verify that all round35~39 observation hooks are fail-open and do
    not contaminate the active PMI+SVD self_embedding path.
    """

    subset_name = "broken.fasttext.fixture"
    dimension = 300

    def __init__(self, *, loaded=True):
        self._loaded = loaded

    def is_loaded(self):
        return self._loaded

    def load(self):
        raise RuntimeError("load_failed_fixture")

    def stats(self):
        return {
            "method": "broken_fixture",
            "dimension": self.dimension,
            "subset_name": self.subset_name,
            "loaded": self._loaded,
            "vocab_size": 0,
            "vectors_shape": None,
        }

    def get_vector(self, *_args, **_kwargs):
        raise RuntimeError("lookup_failed_fixture")

    def get_embedding(self, *_args, **_kwargs):
        raise RuntimeError("embedding_failed_fixture")

    def text_similarity(self, *_args, **_kwargs):
        raise RuntimeError("similarity_failed_fixture")


class OOVFasttextAdapter(BrokenFasttextAdapter):
    """Loaded adapter that returns OOV/zero signals instead of exceptions."""

    subset_name = "oov.fasttext.fixture"

    def load(self):
        self._loaded = True
        return True

    def get_vector(self, *_args, **_kwargs):
        return None

    def get_embedding(self, *_args, **_kwargs):
        return None

    def text_similarity(self, *_args, **_kwargs):
        return 0.0


def test_round40_all_6_modules_are_present_for_pre_swap_audit():
    engine = build_full_engine()

    assert all(TRACE_MODULES[name](engine) is not None for name in TRACE_MODULES)
    assert engine.state_debug is not None
    assert sorted(ALL_6_MODULES) == [
        "attention",
        "compositor",
        "concept_memory",
        "situation_responder",
        "state_debug",
        "streaming",
    ]


def test_round40_all_trace_modules_in_use_by_self_embedding():
    engine = build_full_engine()

    for name, getter in TRACE_MODULES.items():
        assert getter(engine).stats()["in_use_by_generation"] == "wrapper", name
    assert engine.state_debug.snapshot_state()["self_embedding"]["module"] == "embedding_wrapper"


def test_round40_all_trace_modules_have_parallel_observation_option_unloaded():
    engine = build_full_engine()

    for name, getter in TRACE_MODULES.items():
        stats = getter(engine).stats()
        assert stats["fasttext_parallel_observation"] is None, name
        assert stats["fasttext_trace_size"] == 0, name
        assert stats["in_use_by_generation"] == "wrapper", name
    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.self_embedding.stats()["primary_loaded"] is False


def test_round40_trace_modules_use_data_dict_pattern_when_loaded():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    engine.self_embedding.primary = engine.fasttext_embedding

    _exercise_trace_modules(engine)

    for name, getter in TRACE_MODULES.items():
        trace = getattr(getter(engine), "fasttext_trace")
        assert trace, name
        assert all(isinstance(item, dict) for item in trace), name
        assert trace[-1]["mode"] == "parallel_observation"
        assert trace[-1]["in_use_by_generation"] == "wrapper"


def test_round40_all_trace_modules_have_trace_cap_1000():
    engine = build_full_engine()

    for name, getter in TRACE_MODULES.items():
        module = getter(engine)
        assert getattr(module, "fasttext_trace_cap") == 1000, name
        assert module.stats()["fasttext_trace_cap"] == 1000


def test_round40_fasttext_load_activates_all_trace_modules():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    engine.self_embedding.primary = engine.fasttext_embedding

    _exercise_trace_modules(engine)

    for name, getter in TRACE_MODULES.items():
        stats = getter(engine).stats()
        assert stats["fasttext_parallel_observation"] == "fasttext", name
        assert stats["fasttext_trace_size"] > 0, name
        assert stats["fasttext_observation_count"] > 0, name


def test_round40_simultaneous_traces_accumulate_independently():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    engine.self_embedding.primary = engine.fasttext_embedding

    _exercise_trace_modules(engine)
    before = {name: _trace_size(getter(engine)) for name, getter in TRACE_MODULES.items()}
    _exercise_trace_modules(engine)
    after = {name: _trace_size(getter(engine)) for name, getter in TRACE_MODULES.items()}

    for name in TRACE_MODULES:
        # Streaming can intentionally suppress duplicate same-input chunks via
        # dialogue/timing guards, so require no cross-module corruption there
        # and strict accumulation for the non-streaming observation modules.
        if name == "streaming":
            assert after[name] >= before[name], name
        else:
            assert after[name] > before[name], name
    assert engine.attention.fasttext_trace is not engine.compositor.fasttext_trace
    assert engine.compositor.fasttext_trace is not engine.concept_memory.fasttext_trace
    assert engine.concept_memory.fasttext_trace is not engine.situation_responder.fasttext_trace
    assert engine.situation_responder.fasttext_trace is not engine.fasttext_trace


def test_round40_decisions_unchanged_across_all_trace_modules():
    unloaded = build_full_engine()
    loaded = build_full_engine()
    loaded.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    loaded.self_embedding.primary = loaded.fasttext_embedding

    unloaded_results = _exercise_trace_modules(unloaded)
    loaded_results = _exercise_trace_modules(loaded)

    assert unloaded_results["attention"].primary_node == loaded_results["attention"].primary_node
    assert [x for x, _ in unloaded_results["attention"].top_entities] == [x for x, _ in loaded_results["attention"].top_entities]
    assert unloaded_results["compositor"] == loaded_results["compositor"]
    assert unloaded_results["concept_memory"][1].definition == loaded_results["concept_memory"][1].definition
    assert unloaded_results["situation_responder"].core_message == loaded_results["situation_responder"].core_message
    assert unloaded_results["streaming"] == loaded_results["streaming"]


def test_round40_fasttext_load_failure_does_not_break_modules():
    engine = build_full_engine()
    engine.fasttext_embedding = BrokenFasttextAdapter(loaded=False)

    try:
        engine.fasttext_embedding.load()
    except RuntimeError as exc:
        assert "load_failed_fixture" in str(exc)

    results = _exercise_trace_modules(engine)

    assert results["attention"] is not None
    assert results["compositor"]
    assert results["concept_memory"][1].definition == "함께 시간을 보내는 사람"
    assert results["situation_responder"].core_message
    assert results["streaming"]
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_round40_fasttext_lookup_failure_isolated_from_decisions():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    engine.self_embedding.primary = engine.fasttext_embedding

    results = _exercise_trace_modules(engine)

    assert results["attention"].primary_node is not None
    assert results["compositor"]
    assert results["concept_memory"][1].definition == "함께 시간을 보내는 사람"
    assert results["situation_responder"].core_message
    assert results["streaming"]
    for name, getter in TRACE_MODULES.items():
        assert _trace_size(getter(engine)) > 0, name


def test_round40_fasttext_exception_during_parallel_observation_records_error_and_continues():
    engine = build_full_engine()
    engine.fasttext_embedding = BrokenFasttextAdapter(loaded=True)

    results = _exercise_trace_modules(engine)

    assert results["streaming"]
    for name, getter in TRACE_MODULES.items():
        trace = getattr(getter(engine), "fasttext_trace")
        assert trace, name
        assert any("error" in item for item in trace), name
        assert _observation_count(getter(engine)) >= 1, name


def test_round40_fasttext_corruption_does_not_propagate_to_self_embedding_path():
    clean = build_full_engine()
    corrupted = build_full_engine()
    corrupted.fasttext_embedding = BrokenFasttextAdapter(loaded=True)

    clean_results = _exercise_trace_modules(clean)
    corrupted_results = _exercise_trace_modules(corrupted)

    assert clean_results["attention"].primary_node == corrupted_results["attention"].primary_node
    assert clean_results["compositor"] == corrupted_results["compositor"]
    assert clean_results["concept_memory"][1].definition == corrupted_results["concept_memory"][1].definition
    assert clean_results["situation_responder"].core_message == corrupted_results["situation_responder"].core_message
    assert clean_results["streaming"] == corrupted_results["streaming"]


def test_round40_fasttext_unload_during_runtime_gracefully_stops_parallel_observation():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    _exercise_trace_modules(engine)
    before = {name: _trace_size(getter(engine)) for name, getter in TRACE_MODULES.items()}

    engine.fasttext_embedding._loaded = False
    results = _exercise_trace_modules(engine)
    after = {name: _trace_size(getter(engine)) for name, getter in TRACE_MODULES.items()}

    assert results["streaming"]
    assert after == before
    assert engine.fasttext_embedding.is_loaded() is False


def test_round40_three_systems_coexist_safely_with_agp_veto():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.compositor.agp_mode = AGP_MODE_VETO

    output = _sample_compositor(engine)

    assert output == "그렇구나."
    assert getattr(engine.compositor.agp_trace[-1]["result"], "passed", False) is True
    assert engine.compositor.agp_veto_count == 1
    assert engine.compositor.fasttext_trace
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_round40_three_systems_traces_are_separated():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    _sample_compositor(engine)
    _sample_attention(engine)

    assert engine.compositor.agp_trace
    assert engine.compositor.fasttext_trace
    assert engine.compositor.agp_trace is not engine.compositor.fasttext_trace
    assert engine.attention.fasttext_trace is not engine.compositor.fasttext_trace
    assert "fasttext_result" in engine.compositor.fasttext_trace[-1]
    assert "result" in engine.compositor.agp_trace[-1]


def test_round40_three_systems_decisions_independent():
    unloaded = build_full_engine()
    loaded_veto = build_full_engine()
    loaded_veto.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    loaded_veto.self_embedding.primary = loaded_veto.fasttext_embedding
    loaded_veto.agp_adapter.set_mode(AGP_MODE_VETO)
    loaded_veto.compositor.agp_mode = AGP_MODE_VETO

    normal = _sample_compositor(unloaded)
    vetoed = _sample_compositor(loaded_veto)

    assert normal not in AGP_FALLBACK_SURFACE_POOL or vetoed in AGP_FALLBACK_SURFACE_POOL
    assert loaded_veto.compositor.fasttext_trace[-1]["mode"] == "parallel_observation"
    assert loaded_veto.compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert loaded_veto.compositor.stats()["in_use_by_generation"] == "wrapper"


def test_round40_swap_readiness_assessment_returns_data_dict_ready():
    engine = build_full_engine()

    report = assess_main_py_swap_readiness(engine)

    assert isinstance(report, dict)
    assert report["audit_version"] == "v3_round40_pre_swap_audit"
    assert report["swap_readiness"] == "ready"
    assert report["all_6_modules_migrated"] is True
    assert report["fasttext_adapter_ready"] is True
    assert report["self_embedding_is_pmi_svd_50d"] is True
    assert report["concerns"] == []
    assert report["read_only"] is True


def test_round40_swap_readiness_assessment_is_read_only():
    engine = build_full_engine()
    before_loaded = engine.fasttext_embedding.is_loaded()
    before_self = engine.self_embedding
    before_snapshot = engine.state_debug.snapshot_state()

    first = assess_main_py_swap_readiness(engine)
    second = assess_main_py_swap_readiness(engine)
    after_snapshot = engine.state_debug.snapshot_state()

    assert first == second
    assert engine.fasttext_embedding.is_loaded() is before_loaded
    assert engine.self_embedding is before_self
    assert before_snapshot == after_snapshot


def test_round40_engine_self_embedding_still_pmi_svd_and_main_py_unchanged():
    engine = build_full_engine()
    main_text = MAIN.read_text(encoding="utf-8")

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert "engine.self_embedding = EmbeddingWrapper(" in main_text
    assert "engine.self_embedding = engine.fasttext_embedding" not in main_text


def test_round40_audit_is_read_only_for_existing_traces():
    engine = build_full_engine()
    engine.fasttext_embedding = OOVFasttextAdapter(loaded=True)
    _exercise_trace_modules(engine)
    before = {
        name: (list(getattr(getter(engine), "fasttext_trace")), _observation_count(getter(engine)))
        for name, getter in TRACE_MODULES.items()
    }

    _ = assess_main_py_swap_readiness(engine)
    _ = engine.state_debug.snapshot_state()

    after = {
        name: (list(getattr(getter(engine), "fasttext_trace")), _observation_count(getter(engine)))
        for name, getter in TRACE_MODULES.items()
    }
    assert after == before


def test_round40_has_no_final_swap_or_default_fasttext_load():
    engine = build_full_engine()

    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.self_embedding is not engine.fasttext_embedding
    assert engine.self_embedding.stats()["primary_loaded"] is False


def test_round40_code_change_boundary_is_audit_only():
    main_text = MAIN.read_text(encoding="utf-8")
    manifest_text = EXTERNAL_SEED_MANIFEST.read_text(encoding="utf-8")
    round40_text = ROUND40_TEST.read_text(encoding="utf-8")

    assert "round41" in manifest_text
    assert "assess_main_py_swap_readiness" in manifest_text
    assert "pre-swap audit" in round40_text
    assert "engine.self_embedding = engine.fasttext_embedding" not in main_text


def test_round40_self_embedding_adapter_still_unmodified_for_fasttext_runtime():
    self_text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert "FasttextEmbeddingAdapter" not in self_text
    assert "fasttext_embedding" not in self_text
    assert "vectors.npy" not in self_text
