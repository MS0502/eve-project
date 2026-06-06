from copy import deepcopy

from main import build_full_engine
from adapters.external_seed_manifest import (
    correlate_agp_and_wrapper_telemetry,
    measure_seed_drift_baseline,
)


def _fallback_only_word(engine):
    word = "round42_fallback_known"
    engine.self_embedding_backup.embeddings[word] = [1.0, 0.0]
    return word


def test_round42_wrapper_telemetry_returns_data_dict():
    engine = build_full_engine()
    data = engine.self_embedding.telemetry()
    assert isinstance(data, dict)
    assert data["telemetry_version"] == "v3_round42_wrapper_telemetry"
    assert data["read_only"] is True


def test_round42_wrapper_reports_no_primary_hits_when_fasttext_unloaded():
    engine = build_full_engine()
    word = _fallback_only_word(engine)
    engine.self_embedding.get_vector(word)
    data = engine.self_embedding.telemetry()
    assert data["total_calls"] == 1
    assert data["primary_loaded"] is False
    assert data["primary_hits"] == 0
    assert data["primary_hit_rate"] == 0.0
    assert data["fallback_uses"] >= 1
    assert data["fallback_rate"] > 0.0


def test_round42_wrapper_fallback_rate():
    engine = build_full_engine()
    engine.self_embedding_backup.embeddings["fallback_only_round42"] = [1.0, 0.0]
    assert engine.self_embedding.get_embedding("fallback_only_round42") == [1.0, 0.0]
    data = engine.self_embedding.telemetry()
    assert data["total_calls"] == 1
    assert data["fallback_uses"] >= 1
    assert data["fallback_rate"] > 0


def test_round42_wrapper_error_rate():
    engine = build_full_engine()

    class BrokenPrimary:
        def is_loaded(self):
            return True
        def get_vector(self, word):
            raise RuntimeError("boom")
        def get_dimension(self):
            return 300

    engine.self_embedding.primary = BrokenPrimary()
    engine.self_embedding.get_vector("safe_error_word")
    data = engine.self_embedding.telemetry()
    assert data["errors"] >= 1
    assert data["error_rate"] > 0
    assert data["fallback_uses"] >= 1


def test_round42_wrapper_oov_log_tracking():
    engine = build_full_engine()
    token = "round42_unknown_oov_token"
    engine.self_embedding.get_vector(token)
    data = engine.self_embedding.telemetry()
    assert data["oov_log_size"] == 1
    assert data["recent_oov_sample"][-1] == {"word": token, "operation": "get_vector"}


def test_round42_wrapper_oov_log_cap():
    engine = build_full_engine()
    wrapper = engine.self_embedding
    wrapper._oov_log_cap = 3
    for idx in range(5):
        wrapper.get_vector(f"round42_oov_{idx}")
    data = wrapper.telemetry()
    assert data["oov_log_size"] == 3
    assert [row["word"] for row in data["recent_oov_sample"]] == [
        "round42_oov_2",
        "round42_oov_3",
        "round42_oov_4",
    ]


def test_round42_wrapper_telemetry_is_read_only():
    engine = build_full_engine()
    engine.self_embedding.get_vector(_fallback_only_word(engine))
    wrapper = engine.self_embedding
    before = (
        wrapper._call_count,
        wrapper._primary_count,
        wrapper._fallback_count,
        wrapper._error_count,
        list(wrapper._oov_log),
    )
    first = wrapper.telemetry()
    second = wrapper.telemetry()
    after = (
        wrapper._call_count,
        wrapper._primary_count,
        wrapper._fallback_count,
        wrapper._error_count,
        list(wrapper._oov_log),
    )
    assert first == second
    assert before == after


def test_round42_drift_baseline_recorded_at_round42():
    engine = build_full_engine()
    report = measure_seed_drift_baseline(engine)
    assert report["baseline_round"] == 42
    assert report["status"] == "baseline_set_at_round42"
    assert report["fasttext_vector_unchanged"] is True
    assert report["evespecific_learning_substrate"] == "PMI+SVD (fallback path)"


def test_round42_drift_measurement_data_dict():
    engine = build_full_engine()
    engine.self_embedding.get_vector(_fallback_only_word(engine))
    report = measure_seed_drift_baseline(engine)
    assert isinstance(report, dict)
    assert isinstance(report["drift_indicators"], dict)
    assert report["drift_indicators"]["primary_hit_rate"] == 0.0
    assert report["drift_indicators"]["fallback_rate"] > 0.0
    assert report["read_only"] is True


def test_round42_drift_target_documented():
    engine = build_full_engine()
    report = measure_seed_drift_baseline(engine)
    assert "round 200+ avg drift > 0.3" in report["target"]
    assert "v3.1" in report["target"]


def test_round42_no_auto_threshold_adjustment():
    engine = build_full_engine()
    before_category = engine.agp_adapter.category_threshold
    before_hormone = engine.agp_adapter.hormone_threshold
    before_wrapper = engine.self_embedding.telemetry()
    report = measure_seed_drift_baseline(engine)
    assert report["no_auto_threshold_adjustment"] is True
    assert report["no_drift_based_runtime_change"] is True
    assert engine.agp_adapter.category_threshold == before_category
    assert engine.agp_adapter.hormone_threshold == before_hormone
    assert engine.self_embedding.telemetry() == before_wrapper


def test_round42_agp_wrapper_correlation_is_read_only():
    engine = build_full_engine()
    before_compositor = deepcopy(engine.compositor.agp_trace)
    before_speech = deepcopy(engine.speech_hub.agp_trace)
    before_telemetry = engine.self_embedding.telemetry()
    report = correlate_agp_and_wrapper_telemetry(engine)
    assert report["interpretation"] == "data_dict_not_recommendation"
    assert report["no_auto_threshold_adjustment"] is True
    assert report["read_only"] is True
    assert engine.compositor.agp_trace == before_compositor
    assert engine.speech_hub.agp_trace == before_speech
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round42_telemetry_does_not_change_decision():
    engine = build_full_engine()
    word = _fallback_only_word(engine)
    before = list(engine.self_embedding.get_vector(word))
    _ = engine.self_embedding.telemetry()
    after = list(engine.self_embedding.get_vector(word))
    assert before == after


def test_round42_state_debug_exposes_wrapper_telemetry_and_seed_drift():
    engine = build_full_engine()
    engine.self_embedding.get_vector(_fallback_only_word(engine))
    state = engine.state_debug.snapshot_state()
    assert state["wrapper_telemetry"]["primary_hit_rate"] == 0.0
    assert state["wrapper_telemetry"]["fallback_rate"] > 0.0
    assert state["wrapper_telemetry"]["read_only"] is True
    assert state["seed_drift"]["baseline_round"] == 42
    assert state["seed_drift"]["status"] == "tracking_started"
