from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    SMOKE_BASELINE_ROUND,
    compare_fixture_responses,
    normalize_fixture_texts,
    run_conversation_smoke,
)
from tests.fixtures.korean_conversation_fixtures import (
    KOREAN_CONVERSATION_FIXTURES,
    REQUIRED_FIXTURE_CATEGORIES,
    fixture_categories,
    fixture_texts,
)


def test_round43_fixtures_are_deterministic():
    first = fixture_texts()
    second = fixture_texts()
    assert first == second
    assert first == normalize_fixture_texts(KOREAN_CONVERSATION_FIXTURES)
    assert len(first) == 20
    assert len(set(first)) == len(first)


def test_round43_fixtures_cover_diversity():
    categories = set(fixture_categories())
    assert REQUIRED_FIXTURE_CATEGORIES.issubset(categories)
    assert {"greeting", "daily", "emotion", "identity", "reasoning"}.issubset(categories)


def test_round43_smoke_runner_executes_all_fixtures():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts())
    assert result["fixtures_count"] == len(fixture_texts())
    assert len(result["responses"]) == len(fixture_texts())
    assert all(row["input"] for row in result["responses"])
    assert all(isinstance(row["chunks"], list) for row in result["responses"])


def test_round43_smoke_responses_deterministic():
    result = compare_fixture_responses(build_full_engine, fixture_texts()[:6])
    assert result["baseline_round"] == SMOKE_BASELINE_ROUND
    assert result["deterministic"] is True


def test_round43_smoke_telemetry_captured():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts())
    telemetry = result["wrapper_telemetry"]
    assert telemetry["adapter"] == "EmbeddingWrapper"
    assert telemetry["total_calls"] >= result["sampled_token_count"]
    assert telemetry["primary_hits"] + telemetry["fallback_uses"] >= 1


def test_round43_smoke_primary_hit_rate_recorded():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts())
    telemetry = result["wrapper_telemetry"]
    assert "primary_hit_rate" in telemetry
    assert 0.0 <= telemetry["primary_hit_rate"] <= 1.0
    assert telemetry["primary_hits"] > 0
    assert telemetry["primary_hit_rate"] > 0.0


def test_round43_smoke_oov_samples_recorded():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts())
    telemetry = result["wrapper_telemetry"]
    assert telemetry["oov_log_size"] >= 1
    assert isinstance(result["oov_samples_from_run"], list)
    assert result["oov_samples_from_run"]
    assert {"word", "operation"}.issubset(result["oov_samples_from_run"][-1].keys())


def test_round43_smoke_agp_pass_behavior_recorded():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:8])
    agp = result["agp_trace_summary"]
    assert "pass_rate" in agp
    assert 0.0 <= agp["pass_rate"] <= 1.0
    assert isinstance(agp["by_layer"], dict)
    assert isinstance(agp["by_reason"], dict)


def test_round43_smoke_baseline_marked_at_round43():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:3])
    assert result["baseline_round"] == 43
    assert result["seed_drift_baseline"]["baseline_round"] == 42
    assert result["swap_state"] == "engine.self_embedding = wrapper"


def test_round43_smoke_result_is_data_dict():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:3])
    assert isinstance(result, dict)
    for key in [
        "smoke_version",
        "fixtures_count",
        "responses",
        "wrapper_telemetry",
        "agp_trace_summary",
        "oov_samples_from_run",
    ]:
        assert key in result
    assert result["read_only"] is True


def test_round43_smoke_runner_preserves_structural_state():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:5])
    assert result["structural_state_unchanged"] is True
    assert result["structural_state_before"] == result["structural_state_after"]
    assert result["structural_state_after"]["self_embedding_class"] == "EmbeddingWrapper"


def test_round43_smoke_no_rollback_during_run():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:5])
    assert result["rollback_called"] is False
    assert result["rollback_method_preserved"] is True
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_round43_smoke_does_not_change_decision_policy():
    engine = build_full_engine()
    before = engine.self_embedding.telemetry()
    result = run_conversation_smoke(engine, fixture_texts()[:4])
    after = engine.self_embedding.telemetry()
    assert result["decision_policy_changed"] is False
    assert result["thresholds_changed"] is False
    assert after["total_calls"] >= before["total_calls"]
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_round43_smoke_correlation_and_drift_are_read_only_data():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts()[:6])
    correlation = result["agp_wrapper_correlation"]
    drift = result["seed_drift_baseline"]
    assert correlation["interpretation"] == "data_dict_not_recommendation"
    assert correlation["read_only"] is True
    assert drift["no_auto_threshold_adjustment"] is True
    assert drift["no_drift_based_runtime_change"] is True
