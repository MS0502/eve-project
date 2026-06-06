"""EVE v3 round52 — post-medium smoke rerun + primary hit-rate measurement."""

from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke, run_conversation_smoke_with_agp_trace
from adapters.smoke_data_analyzer import (
    ROUND52_ANALYSIS_ROUND,
    ROUND52_ANALYSIS_VERSION,
    compare_three_baselines,
    identify_remaining_lexical_issues,
    project_medium_30k_impact,
    validate_round49_projection_accuracy,
)
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def _post_medium_smoke():
    engine = build_full_engine()
    result = run_conversation_smoke(engine, fixture_texts())
    trace_result = run_conversation_smoke_with_agp_trace(build_full_engine(), fixture_texts())
    # Carry AGP trace pass-rate from the explicit trace runner while keeping
    # wrapper token telemetry from the coverage smoke runner.
    result["agp_trace_summary"] = trace_result["agp_trace_summary"]
    return engine, result, trace_result


def test_round52_post_medium_smoke_executes_all_fixtures():
    _, result, trace = _post_medium_smoke()
    assert result["fixtures_count"] == 20
    assert len(result["responses"]) == 20
    assert trace["fixtures_count"] == 20
    assert result["read_only"] is True


def test_round52_post_medium_primary_hit_rate_reports_no_load_default():
    _, result, _ = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    assert comparison["analysis_version"] == ROUND52_ANALYSIS_VERSION
    assert comparison["analysis_round"] == ROUND52_ANALYSIS_ROUND
    assert comparison["primary_hit_rate_trajectory"]["post_bridge_small_5k"] == 0.25
    # Current default runtime is intentionally no-load unless an operator
    # explicitly authorizes medium30k loading, so artifact-free smoke must not
    # claim a primary-hit improvement.
    assert result["wrapper_telemetry"]["primary_loaded"] is False
    assert comparison["primary_hit_rate_trajectory"]["post_medium_swap_30k"] == 0.0


def test_round52_post_medium_fallback_rate_reports_no_load_default():
    _, result, _ = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    assert comparison["fallback_rate_trajectory"]["post_bridge_small_5k"] == 0.75
    assert result["wrapper_telemetry"]["primary_loaded"] is False
    assert comparison["fallback_rate_trajectory"]["post_medium_swap_30k"] == 1.0


def test_round52_general_korean_oov_remains_honest_when_primary_no_load():
    _, result, _ = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    assert result["wrapper_telemetry"]["primary_loaded"] is False
    assert comparison["oov_pattern_change"]["actual_resolved_in_smoke"] == []
    assert set(comparison["oov_pattern_change"]["actual_general_korean_still_oov_in_smoke"]) == {
        "어때",
        "그래",
        "뭐야",
        "좋아해",
        "군대",
        "코딩",
    }


def test_round52_eve_specific_still_in_oov():
    _, result, _ = _post_medium_smoke()
    remaining = identify_remaining_lexical_issues(result)
    assert "EVE" in remaining["eve_specific_oov_remaining"]
    assert remaining["self_learning_b_priority"] == "high"


def test_round52_three_baseline_comparison_data():
    _, result, _ = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    assert isinstance(comparison, dict)
    assert "primary_hit_rate_trajectory" in comparison
    assert "fallback_rate_trajectory" in comparison
    assert comparison["read_only"] is True


def test_round52_round49_projection_accuracy_records_no_load_low_unexpected():
    _, result, _ = _post_medium_smoke()
    projection = project_medium_30k_impact({"primary_hit_rate": 0.25, "fallback_rate": 0.75}, ["어때", "그래", "EVE"])
    validation = validate_round49_projection_accuracy(projection, result)
    assert validation["analysis_version"] == ROUND52_ANALYSIS_VERSION
    assert validation["projected_range"] == [0.5, 0.65]
    assert validation["actual_primary_hit_rate"] == 0.0
    assert validation["projection_accuracy"] == "low_unexpected"
    assert validation["interpretation_data_dict"] == "manual_decision"


def test_round52_agp_pass_rate_unchanged():
    _, result, trace = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    assert trace["agp_trace_summary"]["pass_rate"] == 1.0
    assert comparison["agp_behavior_unchanged"]["pass_rate"] == 1.0
    assert comparison["agp_behavior_unchanged"]["stability"] == "verified"


def test_round52_smoke_is_read_only_for_analysis():
    engine, result, _ = _post_medium_smoke()
    before_thresholds = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    before_telemetry = engine.self_embedding.telemetry()
    snapshot = deepcopy(result)
    compare_three_baselines(None, None, result)
    validate_round49_projection_accuracy(None, result)
    identify_remaining_lexical_issues(result)
    assert before_thresholds == (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    assert before_telemetry == engine.self_embedding.telemetry()
    assert result == snapshot


def test_round52_no_auto_adjustments():
    _, result, _ = _post_medium_smoke()
    comparison = compare_three_baselines(None, None, result)
    validation = validate_round49_projection_accuracy(None, result)
    remaining = identify_remaining_lexical_issues(result)
    assert comparison["no_wrapper_change"] is True
    assert comparison["no_agp_change"] is True
    assert comparison["no_self_learning_change"] is True
    assert validation["no_auto_adjustment"] is True
    assert remaining["no_auto_vocabulary_addition"] is True


def test_round52_self_learning_b_priority_high():
    _, result, _ = _post_medium_smoke()
    remaining = identify_remaining_lexical_issues(result)
    assert remaining["next_round_53_plus_b_design_required"] is True
    assert remaining["self_learning_b_priority"] == "high"
    assert remaining["no_self_learning_implementation"] is True


def test_round52_report_records_measurement_policy():
    from pathlib import Path

    text = Path("ROUND_V3_R52_REPORT.md").read_text(encoding="utf-8")
    assert "post-medium smoke" in text
    assert "primary_hit_rate" in text
    assert "round53+" in text
