from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke_with_agp_trace
from adapters.smoke_data_analyzer import (
    ROUND48_ANALYSIS_ROUND,
    ROUND48_ANALYSIS_VERSION,
    compare_pre_post_bridge,
    identify_residual_issues,
)
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def _post(limit=6):
    engine = build_full_engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:limit])
    return engine, result


def test_round48_post_bridge_smoke_executes_all_fixtures():
    engine, result = _post(8)
    assert result["fixtures_count"] == 8
    assert len(result["responses"]) == 8
    assert result["traces"]
    assert result["read_only"] is True


def test_round48_post_bridge_agp_pass_rate_significantly_higher():
    _, result = _post(8)
    comparison = compare_pre_post_bridge(None, result)
    assert comparison["analysis_version"] == ROUND48_ANALYSIS_VERSION
    assert comparison["analysis_round"] == ROUND48_ANALYSIS_ROUND
    assert comparison["agp_pass_rate_change"]["pre"] == 0.0
    assert comparison["agp_pass_rate_change"]["post"] > 0.0
    assert comparison["agp_pass_rate_change"]["delta"] > 0.0


def test_round48_post_bridge_overlap_non_zero():
    _, result = _post(8)
    comparison = compare_pre_post_bridge(None, result)
    assert comparison["overlap_change"]["post_zero_overlap_traces"] == 0
    assert comparison["candidate_category_change"]["post_non_empty_traces"] > 0


def test_round48_pre_post_comparison_returns_data():
    _, result = _post(5)
    comparison = compare_pre_post_bridge({"agp_pass_rate": 0.0, "primary_hit_rate": 0.3542}, result)
    assert isinstance(comparison, dict)
    assert comparison["read_only"] is True
    assert comparison["interpretation_data_dict"] == "manual_decision"


def test_round48_wrapper_telemetry_minimal_change_expected_not_enforced():
    _, result = _post(5)
    comparison = compare_pre_post_bridge(None, result)
    telemetry = comparison["wrapper_telemetry_comparison"]
    assert telemetry["bridge_expected_to_change_lexical_coverage"] is False
    assert telemetry["minimal_change_expected"] is True
    assert "primary_hit_rate_delta" in telemetry


def test_round48_extraction_source_distribution():
    _, result = _post(8)
    comparison = compare_pre_post_bridge(None, result)
    distribution = comparison["extraction_source_distribution"]
    assert distribution.get("meaning_bridge", 0) > 0
    assert distribution.get("text_extraction", 0) == 0


def test_round48_residual_fail_fixtures_identified():
    _, result = _post(6)
    residual = identify_residual_issues(result)
    assert isinstance(residual["residual_failures"], list)
    assert "next_round_options" in residual
    assert residual["recommendation_priority"].startswith("manual_decision")


def test_round48_response_path_change_recorded():
    _, result = _post(8)
    comparison = compare_pre_post_bridge(None, result)
    response_path = comparison["response_path_change"]
    assert response_path["pre_fallback_used_due_to_veto"] == 14
    assert response_path["post_fallback_used_due_to_veto"] == 0


def test_round48_comparison_is_read_only():
    engine, result = _post(5)
    before_thresholds = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    before_telemetry = engine.self_embedding.telemetry()
    snapshot = deepcopy(result)
    compare_pre_post_bridge(None, result)
    identify_residual_issues(result)
    after_thresholds = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    after_telemetry = engine.self_embedding.telemetry()
    assert before_thresholds == after_thresholds
    assert before_telemetry == after_telemetry
    assert result == snapshot


def test_round48_no_threshold_changes():
    engine, result = _post(5)
    before = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    compare_pre_post_bridge(None, result)
    after = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    assert before == after


def test_round48_no_auto_decision():
    _, result = _post(5)
    comparison = compare_pre_post_bridge(None, result)
    residual = identify_residual_issues(result)
    assert comparison["no_threshold_change"] is True
    assert comparison["no_wrapper_change"] is True
    assert comparison["no_subset_change"] is True
    assert residual["no_auto_decision"] is True


def test_round48_full_fixture_post_bridge_baseline_is_stable():
    _, result = _post(len(fixture_texts()))
    comparison = compare_pre_post_bridge(None, result)
    residual = identify_residual_issues(result)
    assert result["fixtures_count"] == 20
    assert comparison["agp_pass_rate_change"]["post"] == 1.0
    assert comparison["candidate_category_change"]["post_zero_candidate_traces"] == 0
    assert comparison["overlap_change"]["post_zero_overlap_traces"] == 0
    assert residual["issue_flags"] == ["no_residual_agp_issue_detected"]
