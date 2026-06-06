from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke
from adapters.smoke_data_analyzer import (
    ROUND44_ANALYSIS_ROUND,
    ROUND44_ANALYSIS_VERSION,
    analyze_smoke_data,
)
from tests.fixtures.korean_conversation_fixtures import (
    KOREAN_CONVERSATION_FIXTURES,
    fixture_texts,
)


def _sample_analysis(limit=20):
    engine = build_full_engine()
    fixtures = KOREAN_CONVERSATION_FIXTURES[:limit]
    smoke = run_conversation_smoke(engine, fixture_texts()[:limit])
    return smoke, analyze_smoke_data(smoke, fixtures)


def test_round44_analysis_returns_data_dict():
    smoke, analysis = _sample_analysis(4)
    assert isinstance(analysis, dict)
    assert analysis["analysis_version"] == ROUND44_ANALYSIS_VERSION
    assert analysis["analysis_round"] == ROUND44_ANALYSIS_ROUND
    assert analysis["baseline_round"] == smoke["baseline_round"]
    assert analysis["read_only"] is True


def test_round44_analysis_is_data_only_no_actions():
    _, analysis = _sample_analysis(4)
    assert analysis["data_only"] is True
    assert analysis["decision_policy_changed"] is False
    assert analysis["thresholds_changed"] is False
    assert analysis["subset_promoted"] is False
    assert analysis["fallback_removed"] is False


def test_round44_telemetry_summary_preserves_round43_rates():
    smoke, analysis = _sample_analysis(20)
    telemetry = analysis["telemetry_summary"]
    source = smoke["wrapper_telemetry"]
    assert telemetry["total_calls"] == source["total_calls"]
    assert telemetry["primary_hits"] == source["primary_hits"]
    assert telemetry["fallback_uses"] == source["fallback_uses"]
    assert telemetry["primary_hit_rate"] == source["primary_hit_rate"]
    assert telemetry["fallback_rate"] == source["fallback_rate"]
    assert telemetry["error_rate"] == source["error_rate"]


def test_round44_no_load_coverage_flag_for_current_smoke():
    _, analysis = _sample_analysis(20)
    telemetry = analysis["telemetry_summary"]
    assert telemetry["primary_hit_rate"] == 0.0
    assert telemetry["coverage_status"] == "low_coverage"
    assert "primary_hit_rate_below_0_50" in analysis["manual_review_flags"]


def test_round44_oov_analysis_groups_known_samples():
    _, analysis = _sample_analysis(20)
    oov = analysis["oov_analysis"]
    assert oov["sample_size"] >= 1
    assert oov["unique_oov_count"] >= 1
    assert isinstance(oov["group_counts"], dict)
    assert "project_identity_term" in oov["group_counts"] or "korean_general_oov" in oov["group_counts"]
    assert oov["interpretation"] == "oov_sample_is_bounded_recent_sample_not_full_vocab_audit"


def test_round44_oov_analysis_maps_fixture_categories_without_requiring_primary_load():
    _, analysis = _sample_analysis(20)
    by_category = analysis["oov_analysis"]["by_fixture_category"]
    assert "minsok" in by_category
    assert "identity" in by_category
    observed = by_category["minsok"]["observed_oov_tokens"] + by_category["identity"]["observed_oov_tokens"]
    assert isinstance(observed, list)
    assert analysis["telemetry_summary"]["primary_hit_rate"] == 0.0


def test_round44_fixture_category_summary_preserved():
    _, analysis = _sample_analysis(20)
    summary = analysis["fixture_category_summary"]
    assert summary["fixture_count"] == 20
    assert summary["category_counts"]["greeting"] == 2
    assert summary["category_counts"]["reasoning"] == 4
    assert "안녕" in summary["examples_by_category"]["greeting"]


def test_round44_agp_unknown_category_analysis_recorded():
    _, analysis = _sample_analysis(20)
    agp = analysis["agp_unknown_category_analysis"]
    assert "pass_rate" in agp
    assert "reason_counts" in agp
    assert agp["unknown_category_count"] == 0
    assert agp["no_threshold_change"] is True
    assert "agp_unknown_category_present" not in analysis["manual_review_flags"]


def test_round44_agp_failure_interpretation_separates_runtime_errors():
    _, analysis = _sample_analysis(20)
    agp = analysis["agp_unknown_category_analysis"]
    telemetry = analysis["telemetry_summary"]
    assert telemetry["no_runtime_error_signal"] is True
    if agp["all_failures_unknown_category"]:
        assert agp["first_pass_interpretation"] == "agp_anchor_coverage_gap_not_fasttext_runtime_error"


def test_round44_next_round_candidates_are_manual_only():
    _, analysis = _sample_analysis(20)
    candidates = analysis["next_round_candidates"]
    assert candidates
    assert all(item["auto_apply"] is False for item in candidates)
    names = {item["candidate"] for item in candidates}
    assert "category_coverage_breakdown" in names
    assert "agp_unknown_category_root_cause_analysis" not in names


def test_round44_analysis_read_only_for_smoke_result():
    smoke, _ = _sample_analysis(6)
    before = deepcopy(smoke)
    analysis = analyze_smoke_data(smoke, KOREAN_CONVERSATION_FIXTURES[:6])
    assert smoke == before
    assert analysis["read_only"] is True


def test_round44_analysis_does_not_mutate_engine_state():
    engine = build_full_engine()
    smoke = run_conversation_smoke(engine, fixture_texts()[:6])
    before = engine.self_embedding.telemetry()
    analyze_smoke_data(smoke, KOREAN_CONVERSATION_FIXTURES[:6])
    after = engine.self_embedding.telemetry()
    assert after == before
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"


def test_round44_data_quality_marks_round43_baseline():
    smoke, analysis = _sample_analysis(5)
    quality = analysis["data_quality"]
    assert quality["baseline_round"] == 43
    assert quality["baseline_matches_round43"] is True
    assert quality["has_wrapper_telemetry"] is True
    assert quality["has_agp_summary"] is True


def test_round44_medium_subset_candidate_remains_manual_when_no_load():
    _, analysis = _sample_analysis(20)
    candidates = analysis["next_round_candidates"]
    names = {item["candidate"] for item in candidates}
    assert "medium_30k_subset_evaluation_only" in names
    assert all(item["auto_apply"] is False for item in candidates)
    assert analysis["subset_promoted"] is False
