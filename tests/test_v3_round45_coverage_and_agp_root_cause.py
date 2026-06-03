from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke
from adapters.smoke_data_analyzer import (
    ROUND45_ANALYSIS_ROUND,
    ROUND45_ANALYSIS_VERSION,
    analyze_agp_unknown_category_root_cause,
    analyze_category_coverage,
    confirm_problem_separation,
)
from tests.fixtures.korean_conversation_fixtures import (
    KOREAN_CONVERSATION_FIXTURES,
    fixture_texts,
)


def _sample_smoke(limit=20):
    engine = build_full_engine()
    smoke = run_conversation_smoke(engine, fixture_texts()[:limit])
    fixtures = KOREAN_CONVERSATION_FIXTURES[:limit]
    return engine, smoke, fixtures


def test_round45_category_coverage_returns_data_dict():
    _, smoke, fixtures = _sample_smoke(6)
    result = analyze_category_coverage(smoke, fixtures)
    assert isinstance(result, dict)
    assert result["analysis_version"] == ROUND45_ANALYSIS_VERSION
    assert result["analysis_round"] == ROUND45_ANALYSIS_ROUND
    assert result["read_only"] is True


def test_round45_category_coverage_breakdown_by_required_categories():
    _, smoke, fixtures = _sample_smoke(20)
    result = analyze_category_coverage(smoke, fixtures)
    by_category = result["by_category"]
    for category in ["greeting", "daily", "emotion", "identity", "reasoning"]:
        assert category in by_category
    assert result["required_categories_present"] is True


def test_round45_category_coverage_identifies_best_and_worst():
    _, smoke, fixtures = _sample_smoke(20)
    result = analyze_category_coverage(smoke, fixtures)
    assert result["best_covered"] in result["by_category"]
    assert result["worst_covered"] in result["by_category"]
    assert result["by_category"][result["worst_covered"]]["coverage_status"] in {
        "low_coverage",
        "partial_coverage",
        "good_coverage",
    }


def test_round45_medium_30k_projection_is_data_dict_not_action():
    _, smoke, fixtures = _sample_smoke(20)
    result = analyze_category_coverage(smoke, fixtures)
    projection = result["medium_30k_projected_impact"]
    assert isinstance(projection, dict)
    assert projection["auto_extract_medium_30k"] is False
    assert "rationale" in projection


def test_round45_agp_root_cause_returns_data_dict():
    engine, smoke, _ = _sample_smoke(8)
    result = analyze_agp_unknown_category_root_cause(smoke, engine)
    assert isinstance(result, dict)
    assert result["analysis_round"] == ROUND45_ANALYSIS_ROUND
    assert result["recommendation_priority"] == "manual_decision"
    assert result["read_only"] is True


def test_round45_agp_root_cause_lists_four_hypotheses():
    engine, smoke, _ = _sample_smoke(8)
    result = analyze_agp_unknown_category_root_cause(smoke, engine)
    hypotheses = result["root_cause_hypotheses"]
    assert len(hypotheses) == 4
    assert all(h.startswith(("H1", "H2", "H3", "H4")) for h in hypotheses)


def test_round45_agp_root_cause_includes_evidence_for_each_hypothesis():
    engine, smoke, _ = _sample_smoke(8)
    result = analyze_agp_unknown_category_root_cause(smoke, engine)
    evidence = result["hypothesis_evidence"]
    assert set(evidence) == {"H1", "H2", "H3", "H4"}
    assert all("supported" in row for row in evidence.values())
    assert "thresholds" in evidence["H4"]


def test_round45_agp_root_cause_lists_next_round_options():
    engine, smoke, _ = _sample_smoke(8)
    result = analyze_agp_unknown_category_root_cause(smoke, engine)
    options = result["next_round_options"]
    assert len(options) == 4
    assert options[0].startswith("A:")
    assert result["recommendation_priority"] == "manual_decision"


def test_round45_problem_separation_confirmed():
    result = confirm_problem_separation()
    assert result["problem_1_lexical_coverage"]["blocker"] is False
    assert result["problem_2_agp_anchor"]["blocker"] is True
    assert result["priority"] == "problem_2_first"


def test_round45_problem_2_marked_as_blocker_and_medium_forbidden():
    result = confirm_problem_separation()
    assert result["problem_2_agp_anchor"]["severity"] == "high"
    assert result["medium_30k_forbidden_until_agp_blocker_resolved"] is True
    assert result["recommendation_priority"] == "manual_decision"


def test_round45_analysis_does_not_auto_adjust_thresholds():
    engine, smoke, _ = _sample_smoke(8)
    before = getattr(engine.agp_adapter, "category_threshold", None), getattr(engine.agp_adapter, "hormone_threshold", None)
    result = analyze_agp_unknown_category_root_cause(smoke, engine)
    after = getattr(engine.agp_adapter, "category_threshold", None), getattr(engine.agp_adapter, "hormone_threshold", None)
    assert before == after
    assert result["no_threshold_change"] is True


def test_round45_analysis_does_not_auto_extract_medium_subset():
    _, smoke, fixtures = _sample_smoke(20)
    result = analyze_category_coverage(smoke, fixtures)
    assert result["medium_30k_projected_impact"]["auto_extract_medium_30k"] is False
    separation = confirm_problem_separation()
    assert separation["medium_30k_forbidden_until_agp_blocker_resolved"] is True


def test_round45_analyses_are_read_only_for_smoke_and_engine():
    engine, smoke, fixtures = _sample_smoke(8)
    before_smoke = deepcopy(smoke)
    before_telemetry = engine.self_embedding.telemetry()
    analyze_category_coverage(smoke, fixtures)
    analyze_agp_unknown_category_root_cause(smoke, engine)
    confirm_problem_separation()
    assert smoke == before_smoke
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round45_priority_marked_manual_decision():
    engine, smoke, _ = _sample_smoke(8)
    agp = analyze_agp_unknown_category_root_cause(smoke, engine)
    separation = confirm_problem_separation()
    assert agp["recommendation_priority"] == "manual_decision"
    assert separation["recommendation_priority"] == "manual_decision"
