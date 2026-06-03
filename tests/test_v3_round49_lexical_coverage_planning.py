from copy import deepcopy

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke_with_agp_trace
from adapters.smoke_data_analyzer import (
    ROUND49_ANALYSIS_ROUND,
    ROUND49_ANALYSIS_VERSION,
    classify_oov_by_origin,
    compare_lexical_coverage_strategies,
    project_medium_30k_impact,
    project_self_learning_impact,
)
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def _smoke(limit=8):
    engine = build_full_engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:limit])
    return engine, result


def test_round49_project_medium_30k_returns_data():
    _, smoke = _smoke(8)
    projection = project_medium_30k_impact(smoke, smoke)
    assert projection["analysis_version"] == ROUND49_ANALYSIS_VERSION
    assert projection["analysis_round"] == ROUND49_ANALYSIS_ROUND
    assert projection["strategy"] == "A_medium_30k_extraction"
    assert projection["projected_subset"] == "cc.ko.300.subset.medium.30k"
    assert projection["auto_extract_medium_30k"] is False


def test_round49_project_medium_30k_specifies_eve_specific_partial():
    oov = {"oov_samples_from_run": [{"word": "EVE"}, {"word": "어때"}, {"word": "그래"}]}
    projection = project_medium_30k_impact({}, oov)
    assert "어때" in projection["oov_resolvable_by_30k"]
    assert "EVE" in projection["oov_unresolvable_by_30k"]
    assert projection["alignment_with_appendix_d_drift"].startswith("low")


def test_round49_project_self_learning_returns_data():
    _, smoke = _smoke(5)
    projection = project_self_learning_impact(smoke)
    assert projection["analysis_version"] == ROUND49_ANALYSIS_VERSION
    assert projection["strategy"] == "B_continuous_self_learning"
    assert projection["auto_implement_self_learning"] is False
    assert projection["read_only"] is True


def test_round49_project_self_learning_specifies_minsok_strong():
    projection = project_self_learning_impact({})
    assert projection["minsok_specific_strength"] == "high"
    assert projection["eve_specific_strength"] == "high"
    assert projection["alignment_with_appendix_d_drift"].startswith("high")


def test_round49_compare_strategies_lists_three_and_c_confirmed():
    comparison = compare_lexical_coverage_strategies()
    assert "strategy_A_medium_30k" in comparison
    assert "strategy_B_self_learning" in comparison
    assert "strategy_C_hybrid" in comparison
    assert comparison["selected_strategy"] == "C_hybrid"
    assert comparison["strategy_C_hybrid"]["selected_by_minsok"] is True


def test_round49_compare_strategies_manual_decision():
    comparison = compare_lexical_coverage_strategies()
    assert comparison["manual_decision"] is True
    assert comparison["data_dict_no_recommendation"] is True
    assert comparison["auto_apply"] is False


def test_round49_oov_classification_separates_general_and_eve_specific():
    classified = classify_oov_by_origin(["어때", "그래", "뭐야", "EVE", "민석", "코딩"])
    assert {"어때", "그래", "뭐야", "코딩"}.issubset(set(classified["general_korean"]))
    assert {"EVE", "민석"}.issubset(set(classified["eve_specific"]))
    assert classified["read_only"] is True


def test_round49_projection_is_read_only():
    engine, smoke = _smoke(6)
    before_thresholds = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    before_telemetry = engine.self_embedding.telemetry()
    snapshot = deepcopy(smoke)
    classified = classify_oov_by_origin(smoke)
    project_medium_30k_impact(smoke, classified)
    project_self_learning_impact(smoke)
    compare_lexical_coverage_strategies()
    assert before_thresholds == (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    assert before_telemetry == engine.self_embedding.telemetry()
    assert smoke == snapshot


def test_round49_no_medium_30k_extraction():
    comparison = compare_lexical_coverage_strategies()
    medium = comparison["strategy_A_medium_30k"]
    assert medium["auto_extract_medium_30k"] is False
    assert comparison["no_medium_30k_extraction_this_round"] is True


def test_round49_no_self_learning_mechanism_implementation():
    comparison = compare_lexical_coverage_strategies()
    self_learning = comparison["strategy_B_self_learning"]
    assert self_learning["auto_implement_self_learning"] is False
    assert comparison["no_self_learning_implementation_this_round"] is True


def test_round49_appendix_d_drift_alignment_noted():
    comparison = compare_lexical_coverage_strategies()
    assert comparison["strategy_A_medium_30k"]["alignment_with_appendix_d_drift"].startswith("low")
    assert comparison["strategy_B_self_learning"]["alignment_with_appendix_d_drift"].startswith("high")


def test_round49_c_sequence_records_a_first_b_continuous():
    comparison = compare_lexical_coverage_strategies()
    assert comparison["selected_sequence"] == "A_first_medium_30k_then_B_continuous_self_learning"
    assert comparison["strategy_C_hybrid"]["round50_next"] == "medium_30k_extraction_operator_task"
    assert comparison["strategy_C_hybrid"]["round51_plus_next"] == "self_learning_mechanism_start"
