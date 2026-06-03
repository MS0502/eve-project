from copy import deepcopy

from main import build_full_engine
from adapters.agp_adapter import AGPAdapter, AGP_REASON_UNKNOWN_CATEGORY
from adapters.runtime_smoke_runner import (
    SMOKE_TRACE_BASELINE_ROUND,
    run_conversation_smoke_with_agp_trace,
)
from adapters.smoke_data_analyzer import (
    ROUND46_ANALYSIS_ROUND,
    ROUND46_ANALYSIS_VERSION,
    analyze_extraction_traces,
)
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def _appraisal_meaning():
    return {
        "meaning_layer": "appraisal",
        "label": "emotional_share",
        "subject_type": "self",
        "affect_type": "positive",
        "route_override": None,
        "via": "test",
    }


def _trace_smoke(limit=6):
    engine = build_full_engine()
    result = run_conversation_smoke_with_agp_trace(engine, fixture_texts()[:limit])
    return engine, result


def test_round46_verify_with_trace_returns_dict():
    adapter = AGPAdapter()
    result = adapter.verify_with_trace(
        "기분 좋아",
        meaning=_appraisal_meaning(),
        activated_categories={"appraisal": 0.8, "emotional_share": 0.8, "self": 0.8, "positive": 0.8},
        hormone_state={"cortisol": 0.1},
    )
    assert isinstance(result, dict)
    assert "result" in result
    assert "trace" in result
    assert result["read_only"] is True


def test_round46_trace_includes_candidate_categories():
    adapter = AGPAdapter()
    result = adapter.verify_with_trace("기분 좋아", meaning=_appraisal_meaning(), activated_categories=[])
    trace = result["trace"]
    assert "candidate_categories_extracted" in trace
    assert trace["candidate_count"] == 4
    assert {"appraisal", "emotional_share", "self", "positive"}.issubset(set(trace["candidate_categories_extracted"]))


def test_round46_trace_includes_sa_active():
    adapter = AGPAdapter(category_threshold=0.3)
    result = adapter.verify_with_trace(
        "기분 좋아",
        meaning=_appraisal_meaning(),
        activated_categories={"appraisal": 0.7, "weak": 0.1},
    )
    trace = result["trace"]
    assert "sa_activated_categories" in trace
    assert "appraisal" in trace["sa_activated_categories"]
    assert "weak" not in trace["sa_activated_categories"]


def test_round46_trace_includes_overlap():
    adapter = AGPAdapter(category_threshold=0.3)
    result = adapter.verify_with_trace(
        "기분 좋아",
        meaning=_appraisal_meaning(),
        activated_categories={"appraisal": 0.7, "self": 0.8},
    )
    trace = result["trace"]
    assert "overlap" in trace
    assert set(trace["overlap"]) == {"appraisal", "self"}
    assert trace["overlap_count"] == 2


def test_round46_trace_includes_thresholds():
    adapter = AGPAdapter(category_threshold=0.25, hormone_threshold=0.75)
    result = adapter.verify_with_trace("x", meaning=_appraisal_meaning(), activated_categories=[])
    thresholds = result["trace"]["thresholds_used"]
    assert thresholds["category_threshold"] == 0.25
    assert thresholds["hormone_threshold"] == 0.75


def test_round46_decision_matches_regular_verify():
    adapter = AGPAdapter()
    kwargs = {
        "candidate_response": "기분 좋아",
        "meaning": _appraisal_meaning(),
        "activated_categories": {"appraisal": 0.8},
        "hormone_state": {"cortisol": 0.1},
    }
    traced = adapter.verify_with_trace(**kwargs)["result"]
    regular = adapter.verify(**kwargs)
    assert traced == regular
    assert traced.reason == regular.reason


def test_round46_trace_is_read_only():
    engine = build_full_engine()
    adapter = engine.agp_adapter
    before_thresholds = (adapter.category_threshold, adapter.hormone_threshold)
    before_telemetry = engine.self_embedding.telemetry()
    adapter.verify_with_trace("x", meaning=None, activated_categories=None, engine=engine)
    after_thresholds = (adapter.category_threshold, adapter.hormone_threshold)
    after_telemetry = engine.self_embedding.telemetry()
    assert before_thresholds == after_thresholds
    assert before_telemetry == after_telemetry


def test_round46_smoke_with_trace_runs_all_fixtures():
    _, result = _trace_smoke(5)
    assert result["baseline_round"] == SMOKE_TRACE_BASELINE_ROUND
    assert result["fixtures_count"] == 5
    assert len(result["responses"]) == 5
    assert result["read_only"] is True


def test_round46_smoke_trace_data_per_fixture():
    _, result = _trace_smoke(5)
    assert result["traces"]
    for row in result["traces"]:
        assert {"input", "response", "layer", "agp_trace"}.issubset(row)
        trace = row["agp_trace"]["trace"]
        assert "candidate_categories_extracted" in trace
        assert "sa_activated_categories" in trace
        assert "overlap" in trace


def test_round46_hypothesis_evidence_analyzer_returns_data():
    _, result = _trace_smoke(5)
    analysis = analyze_extraction_traces(result["traces"])
    assert isinstance(analysis, dict)
    assert analysis["analysis_version"] == ROUND46_ANALYSIS_VERSION
    assert analysis["analysis_round"] == ROUND46_ANALYSIS_ROUND
    assert analysis["read_only"] is True


def test_round46_hypothesis_evidence_four_hypotheses():
    _, result = _trace_smoke(5)
    analysis = analyze_extraction_traces(result["traces"])
    for key in ["H1_evidence", "H2_evidence", "H3_evidence", "H4_evidence"]:
        assert key in analysis
        assert "hypothesis" in analysis[key]
        assert "strength" in analysis[key]


def test_round46_evidence_strength_calculated():
    traces = [
        {"agp_trace": {"trace": {"candidate_count": 0, "sa_count": 2, "overlap_count": 0, "weak_overlap_count": 0}}},
        {"agp_trace": {"trace": {"candidate_count": 0, "sa_count": 1, "overlap_count": 0, "weak_overlap_count": 0}}},
    ]
    analysis = analyze_extraction_traces(traces)
    assert analysis["H1_evidence"]["strength"] == "strong"
    assert analysis["most_likely_root_cause"] in {"H1", "H3"}
    assert analysis["next_round_recommendation"] == "manual_decision"


def test_round46_no_threshold_changes():
    engine, result = _trace_smoke(4)
    before = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    analyze_extraction_traces(result["traces"])
    after = (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold)
    assert before == after


def test_round46_no_runtime_changes_from_analysis():
    engine, result = _trace_smoke(4)
    before_trace = deepcopy(result["traces"])
    before_telemetry = engine.self_embedding.telemetry()
    analysis = analyze_extraction_traces(result["traces"])
    assert result["traces"] == before_trace
    assert engine.self_embedding.telemetry() == before_telemetry
    assert analysis["no_agp_runtime_change"] is True
    assert analysis["no_auto_application"] is True
