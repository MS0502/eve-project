"""EVE v3 round18 — AGP threshold first-pass tuning decision.

Round18 is the first data-backed decision round. It records explicit decision
workflow data but does not auto-apply analyzer recommendation data, activate
veto by default, expand fallback pools, or change semantic guards.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from adapters.agp_adapter import (
    AGP_MODE_OBSERVATION,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPResult,
    AGPAdapter,
)
from adapters.agp_threshold_decision import (
    AGP_TUNING_ACTION_NO_CHANGE,
    AGP_TUNING_ACTION_TUNED,
    AGP_TUNING_REASON_INSUFFICIENT_DATA,
    AGP_TUNING_REASON_MANUAL_NO_CHANGE,
    AGP_TUNING_REASON_MANUAL_PROPOSAL,
    build_threshold_tuning_decision,
    validate_threshold_tuning_decision,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.appraisal_classifier import AppraisalClassifier


def _meaning() -> dict:
    return AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()


def _activation_scores(score: float) -> dict[str, float]:
    meaning = _meaning()
    return {
        meaning.get("meaning_layer", "appraisal"): score,
        meaning["label"]: score,
        meaning["subject_type"]: score,
        meaning["affect_type"]: score,
    }


def _trace(passed: bool, *, score: float = 0.4) -> dict:
    return {
        "mode": AGP_MODE_OBSERVATION,
        "candidate_response": "candidate",
        "meaning": {"meaning_layer": "appraisal"},
        "category_activations": _activation_scores(score),
        "hormone_state": {},
        "result": AGPResult(
            passed,
            AGP_REASON_ANCHORED if passed else AGP_REASON_UNKNOWN_CATEGORY,
            candidate_categories=list(_activation_scores(score).keys()),
        ),
        "error": None,
    }


def _report(sample_count: int = 12, minimum_samples: int = 10, *, score: float = 0.4) -> dict:
    rows = [_trace(True, score=score) for _ in range(sample_count)]
    analyzer = AGPTraceAnalyzer(
        sources={"compositor": rows},
        category_threshold=0.3,
        hormone_threshold=0.5,
    )
    return analyzer.first_pass_report(
        simulated_thresholds=[0.2, 0.3, 0.4],
        minimum_samples=minimum_samples,
    )


def test_decision_workflow_with_sufficient_data_allows_explicit_tuning():
    """ready_for_tuning + explicit proposal yields data-only tuned decision."""
    report = _report(sample_count=12, minimum_samples=10)

    decision = build_threshold_tuning_decision(report, category_threshold=0.25)

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert decision["reason"] == AGP_TUNING_REASON_MANUAL_PROPOSAL
    assert decision["old_thresholds"] == {
        "category_threshold": 0.3,
        "hormone_threshold": 0.5,
    }
    assert decision["new_thresholds"] == {
        "category_threshold": 0.25,
        "hormone_threshold": 0.5,
    }
    assert decision["auto_applied"] is False
    assert decision["requires_explicit_set_thresholds"] is True
    assert validate_threshold_tuning_decision(decision) == decision


def test_decision_workflow_with_insufficient_data_no_change():
    """collect_more data quality returns no_change and never tunes implicitly."""
    report = _report(sample_count=3, minimum_samples=10)

    decision = build_threshold_tuning_decision(report, category_threshold=0.2)

    assert decision["action"] == AGP_TUNING_ACTION_NO_CHANGE
    assert decision["reason"] == AGP_TUNING_REASON_INSUFFICIENT_DATA
    assert decision["new_thresholds"] == decision["old_thresholds"]
    assert decision["next_step"] == "collect_more"
    assert decision["rationale_data"]["sample_count"] == 3
    assert decision["rationale_data"]["minimum_samples"] == 10


def test_recommendation_does_not_auto_apply_to_agp_adapter():
    """Report/decision generation must not mutate AGP thresholds."""
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    report = _report(sample_count=12, minimum_samples=10)
    before = (adapter.category_threshold, adapter.hormone_threshold, adapter.mode)

    decision = build_threshold_tuning_decision(report, category_threshold=0.2)

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert (adapter.category_threshold, adapter.hormone_threshold, adapter.mode) == before


def test_explicit_set_thresholds_required_before_runtime_change():
    """Even a tuned decision is only data until set_thresholds() is called."""
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    report = _report(sample_count=12, minimum_samples=10)
    decision = build_threshold_tuning_decision(report, category_threshold=0.2)

    assert adapter.category_threshold == pytest.approx(0.3)
    adapter.set_thresholds(**decision["new_thresholds"])

    assert adapter.category_threshold == pytest.approx(0.2)
    assert adapter.hormone_threshold == pytest.approx(0.5)
    assert adapter.mode == AGP_MODE_OBSERVATION


def test_decision_without_explicit_thresholds_is_no_change():
    """Sufficient data alone does not tune thresholds without manual proposal."""
    report = _report(sample_count=12, minimum_samples=10)

    decision = build_threshold_tuning_decision(report)

    assert decision["action"] == AGP_TUNING_ACTION_NO_CHANGE
    assert decision["reason"] == AGP_TUNING_REASON_MANUAL_NO_CHANGE
    assert decision["new_thresholds"] == decision["old_thresholds"]
    assert decision["next_step"] == "manual_review_or_collect_more"


def test_decision_documented_in_round18_report():
    """Round18 report keeps the decision/rationale/validation audit trail."""
    report_path = Path("ROUND_V3_R18_REPORT.md")
    text = report_path.read_text(encoding="utf-8")

    assert "## Decision" in text
    assert "action:" in text
    assert "## Rationale" in text
    assert "sample_size" in text
    assert "recommendation_data" in text
    assert "## Test Verification" in text


def test_decision_generation_is_read_only_for_report_data():
    """Decision data generation must not mutate report dictionaries."""
    report = _report(sample_count=12, minimum_samples=10)
    before = deepcopy(report)

    decision = build_threshold_tuning_decision(report, category_threshold=0.25)

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert report == before
