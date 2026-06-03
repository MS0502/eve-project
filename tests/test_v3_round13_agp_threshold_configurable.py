"""EVE v3 round13 — explicit AGP threshold configuration.

Round13 makes AGP thresholds configurable while preserving observation-only
runtime behavior. Analyzer recommendation data remains advisory and is never
auto-applied.
"""

import pytest

from adapters.agp_adapter import (
    AGP_MODE_OBSERVATION,
    AGPAdapter,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPResult,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.appraisal_classifier import AppraisalClassifier


def _meaning() -> dict:
    return AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()


def _activation_scores(meaning: dict, score: float) -> dict[str, float]:
    return {
        meaning.get("meaning_layer", "appraisal"): score,
        meaning["label"]: score,
        meaning["subject_type"]: score,
        meaning["affect_type"]: score,
    }


def _trace(result: AGPResult, *, score: float = 0.4) -> dict:
    meaning = _meaning()
    return {
        "mode": AGP_MODE_OBSERVATION,
        "candidate_response": "candidate",
        "meaning": {"meaning_layer": "appraisal"},
        "category_activations": _activation_scores(meaning, score),
        "hormone_state": {},
        "result": result,
        "error": None,
    }


def test_default_thresholds_unchanged():
    adapter = AGPAdapter()

    assert adapter.category_threshold == pytest.approx(0.3)
    assert adapter.hormone_threshold == pytest.approx(0.5)
    assert adapter.mode == AGP_MODE_OBSERVATION


def test_set_thresholds_changes_behavior_without_route_or_veto_activation():
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    meaning = _meaning()
    activated = _activation_scores(meaning, 0.35)

    before = adapter.verify(
        "이 노래가 좋다는 평가로 들려.",
        meaning=meaning,
        activated_categories=activated,
        hormone_state={"cortisol": 0.1},
    )
    adapter.set_thresholds(category_threshold=0.4)
    after = adapter.verify(
        "이 노래가 좋다는 평가로 들려.",
        meaning=meaning,
        activated_categories=activated,
        hormone_state={"cortisol": 0.1},
    )

    assert before.passed is True
    assert before.reason == AGP_REASON_ANCHORED
    assert after.passed is False
    assert after.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert adapter.mode == AGP_MODE_OBSERVATION


def test_invalid_threshold_rejected_and_existing_values_preserved():
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)

    with pytest.raises(ValueError):
        AGPAdapter(category_threshold=-0.1)
    with pytest.raises(ValueError):
        AGPAdapter(hormone_threshold=1.5)
    with pytest.raises(ValueError):
        adapter.set_thresholds(category_threshold=0.4, hormone_threshold=2.0)

    assert adapter.category_threshold == pytest.approx(0.3)
    assert adapter.hormone_threshold == pytest.approx(0.5)


def test_analyzer_recommendation_not_auto_applied_to_agp_thresholds():
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(AGPResult(True, AGP_REASON_ANCHORED), score=0.4)
                for _ in range(10)
            ]
        },
        category_threshold=adapter.category_threshold,
        hormone_threshold=adapter.hormone_threshold,
    )

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.2, 0.3, 0.4])

    assert report["recommendation_data"]["status"] == "candidate"
    assert adapter.category_threshold == pytest.approx(0.3)
    assert adapter.hormone_threshold == pytest.approx(0.5)


def test_threshold_change_does_not_enable_veto_mode():
    adapter = AGPAdapter(mode=AGP_MODE_OBSERVATION)

    adapter.set_thresholds(category_threshold=0.9, hormone_threshold=0.9)

    assert adapter.category_threshold == pytest.approx(0.9)
    assert adapter.hormone_threshold == pytest.approx(0.9)
    assert adapter.mode == AGP_MODE_OBSERVATION
