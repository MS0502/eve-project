"""EVE v3 round12 — AGP threshold analysis helper.

Round12 is advisory-only. It produces deterministic data reports for later
human-reviewed threshold work, but it does not tune AGP thresholds, activate
veto, or use fallback at runtime.
"""

from copy import deepcopy

from adapters.agp_adapter import (
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPResult,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer


def _trace(
    result: AGPResult,
    *,
    layer: str = "composition",
    category_activations=None,
    hormone_state=None,
    text: str = "candidate",
) -> dict:
    row = {
        "mode": "observation",
        "candidate_response": text,
        "meaning": {"meaning_layer": layer},
        "activated_categories": list(result.activated_categories),
        "hormone_state": dict(hormone_state or {}),
        "result": result,
        "error": None,
    }
    if category_activations is not None:
        row["category_activations"] = dict(category_activations)
    return row


def test_threshold_simulation_accuracy():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    AGPResult(True, AGP_REASON_ANCHORED, candidate_categories=["a"]),
                    category_activations={"a": 0.4},
                ),
                _trace(
                    AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, candidate_categories=["b"]),
                    category_activations={"b": 0.1},
                ),
                _trace(
                    AGPResult(False, AGP_REASON_HORMONE_MISMATCH, candidate_categories=["c"]),
                    category_activations={"c": 0.9},
                    hormone_state={"cortisol": 0.9},
                ),
            ]
        }
    )

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.3, 0.5])

    assert report["simulated"]["category_0.3"] == {
        "threshold": 0.3,
        "total": 3,
        "passed": 1,
        "failed": 2,
        "pass_rate": 1 / 3,
        "fail_count": {
            AGP_REASON_UNKNOWN_CATEGORY: 1,
            AGP_REASON_HORMONE_MISMATCH: 1,
        },
    }
    assert report["simulated"]["category_0.5"] == {
        "threshold": 0.5,
        "total": 3,
        "passed": 0,
        "failed": 3,
        "pass_rate": 0.0,
        "fail_count": {
            AGP_REASON_UNKNOWN_CATEGORY: 2,
            AGP_REASON_HORMONE_MISMATCH: 1,
        },
    }


def test_report_includes_current_baseline():
    analyzer = AGPTraceAnalyzer(
        sources={"compositor": []},
        category_threshold=0.35,
        hormone_threshold=0.65,
    )

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.3])

    assert report["current_thresholds"] == {
        "category_threshold": 0.35,
        "hormone_threshold": 0.65,
    }
    assert report["current_pass_rate"] == 0.0
    assert report["sample_count"] == 0


def test_recommendation_data_is_data_not_text():
    rows = []
    for index in range(10):
        rows.append(
            _trace(
                AGPResult(True, AGP_REASON_ANCHORED, candidate_categories=[f"cat_{index}"]),
                category_activations={f"cat_{index}": 0.4},
            )
        )
    analyzer = AGPTraceAnalyzer(sources={"compositor": rows})

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.2, 0.3])
    recommendation = report["recommendation_data"]

    assert isinstance(recommendation, dict)
    assert recommendation["status"] == "candidate"
    assert isinstance(recommendation["rationale_data"], dict)
    assert "category_threshold" in recommendation
    assert "recommendation_text" not in recommendation
    assert "message" not in recommendation
    assert "natural_language" not in recommendation


def test_threshold_analysis_is_advisory_only_and_read_only():
    trace = [
        _trace(
            AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, candidate_categories=["x"]),
            category_activations={"x": 0.0},
            text="fail",
        )
    ]
    before = deepcopy(trace)
    analyzer = AGPTraceAnalyzer(
        sources={"compositor": trace},
        category_threshold=0.3,
        hormone_threshold=0.5,
    )

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.2, 0.3])

    assert trace == before
    assert analyzer.category_threshold == 0.3
    assert analyzer.hormone_threshold == 0.5
    assert report["current_thresholds"] == {
        "category_threshold": 0.3,
        "hormone_threshold": 0.5,
    }


def test_threshold_analysis_handles_insufficient_data():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, candidate_categories=["x"]),
                    category_activations={"x": 0.0},
                )
            ]
        }
    )

    report = analyzer.threshold_analysis_report(simulated_thresholds=[0.2, 0.3])

    assert report["recommendation_data"] == {
        "status": "insufficient_data",
        "category_threshold": None,
        "sample_count": 1,
        "minimum_samples": 10,
        "rationale_data": {
            "simulated_thresholds": ["category_0.2", "category_0.3"],
            "current_pass_rate": 0.0,
        },
    }
