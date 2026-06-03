"""EVE v3 round17 — AGP trace data first-pass analysis.

Round17 applies the read-only analyzer helpers to trace data and returns a
stable data report. It does not tune thresholds, activate veto, expand fallback
surfaces, or generate natural-language recommendations.
"""

from copy import deepcopy

from adapters.agp_adapter import (
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPResult,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.compositor_adapter import CompositionResult
from main import build_full_engine


def _trace(
    result: AGPResult,
    *,
    source_layer: str = "composition",
    category_activations=None,
    hormone_state=None,
    text: str = "candidate",
) -> dict:
    row = {
        "mode": "observation",
        "candidate_response": text,
        "meaning": {"meaning_layer": source_layer},
        "activated_categories": list(result.activated_categories),
        "hormone_state": dict(hormone_state or {}),
        "result": result,
        "error": None,
    }
    if category_activations is not None:
        row["category_activations"] = dict(category_activations)
    return row


def _sample_composition() -> CompositionResult:
    return CompositionResult(
        inputs=("alpha", "beta"),
        concepts=["combined"],
        method="rule",
        confidence=0.5,
    )


def test_first_pass_report_with_synthetic_trace():
    """Deterministic synthetic trace produces expected report fields."""
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    AGPResult(
                        True,
                        AGP_REASON_ANCHORED,
                        candidate_categories=["a"],
                    ),
                    category_activations={"a": 0.6},
                    hormone_state={"cortisol": 0.2},
                ),
                _trace(
                    AGPResult(
                        False,
                        AGP_REASON_UNKNOWN_CATEGORY,
                        candidate_categories=["b"],
                    ),
                    category_activations={"b": 0.0},
                    hormone_state={"cortisol": 0.8},
                ),
            ],
            "speech_hub": [
                _trace(
                    AGPResult(
                        False,
                        AGP_REASON_HORMONE_MISMATCH,
                        candidate_categories=["c"],
                    ),
                    source_layer="speech_hub",
                    category_activations={"c": 0.9},
                    hormone_state={"cortisol": 0.9},
                )
            ],
        }
    )

    report = analyzer.first_pass_report(simulated_thresholds=[0.3], minimum_samples=3)

    assert report["summary"] == {
        "total_observations": 3,
        "pass_rate": 1 / 3,
        "fail_counts_by_reason": {
            AGP_REASON_UNKNOWN_CATEGORY: 1,
            AGP_REASON_HORMONE_MISMATCH: 1,
        },
        "reason_counts": {
            AGP_REASON_ANCHORED: 1,
            AGP_REASON_UNKNOWN_CATEGORY: 1,
            AGP_REASON_HORMONE_MISMATCH: 1,
        },
    }
    assert report["by_layer"]["composition"]["total"] == 2
    assert report["by_layer"]["speech_hub"]["total"] == 1
    assert report["fail_patterns"]["hormone_signature_distribution"] == {
        "low_arousal": 1,
        "cortisol_high": 2,
    }
    assert report["fail_patterns"]["category_activation_distribution"] == {
        "0.0-0.1": 1,
    }
    assert report["threshold_analysis"]["sample_count"] == 3
    assert report["data_quality"] == {
        "sample_count": 3,
        "minimum_samples": 3,
        "sample_size_sufficient": True,
        "recommendation": "ready_for_tuning",
    }


def test_first_pass_report_with_empty_trace():
    """Empty traces return safe zero/collect-more data."""
    analyzer = AGPTraceAnalyzer(sources={"compositor": [], "speech_hub": []})

    report = analyzer.first_pass_report(minimum_samples=1)

    assert report["summary"] == {
        "total_observations": 0,
        "pass_rate": 0.0,
        "fail_counts_by_reason": {},
        "reason_counts": {},
    }
    assert report["by_layer"] == {}
    assert report["fail_patterns"] == {
        "hormone_signature_distribution": {},
        "category_activation_distribution": {},
        "fail_pattern_correlation": {},
    }
    assert report["data_quality"] == {
        "sample_count": 0,
        "minimum_samples": 1,
        "sample_size_sufficient": False,
        "recommendation": "collect_more",
    }


def test_first_pass_report_with_runtime_simulated_trace():
    """A tiny deterministic runtime simulation yields compositor and SpeechHub traces."""
    engine = build_full_engine()

    for _ in range(2):
        engine.compositor.composition_phrase(_sample_composition())
        engine.speech_hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    analyzer = AGPTraceAnalyzer.from_engine(engine)
    report = analyzer.first_pass_report(minimum_samples=1)

    assert report["summary"]["total_observations"] >= 4
    assert "composition" in report["by_layer"]
    assert "speech_hub" in report["by_layer"]
    assert report["data_quality"]["sample_size_sufficient"] is True
    assert report["threshold_analysis"]["current_thresholds"] == {
        "category_threshold": engine.agp_adapter.category_threshold,
        "hormone_threshold": engine.agp_adapter.hormone_threshold,
    }


def test_first_pass_report_is_data_dict_not_text():
    """Round17 report stays data-only and avoids natural-language recommendations."""
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

    report = analyzer.first_pass_report(minimum_samples=10)

    assert isinstance(report, dict)
    assert isinstance(report["data_quality"], dict)
    assert isinstance(report["threshold_analysis"]["recommendation_data"], dict)
    assert "recommendation_text" not in report
    assert "message" not in report
    assert "natural_language" not in report
    assert "recommendation_text" not in report["threshold_analysis"]["recommendation_data"]
    assert "message" not in report["threshold_analysis"]["recommendation_data"]
    assert "natural_language" not in report["threshold_analysis"]["recommendation_data"]


def test_first_pass_report_generation_is_read_only():
    """Report generation must not mutate trace rows or analyzer thresholds."""
    trace = [
        _trace(
            AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, candidate_categories=["x"]),
            category_activations={"x": 0.0},
            hormone_state={"cortisol": 0.8},
        )
    ]
    before = deepcopy(trace)
    analyzer = AGPTraceAnalyzer(
        sources={"compositor": trace},
        category_threshold=0.35,
        hormone_threshold=0.55,
    )

    report = analyzer.first_pass_report(simulated_thresholds=[0.2, 0.3])

    assert trace == before
    assert analyzer.category_threshold == 0.35
    assert analyzer.hormone_threshold == 0.55
    assert report["threshold_analysis"]["current_thresholds"] == {
        "category_threshold": 0.35,
        "hormone_threshold": 0.55,
    }


def test_first_pass_data_quality_flag_collect_more():
    """Sample-size gate must recommend collect_more before tuning."""
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    AGPResult(True, AGP_REASON_ANCHORED, candidate_categories=["a"]),
                    category_activations={"a": 0.9},
                )
            ]
        }
    )

    report = analyzer.first_pass_report(minimum_samples=2)

    assert report["data_quality"] == {
        "sample_count": 1,
        "minimum_samples": 2,
        "sample_size_sufficient": False,
        "recommendation": "collect_more",
    }
    assert report["threshold_analysis"]["recommendation_data"]["status"] == "insufficient_data"
