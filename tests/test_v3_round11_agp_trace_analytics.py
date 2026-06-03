"""EVE v3 round11 — AGP trace analytics extension.

Round11 extends the read-only trace helper with distribution methods. It does
not tune thresholds, activate veto, or use fallback at runtime.
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
    layer: str,
    result: AGPResult,
    *,
    hormone_state=None,
    activated_categories=None,
    category_activations=None,
    text: str = "candidate",
) -> dict:
    row = {
        "mode": "observation",
        "candidate_response": text,
        "meaning": {"meaning_layer": layer},
        "activated_categories": list(activated_categories or result.activated_categories),
        "hormone_state": dict(hormone_state or {}),
        "result": result,
        "error": None,
    }
    if category_activations is not None:
        row["category_activations"] = dict(category_activations)
    return row


def test_hormone_distribution_per_reason():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    "composition",
                    AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY),
                    hormone_state={"cortisol": 0.9},
                ),
                _trace(
                    "composition",
                    AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY),
                    hormone_state={"oxytocin": 0.8},
                ),
                _trace(
                    "composition",
                    AGPResult(False, AGP_REASON_HORMONE_MISMATCH),
                    hormone_state={"cortisol": 0.95},
                ),
            ]
        }
    )

    assert analyzer.hormone_signature_distribution(AGP_REASON_UNKNOWN_CATEGORY) == {
        "cortisol_high": 1,
        "oxytocin_high": 1,
    }
    assert analyzer.hormone_signature_distribution(AGP_REASON_HORMONE_MISMATCH) == {
        "cortisol_high": 1,
    }


def test_category_activation_histogram():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    "composition",
                    AGPResult(
                        False,
                        AGP_REASON_UNKNOWN_CATEGORY,
                        candidate_categories=["novel_food", "appraisal"],
                    ),
                    category_activations={"novel_food": 0.0, "appraisal": 0.42},
                ),
                _trace(
                    "speech_hub",
                    AGPResult(
                        False,
                        AGP_REASON_UNKNOWN_CATEGORY,
                        candidate_categories=["warm_reply"],
                    ),
                    category_activations={"warm_reply": 0.83},
                ),
            ]
        }
    )

    assert analyzer.category_activation_distribution() == {
        "0.0-0.1": 1,
        "0.3-0.5": 1,
        "0.8-1.0": 1,
    }


def test_fail_pattern_correlation_by_reason_layer_and_hormone():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    "composition",
                    AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY),
                    hormone_state={"cortisol": 0.8},
                ),
                _trace(
                    "composition",
                    AGPResult(True, AGP_REASON_ANCHORED),
                    hormone_state={"cortisol": 0.8},
                ),
            ],
            "speech_hub": [
                _trace(
                    "speech_hub",
                    AGPResult(False, AGP_REASON_HORMONE_MISMATCH),
                    hormone_state={"oxytocin": 0.9},
                ),
            ],
        }
    )

    assert analyzer.fail_pattern_correlation() == {
        AGP_REASON_UNKNOWN_CATEGORY: {"composition|cortisol_high": 1},
        AGP_REASON_HORMONE_MISMATCH: {"speech_hub|oxytocin_high": 1},
    }


def test_analytics_extension_is_read_only():
    trace = [
        _trace(
            "composition",
            AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, candidate_categories=["x"]),
            hormone_state={"cortisol": 0.8},
            category_activations={"x": 0.0},
            text="fail",
        )
    ]
    before = deepcopy(trace)
    analyzer = AGPTraceAnalyzer(sources={"compositor": trace})

    analyzer.hormone_signature_distribution()
    analyzer.category_activation_distribution()
    analyzer.fail_pattern_correlation()

    assert trace == before


def test_analytics_handles_no_fails():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace(
                    "composition",
                    AGPResult(True, AGP_REASON_ANCHORED, candidate_categories=["ok"]),
                    hormone_state={"cortisol": 0.2},
                    activated_categories=["ok"],
                )
            ]
        }
    )

    assert analyzer.hormone_signature_distribution(AGP_REASON_UNKNOWN_CATEGORY) == {}
    assert analyzer.category_activation_distribution() == {}
    assert analyzer.fail_pattern_correlation() == {}
