"""EVE v3 round10 — AGP trace/debug helper.

Round10 adds read-only analysis helpers over compositor and speech_hub AGP
observation traces. It does not tune thresholds, activate fallback, or enable
veto.
"""

from copy import deepcopy

from adapters.agp_adapter import (
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPResult,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer


def _trace(layer: str, result: AGPResult, text: str = "candidate") -> dict:
    return {
        "mode": "observation",
        "candidate_response": text,
        "meaning": {"meaning_layer": layer},
        "activated_categories": list(result.activated_categories),
        "hormone_state": {},
        "result": result,
        "error": None,
    }


def test_helper_summarize_by_reason():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace("composition", AGPResult(True, AGP_REASON_ANCHORED)),
                _trace("composition", AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY)),
            ],
            "speech_hub": [
                _trace("speech_hub", AGPResult(False, AGP_REASON_HORMONE_MISMATCH)),
            ],
        }
    )

    assert analyzer.summarize_by_reason() == {
        AGP_REASON_ANCHORED: 1,
        AGP_REASON_UNKNOWN_CATEGORY: 1,
        AGP_REASON_HORMONE_MISMATCH: 1,
    }


def test_helper_summarize_by_layer():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace("composition", AGPResult(True, AGP_REASON_ANCHORED)),
                _trace("composition", AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY)),
            ],
            "speech_hub": [
                _trace("speech_hub", AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY)),
            ],
        }
    )

    by_layer = analyzer.summarize_by_layer()
    assert by_layer["composition"]["total"] == 2
    assert by_layer["composition"]["passed"] == 1
    assert by_layer["composition"]["failed"] == 1
    assert by_layer["speech_hub"]["total"] == 1
    assert by_layer["speech_hub"]["reasons"][AGP_REASON_UNKNOWN_CATEGORY] == 1


def test_helper_pass_rate():
    analyzer = AGPTraceAnalyzer(
        sources={
            "compositor": [
                _trace("composition", AGPResult(True, AGP_REASON_ANCHORED)),
                _trace("composition", AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY)),
                _trace("composition", AGPResult(True, AGP_REASON_ANCHORED)),
            ]
        }
    )

    assert analyzer.pass_rate() == 2 / 3


def test_helper_is_read_only():
    trace = [
        _trace("composition", AGPResult(True, AGP_REASON_ANCHORED), text="ok"),
        _trace("speech_hub", AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY), text="fail"),
    ]
    before = deepcopy(trace)
    analyzer = AGPTraceAnalyzer(sources={"compositor": trace})

    analyzer.summarize_by_reason()
    analyzer.summarize_by_layer()
    analyzer.pass_rate()
    recent = analyzer.recent_fails(1)
    recent[0]["candidate_response"] = "mutated copy"

    assert trace == before


def test_helper_handles_empty_trace():
    analyzer = AGPTraceAnalyzer(sources={"compositor": [], "speech_hub": []})

    assert analyzer.summarize_by_reason() == {}
    assert analyzer.summarize_by_layer() == {}
    assert analyzer.pass_rate() == 0.0
    assert analyzer.recent_fails() == []
