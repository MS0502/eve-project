"""EVE v3 round19 — AGP post-decision stabilization audit.

Round19 extends the round16 audit pattern to the threshold decision layer.
Decision data must remain inert until a human/reviewer explicitly calls
AGPAdapter.set_thresholds(...). This test file adds invariants only; no
production logic changes are required for this round.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
    AGPAdapter,
    AGPResult,
)
from adapters.agp_threshold_decision import (
    AGP_TUNING_ACTION_NO_CHANGE,
    AGP_TUNING_ACTION_TUNED,
    build_threshold_tuning_decision,
    validate_threshold_tuning_decision,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.appraisal_classifier import AppraisalClassifier
from main import build_full_engine


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


def _trace_row(passed: bool, *, score: float = 0.4) -> dict:
    activations = _activation_scores(score)
    return {
        "mode": AGP_MODE_OBSERVATION,
        "candidate_response": "candidate",
        "meaning": {"meaning_layer": "appraisal"},
        "category_activations": activations,
        "hormone_state": {},
        "result": AGPResult(
            passed=passed,
            reason=AGP_REASON_ANCHORED if passed else AGP_REASON_UNKNOWN_CATEGORY,
            candidate_categories=list(activations.keys()),
        ),
        "error": None,
    }


def _report(sample_count: int = 12, minimum_samples: int = 10, *, score: float = 0.4) -> dict:
    rows = [_trace_row(True, score=score) for _ in range(sample_count)]
    analyzer = AGPTraceAnalyzer(
        sources={"compositor": rows},
        category_threshold=0.3,
        hormone_threshold=0.5,
    )
    return analyzer.first_pass_report(
        simulated_thresholds=[0.2, 0.3, 0.4],
        minimum_samples=minimum_samples,
    )


def _tuned_decision() -> dict:
    return build_threshold_tuning_decision(
        _report(sample_count=12, minimum_samples=10),
        category_threshold=0.25,
    )


def _no_change_decision() -> dict:
    return build_threshold_tuning_decision(_report(sample_count=3, minimum_samples=10))


def test_tuned_decision_does_not_affect_compositor_or_speech_hub_modes():
    """A tuned decision object must not switch runtime layer modes."""
    engine = build_full_engine()
    before = (
        engine.agp_adapter.mode,
        engine.compositor.agp_mode,
        engine.speech_hub.agp_mode,
        engine.compositor.agp_veto_count,
        engine.speech_hub.agp_veto_count,
    )

    decision = _tuned_decision()

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert (
        engine.agp_adapter.mode,
        engine.compositor.agp_mode,
        engine.speech_hub.agp_mode,
        engine.compositor.agp_veto_count,
        engine.speech_hub.agp_veto_count,
    ) == before
    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION


def test_tuned_decision_does_not_change_thresholds_without_explicit_set():
    """Decision data is inert until AGPAdapter.set_thresholds(...) is called."""
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    before = (adapter.category_threshold, adapter.hormone_threshold, adapter.mode)

    decision = _tuned_decision()

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert (adapter.category_threshold, adapter.hormone_threshold, adapter.mode) == before
    adapter.set_thresholds(**decision["new_thresholds"])
    assert adapter.category_threshold == pytest.approx(0.25)
    assert adapter.hormone_threshold == pytest.approx(0.5)
    assert adapter.mode == AGP_MODE_OBSERVATION


def test_no_change_decision_is_system_neutral():
    """A no_change/collect_more decision must not mutate the engine."""
    engine = build_full_engine()
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "pool": tuple(AGP_FALLBACK_SURFACE_POOL),
    }

    decision = _no_change_decision()

    assert decision["action"] == AGP_TUNING_ACTION_NO_CHANGE
    assert {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "pool": tuple(AGP_FALLBACK_SURFACE_POOL),
    } == before


def test_decision_generation_does_not_modify_fallback_pool_or_semantic_guard():
    """Decision data must not touch fallback phrasing or frozen guard policy."""
    pool_before = tuple(AGP_FALLBACK_SURFACE_POOL)
    guard_path = Path("adapters/orchestrator_adapter.py")
    guard_before = guard_path.read_text(encoding="utf-8")

    decision = _tuned_decision()

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert tuple(AGP_FALLBACK_SURFACE_POOL) == pool_before
    guard_after = guard_path.read_text(encoding="utf-8")
    assert guard_after == guard_before
    assert "EVE v3 ROUND1 FROZEN TEMPORARY GUARDS" in guard_after
    assert "Do not extend them with new noun keyword lists" in guard_after


def test_decision_generation_does_not_grow_or_mutate_runtime_traces():
    """Threshold decisions must not append to compositor or SpeechHub traces."""
    engine = build_full_engine()
    engine.compositor.composition_phrase(
        # Minimal object compatible with CompositorAdapter.composition_phrase.
        type("Composition", (), {
            "inputs": ("a", "b"),
            "concepts": ["c"],
            "method": "rule",
            "confidence": 0.5,
        })()
    )
    engine.speech_hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)
    before_compositor = deepcopy(engine.compositor.agp_trace)
    before_speech = deepcopy(engine.speech_hub.agp_trace)

    decision = _tuned_decision()

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert engine.compositor.agp_trace == before_compositor
    assert engine.speech_hub.agp_trace == before_speech


def test_decision_is_deterministic_for_same_report_and_manual_proposal():
    """Same report + same proposal must produce identical decision data."""
    report = _report(sample_count=12, minimum_samples=10)
    before = deepcopy(report)

    first = build_threshold_tuning_decision(report, category_threshold=0.25)
    second = build_threshold_tuning_decision(report, category_threshold=0.25)

    assert first == second
    assert report == before
    assert validate_threshold_tuning_decision(first) == first


def test_invalid_decision_data_is_rejected_by_validator():
    """Invalid or unsafe decision data must not validate."""
    valid = _tuned_decision()

    bad_action = dict(valid, action="auto_tune")
    with pytest.raises(ValueError):
        validate_threshold_tuning_decision(bad_action)

    bad_auto_apply = dict(valid, auto_applied=True)
    with pytest.raises(ValueError):
        validate_threshold_tuning_decision(bad_auto_apply)

    bad_requires_flag = dict(valid, requires_explicit_set_thresholds=False)
    with pytest.raises(ValueError):
        validate_threshold_tuning_decision(bad_requires_flag)


def test_decision_does_not_change_default_observation_or_double_lock_behavior():
    """Decision creation must not weaken the explicit double-lock invariant."""
    engine = build_full_engine()
    decision = _tuned_decision()

    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    # Layer locks stay observation, so effective traces remain observation.
    engine.compositor.composition_phrase(
        type("Composition", (), {
            "inputs": ("a", "b"),
            "concepts": ["c"],
            "method": "rule",
            "confidence": 0.5,
        })()
    )
    engine.speech_hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    assert decision["action"] == AGP_TUNING_ACTION_TUNED
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION


def test_round19_report_documents_stabilization_audit_contract():
    """Round19 report should preserve the audit trail for future tuning rounds."""
    report_path = Path("ROUND_V3_R19_REPORT.md")
    text = report_path.read_text(encoding="utf-8")

    assert "## Decision Stabilization Audit" in text
    assert "automatic threshold application" in text
    assert "default observation" in text
    assert "fallback pool" in text
    assert "FROZEN semantic guard" in text
    assert "production logic changed: none" in text
