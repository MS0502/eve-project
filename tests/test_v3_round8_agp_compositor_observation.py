"""EVE v3 round8 — AGP compositor observation-mode integration.

Round8 wires CompositorAdapter to call AGP in observation mode only.
The candidate output must remain unchanged even when AGP fails.
"""

from dataclasses import dataclass

from adapters.agp_adapter import (
    AGP_MODE_OBSERVATION,
    AGPAdapter,
    AGPResult,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.compositor_adapter import CompositionResult, CompositorAdapter
from main import build_full_engine


@dataclass
class _FakeEngine:
    agp_adapter: object
    hormone_adapter: object | None = None


class _FailingAGP:
    mode = AGP_MODE_OBSERVATION

    def __init__(self):
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(dict(kwargs))
        return AGPResult(
            passed=False,
            reason=AGP_REASON_UNKNOWN_CATEGORY,
            fallback="그게 뭐야?",
            activated_categories=list(kwargs.get("activated_categories") or []),
            candidate_categories=["missing"],
        )


def _sample_composition() -> CompositionResult:
    return CompositionResult(
        inputs=("alpha", "beta"),
        concepts=["combined"],
        method="rule",
        confidence=0.5,
    )


def test_agp_mode_default_is_observation():
    assert AGPAdapter().mode == AGP_MODE_OBSERVATION

    compositor = CompositorAdapter()
    assert compositor.agp_mode == AGP_MODE_OBSERVATION


def test_compositor_calls_agp_verify_in_observation_mode():
    agp = _FailingAGP()
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=agp))

    phrase = compositor.composition_phrase(_sample_composition())

    assert phrase == "어, 그렇네."
    assert len(agp.calls) == 1
    assert agp.calls[0]["candidate_response"] == phrase
    assert agp.calls[0]["meaning"]["meaning_layer"] == "composition"


def test_agp_result_recorded_in_trace_without_fallback_activation():
    agp = _FailingAGP()
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=agp))

    phrase = compositor.composition_phrase(_sample_composition())

    assert phrase == "어, 그렇네."
    assert compositor.agp_observation_count == 1
    assert len(compositor.agp_trace) == 1
    trace = compositor.agp_trace[0]
    assert trace["mode"] == AGP_MODE_OBSERVATION
    assert trace["candidate_response"] == phrase
    assert trace["result"].passed is False
    assert trace["result"].fallback == "그게 뭐야?"


def test_output_unchanged_when_agp_fails_in_observation_mode():
    result = _sample_composition()
    without_agp = CompositorAdapter().composition_phrase(result)

    agp = _FailingAGP()
    with_agp = CompositorAdapter(engine=_FakeEngine(agp_adapter=agp)).composition_phrase(result)

    assert without_agp == with_agp == "어, 그렇네."
    assert agp.calls[0]["candidate_response"] == with_agp


def test_build_full_engine_wires_agp_for_compositor_observation_only():
    engine = build_full_engine()

    assert hasattr(engine, "agp_adapter")
    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION

    result = _sample_composition()
    phrase = engine.compositor.composition_phrase(result)

    assert phrase == "어, 그렇네."
    assert engine.compositor.agp_observation_count == 1
    assert engine.compositor.agp_trace[-1]["candidate_response"] == phrase
