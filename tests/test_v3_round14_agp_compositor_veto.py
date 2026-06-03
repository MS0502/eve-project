"""EVE v3 round14 — explicit compositor AGP veto mode.

Round14 is the first AGP round that can replace a compositor candidate, but
only when veto mode is explicitly enabled. The default engine remains
observation-only and SpeechHub is untouched.
"""

from dataclasses import dataclass

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
    AGPAdapter,
    AGPResult,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.compositor_adapter import CompositionResult, CompositorAdapter
from main import build_full_engine


@dataclass
class _FakeEngine:
    agp_adapter: object
    hormone_adapter: object | None = None


class _PassAGP:
    mode = AGP_MODE_VETO

    def __init__(self):
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(dict(kwargs))
        return AGPResult(
            passed=True,
            reason=AGP_REASON_ANCHORED,
            fallback=None,
            activated_categories=list(kwargs.get("activated_categories") or []),
            candidate_categories=list(kwargs.get("activated_categories") or []),
        )

    def fallback_to_surface(self, result):
        return result.fallback


class _FailAGP:
    mode = AGP_MODE_VETO

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

    def fallback_to_surface(self, result):
        return result.fallback


def _sample_composition() -> CompositionResult:
    return CompositionResult(
        inputs=("alpha", "beta"),
        concepts=["combined"],
        method="rule",
        confidence=0.5,
    )


def test_default_engine_and_compositor_remain_observation_mode():
    engine = build_full_engine()

    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION


def test_observation_mode_keeps_existing_compositor_output_even_on_fail():
    fail_agp = _FailAGP()
    fail_agp.mode = AGP_MODE_OBSERVATION
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=fail_agp))

    output = compositor.composition_phrase(_sample_composition())

    assert output == "어, 그렇네."
    assert compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert compositor.agp_observation_count == 1
    assert compositor.agp_veto_count == 0


def test_veto_mode_pass_keeps_candidate_output():
    pass_agp = _PassAGP()
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=pass_agp))
    compositor.agp_mode = AGP_MODE_VETO

    output = compositor.composition_phrase(_sample_composition())

    assert output == "어, 그렇네."
    assert len(pass_agp.calls) == 1
    assert compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert compositor.agp_veto_count == 1


def test_veto_mode_fail_replaces_candidate_with_fallback_surface():
    fail_agp = _FailAGP()
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=fail_agp))
    compositor.agp_mode = AGP_MODE_VETO

    output = compositor.composition_phrase(_sample_composition())

    assert output == "그게 뭐야?"
    assert output != "어, 그렇네."
    assert compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert compositor.agp_trace[-1]["result"].passed is False


def test_veto_requires_explicit_mode_switch_on_adapter_and_compositor():
    agp = _FailAGP()
    agp.mode = AGP_MODE_OBSERVATION
    compositor = CompositorAdapter(engine=_FakeEngine(agp_adapter=agp))
    compositor.agp_mode = AGP_MODE_VETO

    output = compositor.composition_phrase(_sample_composition())

    assert output == "어, 그렇네."
    assert compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert compositor.agp_veto_count == 0


def test_fallback_surface_pool_stays_minimal_and_raw_text_independent():
    adapter = AGPAdapter(mode=AGP_MODE_VETO)
    assert len(AGP_FALLBACK_SURFACE_POOL) <= 5

    first = AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, fallback="그게 뭐야?")
    second = AGPResult(False, AGP_REASON_UNKNOWN_CATEGORY, fallback="그게 뭐야?")

    assert adapter.fallback_to_surface(first) == adapter.fallback_to_surface(second)
    assert adapter.fallback_to_surface(first) in AGP_FALLBACK_SURFACE_POOL


def test_trace_records_veto_mode_without_touching_speech_hub_mode():
    engine = build_full_engine()
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.compositor.agp_mode = AGP_MODE_VETO

    output = engine.compositor.composition_phrase(_sample_composition())

    assert output == "어, 그렇네."
    assert getattr(engine.compositor.agp_trace[-1]["result"], "passed", False) is True
    assert engine.compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION
