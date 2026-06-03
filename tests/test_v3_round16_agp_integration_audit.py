"""EVE v3 round16 — AGP veto integration audit.

Round16 is an audit-only round after compositor and SpeechHub explicit veto
paths were introduced. It adds integration/invariant tests only: no production
behavior, thresholds, fallback pool, or route logic should change here.
"""

from dataclasses import dataclass
from pathlib import Path

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
    AGPAdapter,
    AGPResult,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.agp_trace_analyzer import AGPTraceAnalyzer
from adapters.compositor_adapter import CompositionResult, CompositorAdapter
from adapters.speech_hub import SpeechHub
from main import build_full_engine


@dataclass
class _FakeEngine:
    agp_adapter: object
    hormone_adapter: object | None = None


class _LayerAwareAGP:
    """Deterministic fake AGP for integration audit tests.

    ``outcomes`` maps meaning_layer values such as ``composition`` or
    ``speech_hub`` to True/False. Missing layers fail closed.
    """

    def __init__(self, outcomes: dict[str, bool], mode: str = AGP_MODE_VETO):
        self.mode = mode
        self.outcomes = dict(outcomes)
        self.calls: list[dict] = []
        self.category_threshold = 0.3
        self.hormone_threshold = 0.5

    def verify(self, **kwargs):
        self.calls.append(dict(kwargs))
        meaning = kwargs.get("meaning") or {}
        layer = meaning.get("meaning_layer") if isinstance(meaning, dict) else "unknown"
        passed = bool(self.outcomes.get(layer, False))
        if passed:
            return AGPResult(
                passed=True,
                reason=AGP_REASON_ANCHORED,
                fallback=None,
                activated_categories=list(kwargs.get("activated_categories") or []),
                candidate_categories=[layer],
            )
        return AGPResult(
            passed=False,
            reason=AGP_REASON_UNKNOWN_CATEGORY,
            fallback="그게 뭐야?",
            activated_categories=list(kwargs.get("activated_categories") or []),
            candidate_categories=[layer or "unknown"],
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


def _neutral_greeting():
    return {"emotional_state": "neutral"}


def test_compositor_veto_speech_hub_observation_are_independent():
    """Compositor veto must not implicitly switch SpeechHub to veto."""
    engine = build_full_engine()
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.compositor.agp_mode = AGP_MODE_VETO
    engine.speech_hub.agp_mode = AGP_MODE_OBSERVATION

    compositor_output = engine.compositor.composition_phrase(_sample_composition())
    speech_output = engine.speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert compositor_output == "어, 그렇네."
    assert getattr(engine.compositor.agp_trace[-1]["result"], "passed", False) is True
    assert engine.compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert speech_output == SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    assert engine.speech_hub.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_veto_count == 0


def test_compositor_observation_speech_hub_veto_are_independent():
    """SpeechHub veto must not implicitly switch Compositor to veto."""
    engine = build_full_engine()
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.compositor.agp_mode = AGP_MODE_OBSERVATION
    engine.speech_hub.agp_mode = AGP_MODE_VETO

    compositor_output = engine.compositor.composition_phrase(_sample_composition())
    speech_output = engine.speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert compositor_output == "어, 그렇네."
    assert engine.compositor.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_veto_count == 0
    assert speech_output["core"] == "어, 안녕."
    assert getattr(engine.speech_hub.agp_trace[-1]["result"], "passed", False) is True
    assert engine.speech_hub.agp_trace[-1]["mode"] == AGP_MODE_VETO


def test_both_veto_pass_keeps_outputs_on_both_layers():
    """Both layers in veto mode still keep candidates when AGP passes."""
    agp = _LayerAwareAGP({"composition": True, "speech_hub": True})
    engine = _FakeEngine(agp_adapter=agp)
    compositor = CompositorAdapter(engine=engine)
    speech_hub = SpeechHub(engine=engine)
    compositor.agp_mode = AGP_MODE_VETO
    speech_hub.agp_mode = AGP_MODE_VETO

    compositor_output = compositor.composition_phrase(_sample_composition())
    speech_output = speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert compositor_output == "어, 그렇네."
    assert speech_output == SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    assert compositor.agp_trace[-1]["result"].passed is True
    assert speech_hub.agp_trace[-1]["result"].passed is True


def test_both_veto_compositor_fallback_is_not_replaced_again_by_speech_hub():
    """Fallback metadata prevents SpeechHub from re-verifying/replacing fallback."""
    agp = _LayerAwareAGP({"composition": False, "speech_hub": False})
    engine = _FakeEngine(agp_adapter=agp)
    compositor = CompositorAdapter(engine=engine)
    speech_hub = SpeechHub(engine=engine)
    compositor.agp_mode = AGP_MODE_VETO
    speech_hub.agp_mode = AGP_MODE_VETO

    compositor_output = compositor.composition_phrase(_sample_composition())
    calls_after_compositor = len(agp.calls)
    already_fallback = {
        "core": compositor_output,
        "sub": [],
        "_agp_fallback_applied": True,
    }
    speech_output = speech_hub._apply_agp_candidate(
        already_fallback,
        "greeting_simple",
        _neutral_greeting(),
        seed=0,
        state="neutral",
    )

    assert compositor_output in AGP_FALLBACK_SURFACE_POOL
    assert speech_output == already_fallback
    assert len(agp.calls) == calls_after_compositor
    assert speech_hub.agp_trace[-1]["fallback_already_applied"] is True


def test_both_veto_speech_hub_fails_only_after_compositor_passes():
    """Compositor pass + SpeechHub fail should replace only at SpeechHub."""
    agp = _LayerAwareAGP({"composition": True, "speech_hub": False})
    engine = _FakeEngine(agp_adapter=agp)
    compositor = CompositorAdapter(engine=engine)
    speech_hub = SpeechHub(engine=engine)
    compositor.agp_mode = AGP_MODE_VETO
    speech_hub.agp_mode = AGP_MODE_VETO

    compositor_output = compositor.composition_phrase(_sample_composition())
    speech_output = speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert compositor_output == "어, 그렇네."
    assert compositor.agp_trace[-1]["result"].passed is True
    assert speech_output["core"] in AGP_FALLBACK_SURFACE_POOL
    assert speech_hub.agp_trace[-1]["result"].passed is False


def test_default_mode_threshold_and_pool_invariants_hold():
    """Round16 audit must not change defaults, thresholds, or fallback pool."""
    engine = build_full_engine()

    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION
    assert engine.agp_adapter.category_threshold == 0.3
    assert engine.agp_adapter.hormone_threshold == 0.5
    assert len(AGP_FALLBACK_SURFACE_POOL) <= 5


def test_trace_records_layer_and_mode_consistently_for_both_layers():
    agp = _LayerAwareAGP({"composition": True, "speech_hub": False})
    engine = _FakeEngine(agp_adapter=agp)
    compositor = CompositorAdapter(engine=engine)
    speech_hub = SpeechHub(engine=engine)
    compositor.agp_mode = AGP_MODE_VETO
    speech_hub.agp_mode = AGP_MODE_VETO

    compositor.composition_phrase(_sample_composition())
    speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert compositor.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert compositor.agp_trace[-1]["meaning"]["meaning_layer"] == "composition"
    assert speech_hub.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert speech_hub.agp_trace[-1]["meaning"]["meaning_layer"] == "speech_hub"

    analyzer = AGPTraceAnalyzer({
        "compositor": compositor.agp_trace,
        "speech_hub": speech_hub.agp_trace,
    })
    layer_summary = analyzer.summarize_by_layer()
    assert layer_summary["composition"]["total"] == 1
    assert layer_summary["speech_hub"]["total"] == 1


def test_semantic_guard_frozen_marker_invariant_remains_present():
    text = Path("adapters/orchestrator_adapter.py").read_text(encoding="utf-8")

    assert "EVE v3 ROUND1 FROZEN TEMPORARY GUARDS" in text
    assert "Do not extend them with new noun keyword lists" in text


def test_agp_verify_edge_cases_are_safe_without_runtime_side_effects():
    agp = AGPAdapter()
    category_threshold = agp.category_threshold
    hormone_threshold = agp.hormone_threshold
    mode = agp.mode

    result = agp.verify("", meaning=None, activated_categories=None, hormone_state=None)

    assert result.passed is False
    assert result.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert agp.category_threshold == category_threshold
    assert agp.hormone_threshold == hormone_threshold
    assert agp.mode == mode


def test_audit_analyzer_from_engine_reads_both_trace_sources_without_mutating():
    engine = build_full_engine()
    engine.compositor.composition_phrase(_sample_composition())
    engine.speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)
    before_compositor = list(engine.compositor.agp_trace)
    before_speech_hub = list(engine.speech_hub.agp_trace)

    analyzer = AGPTraceAnalyzer.from_engine(engine)
    summary = analyzer.summarize_by_layer()
    rate = analyzer.pass_rate()

    assert "composition" in summary
    assert "speech_hub" in summary
    assert 0.0 <= rate <= 1.0
    assert engine.compositor.agp_trace == before_compositor
    assert engine.speech_hub.agp_trace == before_speech_hub
