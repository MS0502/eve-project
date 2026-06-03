"""EVE v3 round15 — explicit SpeechHub AGP veto mode.

Round15 mirrors the round14 compositor veto pattern for SpeechHub only.
Default engine behavior remains observation-only. Veto requires a double lock:
engine.agp_adapter.mode == veto and speech_hub.agp_mode == veto.
"""

from dataclasses import dataclass

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
    AGPResult,
    AGP_REASON_ANCHORED,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.speech_hub import SpeechHub
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


def _neutral_greeting():
    return {"emotional_state": "neutral"}


def test_speech_hub_default_engine_remains_observation_mode():
    engine = build_full_engine()

    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION

    expected = SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    actual = engine.speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert actual == expected
    assert engine.speech_hub.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION


def test_speech_hub_veto_mode_pass_keeps_output():
    agp = _PassAGP()
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))
    hub.agp_mode = AGP_MODE_VETO

    expected = SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    actual = hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert actual == expected
    assert len(agp.calls) == 1
    assert hub.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert hub.agp_veto_count == 1


def test_speech_hub_veto_mode_fail_uses_fallback_surface():
    agp = _FailAGP()
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))
    hub.agp_mode = AGP_MODE_VETO

    actual = hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert actual == {"core": "그게 뭐야?", "sub": []}
    assert actual != SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    assert hub.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert hub.agp_trace[-1]["result"].passed is False


def test_speech_hub_veto_requires_double_lock():
    agp = _FailAGP()
    agp.mode = AGP_MODE_OBSERVATION
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))
    hub.agp_mode = AGP_MODE_VETO

    expected = SpeechHub().generate("greeting_simple", _neutral_greeting(), seed=0)
    actual = hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert actual == expected
    assert hub.agp_trace[-1]["mode"] == AGP_MODE_OBSERVATION
    assert hub.agp_veto_count == 0


def test_speech_hub_does_not_reverify_already_applied_fallback():
    agp = _FailAGP()
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))
    hub.agp_mode = AGP_MODE_VETO

    already_fallback = {"core": "그게 뭐야?", "sub": [], "_agp_fallback_applied": True}
    actual = hub._apply_agp_candidate(
        already_fallback,
        "greeting_simple",
        _neutral_greeting(),
        seed=0,
        state="neutral",
    )

    assert actual == already_fallback
    assert agp.calls == []
    assert hub.agp_trace[-1]["fallback_already_applied"] is True
    assert hub.agp_trace[-1]["mode"] == AGP_MODE_VETO


def test_speech_hub_fallback_pool_still_minimal():
    assert len(AGP_FALLBACK_SURFACE_POOL) <= 5


def test_speech_hub_veto_does_not_change_compositor_mode():
    engine = build_full_engine()
    engine.agp_adapter.set_mode(AGP_MODE_VETO)
    engine.speech_hub.agp_mode = AGP_MODE_VETO

    actual = engine.speech_hub.generate("greeting_simple", _neutral_greeting(), seed=0)

    assert actual["core"] == "어, 안녕."
    assert getattr(engine.speech_hub.agp_trace[-1]["result"], "passed", False) is True
    assert engine.speech_hub.agp_trace[-1]["mode"] == AGP_MODE_VETO
    assert engine.compositor.agp_mode == AGP_MODE_OBSERVATION
