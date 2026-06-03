"""EVE v3 round9 — AGP speech_hub observation-mode integration.

Round9 wires SpeechHub to call AGP in observation mode only.
The generated output must remain unchanged even when AGP fails.
"""

from dataclasses import dataclass

from adapters.agp_adapter import (
    AGP_MODE_OBSERVATION,
    AGPResult,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.speech_hub import SpeechHub
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
            fallback={"reason": AGP_REASON_UNKNOWN_CATEGORY, "hint": "honest_unknown"},
            activated_categories=list(kwargs.get("activated_categories") or []),
            candidate_categories=["missing"],
        )


def test_speech_hub_default_agp_mode_is_observation():
    hub = SpeechHub()

    assert hub.agp_mode == AGP_MODE_OBSERVATION
    assert hub.agp_observation_count == 0
    assert hub.agp_trace == []


def test_speech_hub_calls_agp_verify_in_observation_mode():
    agp = _FailingAGP()
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))

    result = hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    expected = SpeechHub().generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)
    expected_text = hub._candidate_text(expected)
    assert result == expected
    assert len(agp.calls) == 1
    call = agp.calls[0]
    assert call["candidate_response"] == expected_text
    assert call["meaning"]["meaning_layer"] == "speech_hub"
    assert call["meaning"]["intent"] == "greeting_simple"
    assert "speech_hub" in call["activated_categories"]


def test_agp_result_recorded_in_speech_hub_trace_without_fallback_activation():
    agp = _FailingAGP()
    hub = SpeechHub(engine=_FakeEngine(agp_adapter=agp))

    result = hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    expected = SpeechHub().generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)
    expected_text = hub._candidate_text(expected)
    assert result == expected
    assert hub.agp_observation_count == 1
    assert len(hub.agp_trace) == 1
    trace = hub.agp_trace[0]
    assert trace["mode"] == AGP_MODE_OBSERVATION
    assert trace["candidate_response"] == expected_text
    assert trace["result"].passed is False
    assert trace["result"].fallback == {"reason": AGP_REASON_UNKNOWN_CATEGORY, "hint": "honest_unknown"}


def test_output_unchanged_when_speech_hub_agp_fails_in_observation_mode():
    without_agp = SpeechHub().generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    agp = _FailingAGP()
    with_agp = SpeechHub(engine=_FakeEngine(agp_adapter=agp)).generate(
        "greeting_simple", {"emotional_state": "neutral"}, seed=0
    )

    assert without_agp == with_agp
    assert agp.calls[0]["candidate_response"] == SpeechHub()._candidate_text(with_agp)


def test_build_full_engine_wires_agp_for_speech_hub_observation_only():
    engine = build_full_engine()

    assert hasattr(engine, "agp_adapter")
    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
    assert engine.speech_hub.agp_mode == AGP_MODE_OBSERVATION

    result = engine.speech_hub.generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)

    expected = SpeechHub().generate("greeting_simple", {"emotional_state": "neutral"}, seed=0)
    expected_text = engine.speech_hub._candidate_text(expected)
    assert result == expected
    assert engine.speech_hub.agp_observation_count == 1
    assert engine.speech_hub.agp_trace[-1]["candidate_response"] == expected_text
