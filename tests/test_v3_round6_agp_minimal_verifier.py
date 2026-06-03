"""EVE v3 round6 — AGP minimal verifier tests.

Round6 introduces output-side AGPResult generation in observation-only mode.
It must not be wired into runtime routing, compositor, or speech_hub yet.
"""

from copy import deepcopy

from adapters.agp_adapter import (
    AGPAdapter,
    AGP_REASON_ANCHORED,
    AGP_REASON_HORMONE_MISMATCH,
    AGP_REASON_UNKNOWN_CATEGORY,
)
from adapters.appraisal_classifier import AppraisalClassifier


def _activated_for(meaning: dict) -> list[str]:
    return [
        meaning.get("meaning_layer", "appraisal"),
        meaning["label"],
        meaning["subject_type"],
        meaning["affect_type"],
    ]


def test_agp_verifier_passes_anchored_response_in_observation_mode():
    adapter = AGPAdapter(hormone_threshold=0.5)
    meaning = AppraisalClassifier().analyze("오늘 날씨 좋다").as_meaning_dict()

    result = adapter.verify(
        "오늘 날씨가 좋다는 말로 들려.",
        meaning=meaning,
        activated_categories=_activated_for(meaning),
        hormone_state={"cortisol": 0.1},
    )

    assert result.passed is True
    assert result.reason == AGP_REASON_ANCHORED
    assert result.fallback is None
    assert result.candidate_categories == _activated_for(meaning)
    assert result.hormone_distance == 0.0


def test_agp_verifier_flags_unknown_category_with_fallback_data_only():
    adapter = AGPAdapter()
    meaning = AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()

    result = adapter.verify(
        "이 노래가 좋다는 평가로 들려.",
        meaning=meaning,
        activated_categories=["appraisal"],
        hormone_state={"cortisol": 0.1},
    )

    assert result.passed is False
    assert result.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert result.fallback is not None
    assert "positive_object_appraisal_statement" in result.candidate_categories


def test_agp_verifier_flags_hormone_mismatch_with_fallback_data_only():
    adapter = AGPAdapter(hormone_threshold=0.5)
    meaning = AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()

    result = adapter.verify(
        "괜찮아, 완전 좋아.",
        meaning=meaning,
        activated_categories=_activated_for(meaning),
        hormone_state={"cortisol": 0.9},
    )

    assert result.passed is False
    assert result.reason == AGP_REASON_HORMONE_MISMATCH
    assert result.fallback is not None
    assert result.hormone_distance == 1.0


def test_agp_verifier_has_no_side_effects_on_inputs_or_adapter_state():
    adapter = AGPAdapter(category_threshold=0.3, hormone_threshold=0.5)
    meaning = AppraisalClassifier().analyze("그 코드 마음에 안 들어").as_meaning_dict()
    activated = _activated_for(meaning)
    hormone_state = {"cortisol": 0.2, "oxytocin": 0.4}

    before_meaning = deepcopy(meaning)
    before_activated = list(activated)
    before_hormone = dict(hormone_state)
    before_adapter_state = dict(adapter.__dict__)

    result = adapter.verify(
        "그 코드가 마음에 안 든다는 평가로 들려.",
        meaning=meaning,
        activated_categories=activated,
        hormone_state=hormone_state,
    )

    assert result.passed is True
    assert meaning == before_meaning
    assert activated == before_activated
    assert hormone_state == before_hormone
    assert adapter.__dict__ == before_adapter_state
