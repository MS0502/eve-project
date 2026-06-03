"""EVE v3 round7 — AGP fallback generation tests.

Round7 fills AGPResult.fallback for failed verification results while keeping
AGP observation-only. Runtime routes, compositor, and speech_hub are untouched.
"""

from copy import deepcopy

from adapters.agp_adapter import (
    AGPAdapter,
    AGP_FALLBACK_HORMONE_MISMATCH_HIGH_CORTISOL,
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


def test_agp_unknown_category_gets_honest_fallback_data_only():
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
    assert "뭐" in result.fallback or "모르" in result.fallback


def test_agp_hormone_mismatch_gets_short_high_cortisol_fallback_data_only():
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
    assert result.fallback == AGP_FALLBACK_HORMONE_MISMATCH_HIGH_CORTISOL
    assert len(result.fallback) <= 5


def test_agp_fallback_generation_is_deterministic_for_same_reason_and_hormone_state():
    adapter = AGPAdapter()
    hormone_state = {"cortisol": 0.9, "oxytocin": 0.1}

    first = adapter._generate_fallback(AGP_REASON_UNKNOWN_CATEGORY, hormone_state)
    second = adapter._generate_fallback(AGP_REASON_UNKNOWN_CATEGORY, dict(hormone_state))

    assert first == second


def test_agp_fallback_generation_has_no_side_effects():
    adapter = AGPAdapter(hormone_threshold=0.5)
    meaning = AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()
    activated = _activated_for(meaning)
    hormone_state = {"cortisol": 0.9, "oxytocin": 0.1}

    before_meaning = deepcopy(meaning)
    before_activated = list(activated)
    before_hormone = dict(hormone_state)
    before_adapter_state = dict(adapter.__dict__)

    result = adapter.verify(
        "괜찮아, 완전 좋아.",
        meaning=meaning,
        activated_categories=activated,
        hormone_state=hormone_state,
    )

    assert result.fallback is not None
    assert meaning == before_meaning
    assert activated == before_activated
    assert hormone_state == before_hormone
    assert adapter.__dict__ == before_adapter_state


def test_agp_fallback_does_not_branch_on_raw_candidate_text():
    adapter = AGPAdapter()
    meaning = AppraisalClassifier().analyze("이 노래 좋다").as_meaning_dict()
    hormone_state = {"cortisol": 0.1}

    first = adapter.verify(
        "첫 번째 아무 문장.",
        meaning=meaning,
        activated_categories=["appraisal"],
        hormone_state=hormone_state,
    )
    second = adapter.verify(
        "완전히 다른 표면 문장.",
        meaning=meaning,
        activated_categories=["appraisal"],
        hormone_state=hormone_state,
    )

    assert first.reason == second.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert first.fallback == second.fallback
