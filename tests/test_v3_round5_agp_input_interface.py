"""EVE v3 round5 — AGP input-side interface tests.

Round5 lets AGP accept AppraisalClassifier meaning summaries without changing
runtime routing or performing output verification yet.
"""

from adapters.agp_adapter import (
    AGPAdapter,
    AGP_INPUT_LAYER_APPRAISAL,
    AGP_INPUT_REASON_APPRAISAL_ACCEPTED,
    AGP_INPUT_REASON_NO_MEANING,
    AGP_INPUT_REASON_UNSUPPORTED_MEANING,
)
from adapters.appraisal_classifier import (
    AFFECT_NEGATIVE,
    AppraisalClassifier,
    SUBJECT_TARGET,
)
from main import build_full_engine


def test_agp_accepts_appraisal_meaning_dict_without_runtime_verification():
    classifier = AppraisalClassifier()
    meaning = classifier.analyze("그 코드 마음에 안 들어").as_meaning_dict()

    result = AGPAdapter().accept_input_meaning(meaning)

    assert result.accepted is True
    assert result.reason == AGP_INPUT_REASON_APPRAISAL_ACCEPTED
    assert result.meaning_layer == AGP_INPUT_LAYER_APPRAISAL
    assert result.requires_output_verification is True
    assert result.normalized_meaning["subject_type"] == SUBJECT_TARGET
    assert result.normalized_meaning["affect_type"] == AFFECT_NEGATIVE
    assert result.normalized_meaning["route_override"] == "small_talk"


def test_agp_accepts_appraisal_result_object_directly():
    appraisal = AppraisalClassifier().analyze("이 노래 좋다")

    result = AGPAdapter().accept_input_meaning(appraisal)

    assert result.accepted is True
    assert result.normalized_meaning["label"] == "positive_object_appraisal_statement"
    assert result.normalized_meaning["meaning_layer"] == "appraisal"


def test_agp_rejects_missing_or_unsupported_input_meaning_without_fallback_side_effect():
    adapter = AGPAdapter()

    empty = adapter.accept_input_meaning(None)
    assert empty.accepted is False
    assert empty.reason == AGP_INPUT_REASON_NO_MEANING
    assert empty.normalized_meaning == {}

    unsupported = adapter.accept_input_meaning({"intent": "statement"})
    assert unsupported.accepted is False
    assert unsupported.reason == AGP_INPUT_REASON_UNSUPPORTED_MEANING
    assert unsupported.normalized_meaning == {"intent": "statement"}


def test_orchestrator_appraisal_trace_can_feed_agp_without_route_change():
    engine = build_full_engine()
    meaning = engine.lu.parse("오늘 날씨 좋다")

    trace = engine.orchestrator_adapter.classify_with_trace(meaning)
    agp_input = AGPAdapter().accept_input_meaning(trace["appraisal"])

    assert trace["situation"] == "small_talk"
    assert engine.orchestrator_adapter.classify(meaning) == "small_talk"
    assert agp_input.accepted is True
    assert agp_input.normalized_meaning["label"] == "ambient_weather_statement"
