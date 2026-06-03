"""EVE v3 round4 — AppraisalClassifier meaning-field tests.

Round4 must not expand the frozen marker lists. It only exposes stable meaning
fields that later AGP input-side checks can consume.
"""

from adapters.appraisal_classifier import (
    AFFECT_AFFECTIVE_TONE,
    AFFECT_NEGATIVE,
    AFFECT_POSITIVE,
    AFFECT_WEATHER_APPRAISAL,
    AppraisalClassifier,
    LABEL_EXTERNAL_AFFECTIVE_TONE,
    LABEL_NEGATIVE_OBJECT,
    LABEL_POSITIVE_OBJECT,
    SUBJECT_TARGET,
    SUBJECT_USER,
)
from main import build_full_engine


def test_appraisal_result_exposes_meaning_fields_for_target_appraisals():
    classifier = AppraisalClassifier()

    positive = classifier.analyze("이 노래 좋다")
    assert positive.label == LABEL_POSITIVE_OBJECT
    assert positive.subject_type == SUBJECT_TARGET
    assert positive.affect_type == AFFECT_POSITIVE
    assert positive.meaning_layer == "appraisal"

    negative = classifier.analyze("커피 싫어")
    assert negative.label == LABEL_NEGATIVE_OBJECT
    assert negative.subject_type == SUBJECT_TARGET
    assert negative.affect_type == AFFECT_NEGATIVE

    tone = classifier.analyze("이 장면 슬퍼")
    assert tone.label == LABEL_EXTERNAL_AFFECTIVE_TONE
    assert tone.subject_type == SUBJECT_TARGET
    assert tone.affect_type == AFFECT_AFFECTIVE_TONE


def test_appraisal_result_preserves_user_affect_as_user_subject():
    classifier = AppraisalClassifier()

    result = classifier.analyze("오늘 기분 좋다")
    assert result.label == "none"
    assert result.subject_type == SUBJECT_USER
    assert result.affect_type == "none"
    assert result.is_user_affect
    assert not result.should_route_small_talk


def test_appraisal_result_as_meaning_dict_is_stable_for_agp_input():
    classifier = AppraisalClassifier()
    result = classifier.analyze("오늘 날씨 좋다")

    meaning = result.as_meaning_dict()
    assert meaning == {
        "label": "ambient_weather_statement",
        "subject_type": SUBJECT_TARGET,
        "affect_type": AFFECT_WEATHER_APPRAISAL,
        "route_override": "small_talk",
        "via": "semantic_guard",
        "is_target_appraisal": True,
        "is_user_affect": False,
        "reason": "ambient_weather_statement",
    }


def test_orchestrator_trace_includes_appraisal_meaning_without_behavior_change():
    engine = build_full_engine()
    meaning = engine.lu.parse("그 코드 마음에 안 들어")

    trace = engine.orchestrator_adapter.classify_with_trace(meaning)
    assert trace["situation"] == "small_talk"
    assert trace["via"] == "semantic_guard"
    assert trace["appraisal"]["subject_type"] == SUBJECT_TARGET
    assert trace["appraisal"]["affect_type"] == AFFECT_NEGATIVE
    assert engine.orchestrator_adapter.classify(meaning) == "small_talk"
