"""EVE v3 round3 — AppraisalClassifier consolidation tests."""

from adapters.appraisal_classifier import (
    AppraisalClassifier,
    LABEL_AMBIENT_WEATHER,
    LABEL_EXTERNAL_AFFECTIVE_TONE,
    LABEL_NEGATIVE_OBJECT,
    LABEL_POSITIVE_OBJECT,
)
from main import build_full_engine


def test_appraisal_classifier_consolidates_existing_positive_negative_guards():
    classifier = AppraisalClassifier()

    positive = classifier.analyze("이 노래 좋다")
    assert positive.label == LABEL_POSITIVE_OBJECT
    assert positive.should_route_small_talk
    assert positive.via == "semantic_guard"

    negative = classifier.analyze("커피 싫어")
    assert negative.label == LABEL_NEGATIVE_OBJECT
    assert negative.should_route_small_talk
    assert negative.via == "semantic_guard"


def test_appraisal_classifier_preserves_weather_and_affective_tone_guards():
    classifier = AppraisalClassifier()

    weather = classifier.analyze("오늘 날씨 좋다")
    assert weather.label == LABEL_AMBIENT_WEATHER
    assert weather.route_override == "small_talk"

    tone = classifier.analyze("이 장면 슬퍼")
    assert tone.label == LABEL_EXTERNAL_AFFECTIVE_TONE
    assert tone.route_override == "small_talk"


def test_appraisal_classifier_does_not_capture_user_inner_affect():
    classifier = AppraisalClassifier()

    for text in ["오늘 기분 좋다", "나 슬퍼", "이 노래 들으니까 외로워"]:
        result = classifier.analyze(text)
        assert not result.should_route_small_talk, (text, result)
        assert result.label == "none"
        assert result.is_user_affect


def test_orchestrator_uses_appraisal_classifier_trace_without_behavior_change():
    engine = build_full_engine()
    meaning = engine.lu.parse("그 코드 마음에 안 들어")

    trace = engine.orchestrator_adapter.classify_with_trace(meaning)
    assert trace["situation"] == "small_talk"
    assert trace["matched_pattern"] == LABEL_NEGATIVE_OBJECT
    assert trace["via"] == "semantic_guard"
    assert engine.orchestrator_adapter.classify(meaning) == "small_talk"
