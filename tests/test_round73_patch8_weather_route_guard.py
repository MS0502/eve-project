from main import build_full_engine


def test_positive_weather_observation_is_not_emotional_share():
    engine = build_full_engine()

    for text in ["오늘 날씨 좋다", "오늘 날씨가 좋다", "날씨가 맑고 화창해"]:
        meaning = engine.lu.parse(text)
        trace = engine.orchestrator_adapter.classify_with_trace(meaning)
        assert trace["situation"] == "small_talk"
        assert trace["via"] == "semantic_guard"


def test_user_mood_positive_still_routes_to_emotional_share():
    engine = build_full_engine()

    for text in ["기분 좋아 행복해", "오늘 기분 좋다", "나도 날씨 좋아서 기분 좋아"]:
        meaning = engine.lu.parse(text)
        assert engine.orchestrator_adapter.classify(meaning) == "emotional_share"


def test_weather_question_still_routes_as_question_not_guarded_small_talk():
    engine = build_full_engine()
    meaning = engine.lu.parse("오늘 날씨 어때?")
    trace = engine.orchestrator_adapter.classify_with_trace(meaning)

    assert trace["situation"] == "factual_question"
    assert trace["via"] in {"intent_question", "pattern"}


def test_state_debug_weather_guard_reports_no_false_emotion_risk():
    engine = build_full_engine()
    dump = engine.state_debug.diagnose_text("오늘 날씨 좋다")

    assert dump["route"]["situation"] == "small_talk"
    assert dump["route"]["via"] == "semantic_guard"
    assert "positive_weather_may_be_false_emotion" not in dump["risks"]
