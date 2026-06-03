from main import build_full_engine


def test_positive_object_appraisal_is_not_emotional_share():
    engine = build_full_engine()

    cases = [
        "이 노래 좋다",
        "영화 좋다",
        "커피 좋다",
        "분위기 좋다",
        "이 사진 좋아",
        "나 이 노래 좋아",
        "그 코드 괜찮다",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        trace = engine.orchestrator_adapter.classify_with_trace(meaning)
        assert trace["situation"] == "small_talk", (text, trace)
        assert trace["via"] == "semantic_guard", (text, trace)
        assert trace["matched_pattern"] == "positive_object_appraisal_statement"


def test_inner_affect_and_interpersonal_positive_still_route_to_emotional_share():
    engine = build_full_engine()

    cases = [
        "기분 좋아 행복해",
        "오늘 기분 좋다",
        "너 좋아",
        "이브 좋아",
        "이 노래 들으니까 기분 좋아",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        assert engine.orchestrator_adapter.classify(meaning) == "emotional_share", text


def test_state_debug_object_appraisal_guard_reports_no_false_emotion_risk():
    engine = build_full_engine()
    dump = engine.state_debug.diagnose_text("이 노래 좋다")

    assert dump["route"]["situation"] == "small_talk"
    assert dump["route"]["via"] == "semantic_guard"
    assert "positive_object_appraisal_may_be_false_emotion" not in dump["risks"]
