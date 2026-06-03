from main import build_full_engine


def test_negative_object_appraisal_is_not_emotional_share():
    engine = build_full_engine()

    cases = [
        "이 노래 싫다",
        "영화 별로다",
        "커피 싫어",
        "이 사진 별로야",
        "나 이 노래 싫어",
        "그 코드 마음에 안 들어",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        trace = engine.orchestrator_adapter.classify_with_trace(meaning)
        assert trace["situation"] == "small_talk", (text, trace)
        assert trace["via"] == "semantic_guard", (text, trace)
        assert trace["matched_pattern"] == "negative_object_appraisal_statement"


def test_negative_inner_affect_and_interpersonal_still_route_to_emotional_share():
    engine = build_full_engine()

    cases = [
        "기분 별로야",
        "이 노래 들으니까 기분 별로야",
        "오늘 마음이 안 좋아",
        "너 싫어",
        "이브 싫어",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        assert engine.orchestrator_adapter.classify(meaning) == "emotional_share", text


def test_state_debug_negative_object_appraisal_guard_reports_no_false_emotion_risk():
    engine = build_full_engine()
    dump = engine.state_debug.diagnose_text("커피 싫어")

    assert dump["route"]["situation"] == "small_talk"
    assert dump["route"]["via"] == "semantic_guard"
    assert dump["route"]["matched_pattern"] == "negative_object_appraisal_statement"
    assert "negative_object_appraisal_may_be_false_emotion" not in dump["risks"]
