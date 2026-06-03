from main import build_full_engine


def test_external_threat_or_discomfort_is_not_emotional_share():
    engine = build_full_engine()

    cases = [
        "공포영화 무서워",
        "이 의자 불편해",
        "그 상황 위험해",
        "여기 위험하다",
        "이 코드 위험해",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        trace = engine.orchestrator_adapter.classify_with_trace(meaning)
        assert trace["situation"] == "small_talk", (text, trace)
        assert trace["via"] == "semantic_guard", (text, trace)
        assert trace["matched_pattern"] == "external_threat_or_discomfort_statement"


def test_inner_fear_and_discomfort_still_route_to_emotional_share():
    engine = build_full_engine()

    cases = [
        "나 무서워",
        "기분이 불편해",
        "마음이 불편해",
        "불안하고 걱정돼",
        "너 무서워",
    ]
    for text in cases:
        meaning = engine.lu.parse(text)
        assert engine.orchestrator_adapter.classify(meaning) == "emotional_share", text


def test_state_debug_external_threat_guard_reports_no_false_emotion_risk():
    engine = build_full_engine()
    dump = engine.state_debug.diagnose_text("공포영화 무서워")

    assert dump["route"]["situation"] == "small_talk"
    assert dump["route"]["via"] == "semantic_guard"
    assert dump["route"]["matched_pattern"] == "external_threat_or_discomfort_statement"
    assert "external_threat_or_discomfort_may_be_false_emotion" not in dump["risks"]
