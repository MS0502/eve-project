from main import build_full_engine
from utils.types import Meaning


def test_external_affective_tone_is_not_user_emotional_share():
    engine = build_full_engine()
    cases = [
        "영화 슬프다",
        "이 장면 슬퍼",
        "이 이야기는 우울하다",
        "그 노래 외로워",
    ]
    for text in cases:
        trace = engine.orchestrator_adapter.classify_with_trace(Meaning(raw_text=text))
        assert trace["situation"] == "small_talk", (text, trace)
        assert trace["via"] == "semantic_guard", (text, trace)
        assert trace["matched_pattern"] == "external_affective_tone_statement", (text, trace)


def test_user_affective_reaction_still_routes_to_emotional_share():
    engine = build_full_engine()
    cases = [
        "나 슬퍼",
        "기분이 우울해",
        "영화 보고 슬펐어",
        "이 노래 들으니까 외로워",
    ]
    for text in cases:
        assert engine.orchestrator_adapter.classify(Meaning(raw_text=text)) == "emotional_share", text


def test_affective_tone_guard_prevents_broad_teaching_pattern_capture():
    engine = build_full_engine()
    trace = engine.orchestrator_adapter.classify_with_trace(Meaning(raw_text="이 이야기는 우울하다"))
    assert trace["situation"] == "small_talk"
    assert trace["via"] == "semantic_guard"
