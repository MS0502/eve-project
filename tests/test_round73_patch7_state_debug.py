from main import build_full_engine


def test_state_debug_is_wired_and_read_only_for_route_history():
    engine = build_full_engine()
    before = len(engine.orchestrator_adapter.history(100))
    dump = engine.state_debug.diagnose_text("이브야")
    after = len(engine.orchestrator_adapter.history(100))

    assert before == after
    assert dump["input"] == "이브야"
    assert dump["route"]["situation"] == "greeting"
    assert dump["route"]["via"] == "pattern"
    assert "user_presence_adapter" in dump["route"]["modules_selected"]
    assert "activation_top" in dump["state"]


def test_state_debug_reports_weather_guard_route():
    engine = build_full_engine()
    dump = engine.state_debug.diagnose_text("오늘 날씨 좋다")

    assert dump["route"]["situation"] == "small_talk"
    assert dump["route"]["via"] == "semantic_guard"
    assert "positive_weather_may_be_false_emotion" not in dump["risks"]


def test_state_debug_detects_state_that_can_intercept_next_input():
    engine = build_full_engine()
    # Simulate EVE having just asked for a concept explanation.
    engine.dialogue_context.record_eve_response(
        "인사 처음 들어봐. 설명해주면 기억할게.",
        recent_user_input="인사가 뭔지 알아?",
    )

    dump = engine.state_debug.diagnose_text("처음 만났을 때 하는 말이야")

    assert dump["state"]["dialogue_context"]["pending"] == "인사"
    assert "pending_dialogue_answer_can_intercept_next_input" in dump["risks"]
    rendered = engine.state_debug.render(dump)
    assert "pending_question: 인사" in rendered
    assert "route:" in rendered


def test_state_debug_exposes_learned_teaching_injection_risk():
    engine = build_full_engine()
    engine.teaching_adapter.learn("응", ["힘들"])

    dump = engine.state_debug.diagnose_text("오늘 힘들다")

    assert dump["state"]["teaching"]["learned_count"] == 1
    assert dump["state"]["teaching"]["top"][0]["response"] == "응"
    assert "learned_teaching_response_can_inject_output" in dump["risks"]
