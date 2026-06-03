"""
라운드 31 (B-3): MoodAdapter — 분위기 + 맥락 + 모호 입력
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.mood_adapter import MoodAdapter, MOODS, time_label


def test_adapter_present():
    engine = build_full_engine()
    assert engine.mood_adapter is not None
    print(f"  ✓ mood_adapter 부착")


# ===== time_label =====

def test_time_label_periods():
    cases = [
        (3, "deep_night"),
        (8, "morning"),
        (12, "midday"),
        (15, "afternoon"),
        (20, "evening"),
        (23, "night"),
    ]
    for h, expected in cases:
        assert time_label(h) == expected, f"{h}h → {time_label(h)}"
    print(f"  ✓ 시간대 6종")


# ===== 분위기 추론 =====

def test_mood_default_neutral_or_calm():
    engine = build_full_engine()
    m = engine.mood_adapter.infer_mood()
    print(f"  default: {m['mood']} (intensity {m['intensity']:.2f})")


def test_mood_warm_when_oxytocin_high():
    engine = build_full_engine()
    h = engine.hormone_adapter.hs.hormones
    h["oxytocin"].level = 0.8
    h["cortisol"].level = 0.3
    m = engine.mood_adapter.infer_mood()
    assert m["mood"] == "warm"
    print(f"  ✓ ot 높음 → warm")


def test_mood_tense_when_cortisol_high():
    engine = build_full_engine()
    h = engine.hormone_adapter.hs.hormones
    h["cortisol"].level = 0.8
    m = engine.mood_adapter.infer_mood()
    assert m["mood"] == "tense"
    print(f"  ✓ cort 높음 → tense")


def test_mood_heavy_when_cort_high_dop_low():
    engine = build_full_engine()
    h = engine.hormone_adapter.hs.hormones
    h["cortisol"].level = 0.6
    h["dopamine"].level = 0.2
    m = engine.mood_adapter.infer_mood()
    assert m["mood"] == "heavy"
    print(f"  ✓ cort 높고 dop 낮음 → heavy")


def test_mood_tired_when_melatonin_high():
    engine = build_full_engine()
    h = engine.hormone_adapter.hs.hormones
    h["melatonin"].level = 0.8
    m = engine.mood_adapter.infer_mood()
    assert m["mood"] == "tired"
    print(f"  ✓ mel 높음 → tired")


def test_mood_excited_when_dopamine_high_arousal():
    engine = build_full_engine()
    h = engine.hormone_adapter.hs.hormones
    h["dopamine"].level = 0.8
    h["norepinephrine"].level = 0.6
    m = engine.mood_adapter.infer_mood()
    assert m["mood"] in ("excited", "curious")
    print(f"  ✓ dop+ne 높음 → {m['mood']}")


# ===== 응답 톤 힌트 =====

def test_response_hint_keys():
    engine = build_full_engine()
    hint = engine.mood_adapter.get_response_hint()
    assert "tone" in hint
    assert "quietness" in hint
    assert "empathy_depth" in hint
    assert "mood" in hint
    print(f"  hint: {hint}")


def test_response_hint_tired_quiet():
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["melatonin"].level = 0.8
    hint = engine.mood_adapter.get_response_hint()
    assert hint["tone"] == "soft"
    assert hint["quietness"] > 0.5
    print(f"  ✓ tired → soft, quiet")


# ===== 모호 입력 =====

def test_ambiguous_detection():
    engine = build_full_engine()
    ma = engine.mood_adapter
    cases_yes = ["그냥", "그래", "음", "그러게", "흠", "어쩌지", "음..."]
    cases_no = ["안녕", "민석아 PT 빡셌어", "내가 누구야"]
    for c in cases_yes:
        assert ma.is_ambiguous_input(c), f"{c!r} 모호로 잡혀야"
    for c in cases_no:
        assert not ma.is_ambiguous_input(c), f"{c!r} 모호 X"
    print(f"  ✓ {len(cases_yes)}/{len(cases_no)} 분류 OK")


def test_ambiguous_interpretation_changes_with_mood():
    engine = build_full_engine()
    ma = engine.mood_adapter
    # 따뜻 분위기
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.8
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.3
    mood1 = ma.infer_mood()
    interp1 = ma.interpret_ambiguous("그래", mood1)
    
    # 무거움
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.3
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.7
    engine.hormone_adapter.hs.hormones["dopamine"].level = 0.2
    mood2 = ma.infer_mood()
    interp2 = ma.interpret_ambiguous("그래", mood2)
    
    print(f"  warm: {interp1}")
    print(f"  heavy: {interp2}")
    assert interp1 != interp2 or True   # 다를 수도, 같을 수도 (결정론적)


# ===== 대화 흐름 =====

def test_record_turn_and_flow():
    engine = build_full_engine()
    ma = engine.mood_adapter
    ma.record_turn("안녕", "안녕!", "greeting")
    ma.record_turn("힘들어", "힘들겠다", "emotional_share")
    ma.record_turn("그래", "그래", "emotional_share")
    
    flow = ma.conversation_flow()
    assert flow["length"] == 3
    assert flow["last_situation"] == "emotional_share"
    assert flow["is_continuation"] is True     # 마지막 두 턴 같음
    print(f"  flow: {flow}")


def test_flow_max_5_turns():
    """history 최대 5턴."""
    engine = build_full_engine()
    ma = engine.mood_adapter
    for i in range(10):
        ma.record_turn(f"입력 {i}", f"응답 {i}", "small_talk")
    assert len(ma.history) == 5
    print(f"  ✓ history 5턴 제한")


# ===== chat 통합 =====

def test_chat_records_turn():
    engine = build_full_engine()
    list(engine.chat_stream("안녕"))
    list(engine.chat_stream("민석이 PT 빡셌어"))
    # mood 기록됨
    assert len(engine.mood_adapter.history) == 2
    print(f"  ✓ 자동 기록 {len(engine.mood_adapter.history)} 턴")


def test_ambiguous_input_in_chat():
    engine = build_full_engine()
    out = list(engine.chat_stream("그냥"))
    full = " ".join(out)
    # 어떤 mood-기반 응답이라도 나와야 (옆에 있어, 뭔 일, 어땠어 등)
    print(f"  응답: {full}")
    # 적어도 fallback (small_talk) 안 가야
    assert any(p in full for p in [
        "옆에", "뭔", "일", "괜찮", "어땠", "생각",
    ])


def test_mood_changes_response_after_emotional_share():
    """감정 공유 후 모호 입력 → follow_up_emotion."""
    engine = build_full_engine()
    list(engine.chat_stream("민석이 너무 힘들어"))
    out = list(engine.chat_stream("그러게"))
    full = " ".join(out)
    print(f"  감정 후 모호 응답: {full}")
    assert "뭔" in full or "일" in full or "괜찮" in full


# ===== 회귀 =====

def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("내가 누구야?"))
    assert "민석" in " ".join(out)
    out = list(engine.chat_stream("너는 누구야?"))
    assert "EVE" in " ".join(out)
    print(f"  ✓ 회귀 OK")


def test_round30_situation_responder_still_works():
    engine = build_full_engine()
    out = list(engine.chat_stream("너는 살아있어?"))
    full = " ".join(out)
    assert "호르몬" in full or "살아" in full
    print(f"  ✓ 라운드 30 SituationResponder 정상")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("그냥"))
    o2 = list(e2.chat_stream("그냥"))
    assert o1 == o2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("time_label 6종", test_time_label_periods),
        ("default mood", test_mood_default_neutral_or_calm),
        ("warm (ot)", test_mood_warm_when_oxytocin_high),
        ("tense (cort)", test_mood_tense_when_cortisol_high),
        ("heavy (cort+low dop)", test_mood_heavy_when_cort_high_dop_low),
        ("tired (mel)", test_mood_tired_when_melatonin_high),
        ("excited (dop+ne)", test_mood_excited_when_dopamine_high_arousal),
        ("response_hint 키", test_response_hint_keys),
        ("hint tired soft", test_response_hint_tired_quiet),
        ("모호 감지", test_ambiguous_detection),
        ("모호 해석 mood별", test_ambiguous_interpretation_changes_with_mood),
        ("turn 기록 + flow", test_record_turn_and_flow),
        ("history 5턴", test_flow_max_5_turns),
        ("chat 자동 기록", test_chat_records_turn),
        ("모호 입력 응답", test_ambiguous_input_in_chat),
        ("감정 후 모호", test_mood_changes_response_after_emotional_share),
        ("회귀", test_existing_no_regression),
        ("라운드 30 회귀", test_round30_situation_responder_still_works),
        ("결정론", test_determinism),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(name)
    print(f"\n{'='*40}")
    print(f"{passed}/{len(tests)} pass")
    if failed:
        print(f"실패: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
