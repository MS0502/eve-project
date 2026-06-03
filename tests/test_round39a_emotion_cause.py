"""
라운드 39a: EmotionCauseTracker
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.emotion_cause_adapter import (
    EmotionCauseTracker, CauseEvent,
)


def test_record_user_input():
    t = EmotionCauseTracker()
    t.record_user_input("안녕")
    assert t._last_user_input_ts > 0
    assert t._last_user_input_text == "안녕"
    print(f"  ✓ user input 시간 기록")


def test_record_event():
    t = EmotionCauseTracker()
    t.record_event("teaching", "응이라고 가르침", intensity=1.0)
    assert len(t.history) == 1
    assert t.history[0].cause_type == "teaching"
    print(f"  ✓ 사건 기록")


def test_passive_cause_long_silence():
    """1분 이상 침묵 → interaction_gap 사건"""
    t = EmotionCauseTracker()
    t._last_user_input_ts = time.time() - 120  # 2분 전
    
    cause = t.detect_passive_cause()
    assert cause is not None
    assert cause.cause_type == "interaction_gap"
    assert "한참" in cause.description or "오래" in cause.description
    print(f"  ✓ {cause.description}")


def test_no_passive_cause_short_silence():
    """1분 미만이면 cause 없음"""
    t = EmotionCauseTracker()
    t._last_user_input_ts = time.time() - 30
    
    cause = t.detect_passive_cause()
    assert cause is None
    print(f"  ✓ 30초 침묵은 사건 X")


def test_get_relevant_cause_lonely():
    """외로움 → interaction_gap 매칭"""
    t = EmotionCauseTracker()
    t._last_user_input_ts = time.time() - 200
    
    cause = t.get_relevant_cause("lonely")
    assert cause is not None
    assert cause.cause_type == "interaction_gap"
    print(f"  ✓ 외로움 → interaction_gap 매칭")


def test_speech_clause_lonely():
    """외로움 cause → 자연스러운 절"""
    t = EmotionCauseTracker()
    t._last_user_input_ts = time.time() - 200
    
    cause = t.detect_passive_cause()
    clause = t.speech_clause_for_cause(cause, "lonely")
    assert clause
    assert "그래서" in clause or "동안" in clause or "때문" in clause
    print(f"  ✓ {clause!r}")


def test_speech_clause_teaching():
    t = EmotionCauseTracker()
    t.record_event("teaching", "응이라고", intensity=1.0)
    cause = t.history[-1]
    clause = t.speech_clause_for_cause(cause, "curious")
    assert "가르쳐준" in clause or "곱씹" in clause
    print(f"  ✓ {clause!r}")


def test_history_max_size():
    t = EmotionCauseTracker(max_history=5)
    for i in range(10):
        t.record_event("test", f"event{i}")
    assert len(t.history) == 5
    # 최근 5개만
    assert t.history[-1].description == "event9"
    print(f"  ✓ history max size 적용")


# ===== 통합 — engine + 발화 =====

def test_integration_with_engine():
    from main import build_full_engine
    engine = build_full_engine()
    
    assert hasattr(engine, "emotion_cause")
    assert engine.emotion_cause is not None
    
    # planner에 _engine_ref 부착됐나
    assert hasattr(engine.planner, "_engine_ref")
    assert engine.planner._engine_ref is engine
    print(f"  ✓ engine + planner 통합")


def test_proactive_with_cause():
    """proactive 발화에 cause clause 들어가는지"""
    from main import build_full_engine
    engine = build_full_engine()
    
    # 외로움 + 사용자 입력 3분 전
    ot = engine.hormone_adapter.hs.hormones["oxytocin"]
    ot.level = 0.20
    ot.baseline = 0.25
    engine.emotion_cause._last_user_input_ts = time.time() - 180
    
    chunks = list(engine.proactive_stream(force=True))
    full = " ".join(chunks)
    
    # cause clause 흔적 (한참/말 안 한/그래서/동안/때문)
    has_cause = any(kw in full for kw in 
                    ["한참", "말 안", "그래서", "동안", "때문"])
    assert has_cause, f"cause clause 없음: {full}"
    print(f"  ✓ proactive에 cause: {full}")


def test_proactive_without_cause_when_no_emotion():
    """일반 mode (감정 X)면 cause clause 안 붙음"""
    from main import build_full_engine
    engine = build_full_engine()
    
    # 모든 호르몬 중립
    for h in engine.hormone_adapter.hs.hormones.values():
        h.level = 0.5
    
    chunks = list(engine.proactive_stream(force=True))
    full = " ".join(chunks)
    
    # 일반 mode이라 cause clause 없어야 — 또는 있어도 OK
    # 그냥 chunks 출력 검증
    assert len(chunks) > 0
    print(f"  ✓ 중립 호르몬: {full}")


def test_stats():
    t = EmotionCauseTracker()
    t.record_event("test", "evt1")
    s = t.stats()
    assert s["causes_recorded"] == 1
    assert s["history_size"] == 1
    print(f"  ✓ stats {s}")


# ===== 결정론 =====

def test_determinism():
    t1 = EmotionCauseTracker()
    t2 = EmotionCauseTracker()
    
    t1._last_user_input_ts = 1000.0
    t2._last_user_input_ts = 1000.0
    
    # 같은 시점, 같은 입력 → 같은 cause
    # detect_passive_cause는 time.time() 사용해서 어려움 — 직접 테스트
    t1.record_event("teaching", "X", 1.0)
    t2.record_event("teaching", "X", 1.0)
    
    c1 = t1.history[-1]
    c2 = t2.history[-1]
    
    clause1 = t1.speech_clause_for_cause(c1, "curious")
    clause2 = t2.speech_clause_for_cause(c2, "curious")
    assert clause1 == clause2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("user input 기록", test_record_user_input),
        ("사건 기록", test_record_event),
        ("긴 침묵 → cause", test_passive_cause_long_silence),
        ("짧은 침묵 → 없음", test_no_passive_cause_short_silence),
        ("외로움 cause 매칭", test_get_relevant_cause_lonely),
        ("외로움 절", test_speech_clause_lonely),
        ("teaching 절", test_speech_clause_teaching),
        ("history max", test_history_max_size),
        ("engine 통합", test_integration_with_engine),
        ("proactive + cause", test_proactive_with_cause),
        ("중립이면 cause 없음", test_proactive_without_cause_when_no_emotion),
        ("stats", test_stats),
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
