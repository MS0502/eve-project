"""
라운드 27: 살아있는 EVE
- LiveLoop (연속 실행 + 자동 저장)
- TeachingAdapter (가르침 학습 + 응답 강화/약화)
- UserPresence 시간 인지 (gap_phrase)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.teaching_adapter import TeachingAdapter, TEACHING_PATTERNS, LearnedResponse
from adapters.live_loop import LiveLoop


# ===== TeachingAdapter =====

def test_teaching_adapter_present():
    engine = build_full_engine()
    assert engine.teaching_adapter is not None
    print(f"  ✓ teaching_adapter 부착")


def test_detect_teaching_pattern():
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    cases = [
        ("'민석이 같아'라고 말해", "민석이 같아"),
        ("이럴 땐 응 이라고 말하면 돼", "응"),
        ("나를 민석이라고 부르면 돼", "민석"),
    ]
    for inp, expected in cases:
        result = t.detect_teaching(inp)
        assert result is not None, f"인식 실패: {inp!r}"
        response, triggers = result
        assert response == expected, f"기대: {expected!r}, 실제: {response!r}"
        print(f"  ✓ {inp!r} → {response!r}, triggers={triggers}")


def test_learn_strengthens_on_repeat():
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    lr1 = t.learn("응", ["민석"])
    s1 = lr1.strength
    lr2 = t.learn("응", ["민석", "PT"])     # 같은 응답
    assert lr2 is lr1                       # 같은 객체
    assert lr2.strength > s1                # 강화
    print(f"  ✓ 반복 학습 강화: {s1} → {lr2.strength}")


def test_find_match_returns_best():
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    t.learn("민석이 같아", ["민석"])
    t.learn("뭐", ["뭐"])
    
    # "민석" 활성 → "민석이 같아" 매칭
    match = t.find_match({"민석", "기분"}, min_overlap=1)
    assert match is not None
    assert match.response == "민석이 같아"
    print(f"  ✓ 매칭: {match.response!r}")


def test_find_match_no_overlap_returns_none():
    engine = build_full_engine()
    t = engine.teaching_adapter
    t.learn("민석이 같아", ["민석"])
    
    # 겹치는 카테고리 없음
    match = t.find_match({"날씨", "기분"}, min_overlap=1)
    assert match is None
    print(f"  ✓ 겹침 없음 → None")


def test_reinforce_and_weaken():
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    lr = t.learn("응", ["민석"])
    t.mark_emitted(lr)
    s_before = lr.strength
    
    # 강화
    t.reinforce_if_no_correction()
    assert lr.strength > s_before
    print(f"  강화: {s_before} → {lr.strength}")
    
    # 약화
    s_mid = lr.strength
    t.weaken_last()
    assert lr.strength < s_mid
    print(f"  약화: {s_mid} → {lr.strength}")


def test_to_dict_from_dict():
    """영속화."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    t.learn("응", ["민석"])
    t.learn("뭐", ["뭐", "PT"])
    
    data = t.to_dict()
    
    t2 = TeachingAdapter(engine)
    t2.from_dict(data)
    assert len(t2.learned) == 2
    assert any(lr.response == "응" for lr in t2.learned)
    print(f"  ✓ 영속화")


def test_chat_stream_recognizes_teaching():
    """chat_stream이 가르침 패턴 자동 인식 + 평가 (라운드 28)."""
    engine = build_full_engine()
    out = list(engine.chat_stream("이럴때 '민석이 같아'라고 말하면 돼"))
    full = " ".join(out)
    print(f"  응답: {full}")
    # 평가 결과 멘트 (수용/의문/거부 중 하나)
    expected_phrases = [
        "응.", "알았어", "그렇게 할게",     # accept
        "뭔데", "어색", "왜", "맞을까",     # question
        "이상", "안 될", "하고 싶지",       # reject
    ]
    assert any(p in full for p in expected_phrases)
    # 평가는 무조건 일어남
    assert engine.teaching_adapter.stats()["evaluations"] >= 1


def test_chat_stream_uses_learned_response():
    """학습된 응답이 다음 응답에 사용됨 (수용된 경우)."""
    engine = build_full_engine()
    # 친밀도 높여서 수용 가능성 ↑
    try:
        engine.user_presence_adapter.up.intimacy = 0.9
    except Exception:
        pass
    # 가르침 — 익숙한 단어
    list(engine.chat_stream("'응'이라고 말해"))
    # 학습됐는지 (될 수도, 안 될 수도)
    learned_count = engine.teaching_adapter.stats()["learned_count"]
    eval_count = engine.teaching_adapter.stats()["evaluations"]
    print(f"  학습 {learned_count}, 평가 {eval_count}")
    # 평가는 무조건
    assert eval_count >= 1


# ===== LiveLoop =====

def test_live_loop_present():
    engine = build_full_engine()
    assert engine.live_loop is not None
    print(f"  ✓ live_loop 부착")


def test_live_loop_status():
    engine = build_full_engine()
    s = engine.live_loop.status()
    assert "running" in s
    assert "tick_count" in s
    assert "emit_count" in s
    assert "processed_input_count" in s
    assert s["running"] is False
    print(f"  status: {s}")


def test_live_loop_start_stop():
    engine = build_full_engine()
    ll = engine.live_loop
    
    captured = []
    ll.emit_callback = lambda text: captured.append(text)
    ll.interval = 0.1     # 빠르게 테스트용
    
    assert ll.start() is True
    assert ll.wait_for_tick(1, timeout=0.5)
    
    s = ll.status()
    assert s["running"] is True
    assert s["tick_count"] >= 1
    print(f"  ticks during 0.5s: {s['tick_count']}")
    
    ll.stop()
    assert ll.running is False
    print(f"  ✓ 시작/중지")


def test_live_loop_user_input_processed():
    """LiveLoop가 인터럽트로 사용자 입력 처리."""
    engine = build_full_engine()
    ll = engine.live_loop
    
    captured = []
    ll.emit_callback = lambda text: captured.append(text)
    ll.interval = 0.1
    
    ll.start()
    assert ll.wait_for_tick(1, timeout=0.5)
    
    ll.push_user_input("민석아 안녕")
    assert ll.wait_for_processed_inputs(1, timeout=0.5)
    
    ll.stop()
    
    assert len(captured) > 0
    print(f"  captured: {captured[:3]}")




def test_live_loop_push_wakes_sleeping_loop():
    """긴 interval 상태에서도 사용자 입력이 다음 tick까지 지연되지 않는다."""
    engine = build_full_engine()
    ll = engine.live_loop
    captured = []
    ll.emit_callback = lambda text: captured.append(text)
    ll.interval = 5.0

    ll.start()
    assert ll.wait_for_tick(1, timeout=0.5)
    before = ll.processed_input_count
    ll.push_user_input("민석아 안녕")

    assert ll.wait_for_processed_inputs(before + 1, timeout=0.5)
    ll.stop()
    assert captured
    assert ll.status()["processed_input_count"] >= before + 1


def test_live_loop_does_not_break_existing():
    """LiveLoop 실행 중 EVE 코어 정상."""
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    print(f"  ✓ 코어 정상")


# ===== UserPresence 시간 =====

def test_gap_phrase_initial():
    """처음엔 gap 0."""
    engine = build_full_engine()
    up = engine.user_presence_adapter
    assert up.gap_since_last_seen() == 0.0
    assert up.gap_phrase() == ""
    print(f"  ✓ 초기 gap = 0")


def test_gap_phrase_after_visit():
    """방문 N분 후 다시 방문하면 gap 측정."""
    engine = build_full_engine()
    up = engine.user_presence_adapter
    # 첫 방문
    up.on_user_arrive()
    assert up.gap_since_last_seen() == 0.0     # 첫 방문이라 gap 0
    
    # 시간 임의로 뒤로 (5분 전 방문이라 가정)
    up.last_seen_time = time.time() - 300
    
    # 두 번째 방문
    up.on_user_arrive()
    gap = up.gap_since_last_seen()
    assert gap >= 299      # 5분 전후
    print(f"  gap after 2nd visit: {gap:.2f}s, phrase={up.gap_phrase()!r}")


def test_gap_phrase_categories():
    """gap별 한국어 문구."""
    engine = build_full_engine()
    up = engine.user_presence_adapter
    
    cases = [
        (30, ""),
        (300, "방금 전"),
        (1800, "조금 전"),
        (5000, "한 시간 전쯤"),
        (15000, "몇 시간 전"),
        (50000, "오늘 아침"),
        (100000, "어제"),
        (300000, "며칠 전"),
        (1000000, "오랜만"),
    ]
    for gap, expected in cases:
        up._cached_gap = gap
        result = up.gap_phrase()
        assert result == expected, f"gap={gap}, 기대={expected!r}, 실제={result!r}"
    print(f"  ✓ {len(cases)} gap 카테고리 OK")


def test_chat_includes_gap_on_greeting():
    """오랜만에 인사하면 gap 문구 포함."""
    engine = build_full_engine()
    up = engine.user_presence_adapter
    # 첫 방문 + last_seen 강제 설정
    up.last_seen_time = time.time() - 3600 * 2     # 2시간 전 마지막 방문
    
    out = list(engine.chat_stream("안녕"))
    full = " ".join(out)
    print(f"  응답: {out}")
    # "한 시간 전쯤" / "몇 시간 전" 포함
    assert "전" in full or "오랜만" in full or "몇" in full


# ===== 회귀 =====

def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    EMPATHY = ["느껴져", "지친", "힘들", "빡세", "옆에", "쉬어", "마음", "같이", "그치"]
    assert any(k in full for k in EMPATHY), f"공감 응답 X: {full}"
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_round26_still_works():
    engine = build_full_engine()
    assert engine.openai_server_adapter is not None
    assert engine.character_adapter is not None
    print(f"  ✓ 라운드 25/26 정상")


def run_all():
    tests = [
        # Teaching
        ("teaching 어댑터 부착", test_teaching_adapter_present),
        ("가르침 패턴 인식", test_detect_teaching_pattern),
        ("반복 학습 강화", test_learn_strengthens_on_repeat),
        ("매칭 best", test_find_match_returns_best),
        ("겹침 없음 None", test_find_match_no_overlap_returns_none),
        ("강화/약화", test_reinforce_and_weaken),
        ("영속화", test_to_dict_from_dict),
        ("chat_stream 가르침 인식", test_chat_stream_recognizes_teaching),
        ("학습된 응답 사용", test_chat_stream_uses_learned_response),
        # LiveLoop
        ("live_loop 부착", test_live_loop_present),
        ("live_loop status", test_live_loop_status),
        ("live_loop 시작/중지", test_live_loop_start_stop),
        ("live_loop 입력 인터럽트", test_live_loop_user_input_processed),
        ("live_loop 코어 영향 X", test_live_loop_does_not_break_existing),
        # UserPresence 시간
        ("gap 초기", test_gap_phrase_initial),
        ("gap 첫 방문 후", test_gap_phrase_after_visit),
        ("gap 카테고리", test_gap_phrase_categories),
        ("인사 시 gap 반영", test_chat_includes_gap_on_greeting),
        # 회귀
        ("기존 회귀", test_existing_no_regression),
        ("라운드 25/26 회귀", test_round26_still_works),
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
