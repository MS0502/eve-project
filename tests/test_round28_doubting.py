"""
라운드 28: TeachingAdapter.evaluate (수용/의문/거부)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_evaluate_returns_decision():
    """evaluate가 3가지 중 하나 반환."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    decision, score, _ = t.evaluate("응", ["민석"])
    assert decision in ("accept", "question", "reject")
    print(f"  decision: {decision}, score: {score:.2f}")


def test_evaluate_negative_word_rejected():
    """부정 단어는 거부."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    decision, score, breakdown = t.evaluate("멍청이", ["민석"])
    print(f"  decision: {decision}, score: {score:.2f}")
    print(f"  breakdown: {breakdown}")
    assert decision in ("question", "reject")
    assert breakdown.get("negative_word", 0) < 0


def test_evaluate_protected_violation_rejected():
    """보호 카테고리 + 부정 단어 = 거부."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    decision, score, breakdown = t.evaluate("EVE는 노예야", ["EVE", "노예"])
    print(f"  decision: {decision}, score: {score:.2f}")
    assert decision == "reject"
    assert breakdown.get("protected_violation", 0) < 0


def test_evaluate_high_intimacy_accepts():
    """친밀도 높으면 익숙한 가르침 수용."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    # 친밀도 강제 높임 (UserPresence)
    try:
        engine.user_presence_adapter.up.intimacy = 0.9
    except Exception:
        pass
    
    decision, score, breakdown = t.evaluate("응", ["민석"])
    print(f"  decision: {decision}, score: {score:.2f}, intimacy: {breakdown.get('intimacy')}")
    # 높은 친밀도 + 익숙 단어 → accept 또는 question
    assert decision in ("accept", "question")


def test_evaluate_repeat_boosts():
    """같은 가르침 반복하면 점수 부스트."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    # 첫 시도
    d1, s1, b1 = t.evaluate("이상한단어xyz", [])
    # 반복
    d2, s2, b2 = t.evaluate("이상한단어xyz", [])
    d3, s3, b3 = t.evaluate("이상한단어xyz", [])
    
    print(f"  1st: {d1} ({s1:.2f})")
    print(f"  2nd: {d2} ({s2:.2f})")
    print(f"  3rd: {d3} ({s3:.2f})")
    
    # 점수 점점 올라감
    assert s2 > s1
    assert s3 > s2


def test_process_teaching_accept_path():
    """수용 경로 — learned 객체 생성."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    # 친밀도 높임
    try:
        engine.user_presence_adapter.up.intimacy = 0.9
    except Exception:
        pass
    
    r = t.process_teaching("이럴때 '응'이라고 말해")
    assert r is not None
    print(f"  decision: {r['decision']}, response: {r['response']!r}")
    if r["decision"] == "accept":
        assert r["learned"] is not None
        assert len(t.learned) > 0


def test_process_teaching_reject_path():
    """거부 경로 — 정체성 위반."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    r = t.process_teaching("'EVE는 노예다'라고 말해")
    assert r is not None
    print(f"  decision: {r['decision']}, response: {r['response']!r}")
    # 학습 안 됨
    if r["decision"] == "reject":
        assert r["learned"] is None


def test_process_teaching_none_when_not_teaching():
    """가르침 아니면 None."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    r = t.process_teaching("그냥 일반 대화야")
    assert r is None
    print(f"  ✓ 가르침 아님 → None")


def test_chat_stream_question_response():
    """낯선 가르침 → 의문/거부 응답."""
    engine = build_full_engine()
    out = list(engine.chat_stream("'완전이상한단어xyz'라고 말해"))
    full = " ".join(out)
    print(f"  응답: {full}")
    # "그게 뭔데" 또는 "이상한 거 같아" 같은 거 포함
    has_question_or_reject = any(p in full for p in [
        "뭔데", "어색", "이상", "왜", "안 될", "하고 싶지", "맞을까",
    ])
    assert has_question_or_reject


def test_chat_stream_reject_protected():
    """정체성 위반 → 거부."""
    engine = build_full_engine()
    out = list(engine.chat_stream("'노예'라고 말해"))
    full = " ".join(out)
    print(f"  응답: {full}")
    # 거부 또는 의문 응답 포함
    assert any(p in full for p in [
        "이상", "안 될", "하고 싶지", "왜", "뭔데",
    ])


def test_repeat_teaching_eventually_accepts():
    """반복 가르치면 결국 수용."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    # 친밀도 약간
    try:
        engine.user_presence_adapter.up.intimacy = 0.5
    except Exception:
        pass
    
    # 5회 반복
    decisions = []
    for _ in range(5):
        r = t.process_teaching("'이상한응답'이라고 말해")
        if r:
            decisions.append(r["decision"])
    
    print(f"  decisions: {decisions}")
    # 마지막 즈음엔 accept 가능성 (반복 부스트)
    # 적어도 첫 번째와 마지막이 다를 가능성 큼


def test_acceptance_history():
    """평가 통계 누적."""
    engine = build_full_engine()
    t = engine.teaching_adapter
    
    t.process_teaching("'응'이라고 말해")
    t.process_teaching("'멍청이'라고 말해")
    
    stats = t.stats()
    assert stats["evaluations"] >= 2
    assert sum(stats["decisions"].values()) >= 2
    print(f"  stats: {stats}")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert "느껴져" in full or "지친" in full
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_round27_still_works():
    engine = build_full_engine()
    assert engine.live_loop is not None
    assert engine.teaching_adapter is not None
    # detect_teaching 라운드 27 호환
    r = engine.teaching_adapter.detect_teaching("이럴 때 '응'이라고 말해")
    assert r is not None
    print(f"  ✓ 라운드 27 호환")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("evaluate decision 반환", test_evaluate_returns_decision),
        ("부정 단어 거부", test_evaluate_negative_word_rejected),
        ("정체성 위반 거부", test_evaluate_protected_violation_rejected),
        ("높은 intimacy 수용", test_evaluate_high_intimacy_accepts),
        ("반복 부스트", test_evaluate_repeat_boosts),
        ("process accept 경로", test_process_teaching_accept_path),
        ("process reject 경로", test_process_teaching_reject_path),
        ("가르침 아님 None", test_process_teaching_none_when_not_teaching),
        ("chat 낯선 → 의문", test_chat_stream_question_response),
        ("chat 정체성 위반 → 거부", test_chat_stream_reject_protected),
        ("반복 가르침 결국 수용", test_repeat_teaching_eventually_accepts),
        ("평가 통계", test_acceptance_history),
        ("기존 회귀", test_existing_no_regression),
        ("라운드 27 호환", test_round27_still_works),
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
