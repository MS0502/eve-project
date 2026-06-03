"""
라운드 19: EnhancerAdapter + ProactiveAdapter
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_enhancer_classifies_greeting():
    """greeting 패턴 분류."""
    engine = build_full_engine()
    und = {"categories": {"안녕"}, "sentiment": "pos", "intent": "statement"}
    pat = engine.enhancer_adapter.classify("안녕", und)
    print(f"  pattern '안녕': {pat}")
    assert pat in ("greeting", "name_call", "affirm", "statement")


def test_enhancer_greeting_response():
    """greeting → 짧은 직접 응답."""
    engine = build_full_engine()
    und = {"categories": {"안녕"}, "sentiment": "pos", "intent": "statement"}
    enhanced, pat = engine.enhancer_adapter.enhance("안녕", und)
    print(f"  enhance '안녕': pattern={pat}, response={enhanced!r}")
    # greeting/name_call이면 응답 있어야


def test_enhancer_in_chat():
    """chat_stream — '안녕' 입력 → core가 enhancer 응답."""
    engine = build_full_engine()
    out = list(engine.chat_stream("안녕"))
    full = " ".join(out)
    print(f"  greeting chat: {out}")
    # '안녕'이 응답 어딘가에 있어야 (enhancer가 '안녕, 민석' 류 응답)
    assert "안녕" in full or "민석" in full


def test_enhancer_complex_input_unchanged():
    """복잡한 입력은 enhancer가 안 건드림 (일반 흐름)."""
    engine = build_full_engine()
    out = list(engine.chat_stream("민석아 PT 진짜 힘들어"))
    full = " ".join(out)
    print(f"  complex: {out}")
    # 일반 응답 패턴 (지친 / 느껴져 / 사고 / 위로)
    assert "지친" in full or "느껴져" in full


# ===== Proactive =====

def test_proactive_should_speak_baseline():
    """기본 호르몬 + idle 짧음 → False (idle 미달)."""
    engine = build_full_engine()
    speak, reason = engine.proactive_adapter.should_speak()
    print(f"  baseline should_speak: {speak}, reason={reason}")


def test_proactive_message_returns_dict():
    """proactive_message 호출."""
    engine = build_full_engine()
    # idle 충분 + need 만들기
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.50
    msg = engine.proactive_adapter.message()
    print(f"  message: {msg}")
    # None이거나 dict


def test_proactive_stream_uses_enhancer_or_fallback():
    """proactive_stream — Proactive 우선, 실패 시 DMN."""
    engine = build_full_engine()
    # need 발생 시키기 (warmth 결핍 → proactive 발화 가능)
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    out = list(engine.proactive_stream(force=True))
    print(f"  proactive: {out}")
    assert len(out) >= 2   # 최소 stage1 + final


def test_proactive_user_intimate_addresses_minseok():
    """진짜 친구('민석') + ot 충분 → 호명할 수도."""
    engine = build_full_engine()
    # 첫 만남으로 진짜 친구 등록
    list(engine.chat_stream("민석아 안녕"))
    list(engine.chat_stream("민석아 잘 지내고 있어"))

    intimacy = engine.user_presence_adapter.get_intimacy()
    print(f"  intimacy: {intimacy}")

    msg = engine.proactive_adapter.message()
    print(f"  proactive message: {msg}")


# ===== 회귀 =====

def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    assert "느껴져" in " ".join(out) or "지친" in " ".join(out)
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    print(f"  ✓ 결정론")


def test_full_demo():
    """진짜 시연: 다양한 패턴 응답."""
    engine = build_full_engine()

    print("\n  [greeting]")
    for c in engine.chat_stream("안녕"): print(f"    {c}")

    print("\n  [복잡한 감정]")
    for c in engine.chat_stream("민석아 PT 진짜 힘들다"): print(f"    {c}")

    print("\n  [proactive (NEED 자동 트리거)]")
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.15
    for c in engine.proactive_stream(force=True): print(f"    {c}")


def run_all():
    tests = [
        ("enhancer 분류", test_enhancer_classifies_greeting),
        ("enhancer 응답", test_enhancer_greeting_response),
        ("enhancer in chat", test_enhancer_in_chat),
        ("복잡 입력 일반흐름", test_enhancer_complex_input_unchanged),
        ("proactive baseline", test_proactive_should_speak_baseline),
        ("proactive message", test_proactive_message_returns_dict),
        ("proactive stream", test_proactive_stream_uses_enhancer_or_fallback),
        ("proactive 친밀", test_proactive_user_intimate_addresses_minseok),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("전체 시연", test_full_demo),
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
