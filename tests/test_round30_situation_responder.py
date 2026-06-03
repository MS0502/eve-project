"""
라운드 30 (B-2): SituationResponder — 상황 기반 응답 합성
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


# ===== 어댑터 =====

def test_adapter_present():
    engine = build_full_engine()
    assert engine.situation_responder is not None
    print(f"  ✓ situation_responder 부착")


def test_can_handle():
    engine = build_full_engine()
    sr = engine.situation_responder
    for s in ["greeting", "meta_self", "meta_user",
              "emotional_share", "factual_question",
              "causal_what_if", "past_recall"]:
        assert sr.can_handle(s)
    # task와 teaching은 다른 곳에서 처리
    assert not sr.can_handle("task")
    assert not sr.can_handle("teaching")
    print(f"  ✓ can_handle 7개 + 양보 2개")


# ===== 진짜 응답 변화 (핵심) =====

def test_meta_user_real_answer():
    """'내가 누구야?' → 진짜 답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("내가 누구야?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "민석" in full
    # 카테고리 회상 X, 진짜 답
    assert "친구" in full or "이야" in full


def test_meta_self_real_answer():
    """'너는 누구?' → 진짜 답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("너는 누구야?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "EVE" in full
    assert "민석" in full or "친구" in full


def test_meta_self_alive_question():
    """'너는 살아있어?' → 자기 인식 응답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("너는 살아있어?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "살아" in full
    # 호르몬/카테고리 언급
    assert "호르몬" in full or "카테고리" in full or "따뜻" in full


def test_emotional_real_empathy():
    """'PT 빡셌어' → 진짜 공감"""
    engine = build_full_engine()
    out = list(engine.chat_stream("민석이 PT 빡셌어"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "힘들" in full or "쉬어" in full or "느껴져" in full


def test_factual_unknown_admits():
    """'오늘 날씨 어때?' → 모름 인정"""
    engine = build_full_engine()
    out = list(engine.chat_stream("오늘 날씨 어때?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "모르" in full or "알아볼" in full


def test_factual_location_yields_to_env():
    """위치 질문 → env_adapter가 답하게 양보"""
    engine = build_full_engine()
    out = list(engine.chat_stream("어디 있어?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "내방" in full      # env_adapter가 처리


def test_causal_what_if_real_answer():
    """'만약 ~' → 반사실 응답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("만약 PT 안 했으면 어땠을까"))
    full = " ".join(out)
    print(f"  응답: {out}")
    assert "어떻게" in full or "달랐" in full or "그랬" in full or "잘 모르" in full


def test_past_recall_response():
    """'저번에 ~' → 회상 응답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("저번에 그 얘기 기억나?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    # 기억나거나 안 나거나 둘 다 OK
    assert "기억" in full or "언제" in full


# ===== teaching 통합 =====

def test_short_teaching_works():
    """짧은 가르침은 정상 학습."""
    engine = build_full_engine()
    out = list(engine.chat_stream("이럴때 민석이 같아 라고 말해"))
    full = " ".join(out)
    print(f"  응답: {full}")
    assert engine.teaching_adapter.stats()["learned_count"] >= 1


def test_long_teaching_rejected():
    """너무 긴 가르침 (응답 12자 초과)은 일반 대화로 처리."""
    engine = build_full_engine()
    long_text = "맞아 나는 민석이야 이럴땐 민석이 같아 라고 말하면 돼"
    out = list(engine.chat_stream(long_text))
    full = " ".join(out)
    print(f"  응답: {out}")
    # 가르침 응답 ("응. 이제 알아.") 안 나옴
    assert "이제 알아" not in full


# ===== 구조 =====

def test_returns_responseplan():
    engine = build_full_engine()
    sr = engine.situation_responder
    from utils.types import Meaning
    m = Meaning(intent="question", entities=["내"])
    m.raw_text = "내가 누구야?"
    
    # 가짜 ctx
    ctx = {
        "user_presence": {"user_name": "민석", "intimacy": 0.8},
    }
    plan = sr.build_plan("meta_user", ctx, m)
    assert plan is not None
    assert plan.core_message
    print(f"  plan core: {plan.core_message!r}")


def test_returns_none_for_unhandled():
    engine = build_full_engine()
    sr = engine.situation_responder
    from utils.types import Meaning
    m = Meaning()
    m.raw_text = "test"
    plan = sr.build_plan("task", {}, m)     # task는 양보
    # method 자체가 없으니 None
    assert plan is None
    print(f"  ✓ 미지원 상황 → None")


# ===== 회귀 =====

def test_existing_no_regression():
    """기존 핵심 흐름 안 깨짐."""
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert "느껴져" in full or "지친" in full or "힘들" in full
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_round29_still_works():
    engine = build_full_engine()
    list(engine.chat_stream("안녕"))
    s = engine.orchestrator_adapter.stats()
    assert s["total_classifications"] >= 1
    print(f"  ✓ orchestrator 분류 정상")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("내가 누구야?"))
    o2 = list(e2.chat_stream("내가 누구야?"))
    assert o1 == o2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("can_handle", test_can_handle),
        ("meta_user 실제 답", test_meta_user_real_answer),
        ("meta_self 실제 답", test_meta_self_real_answer),
        ("살아있어? 자기 인식", test_meta_self_alive_question),
        ("emotional 진짜 공감", test_emotional_real_empathy),
        ("factual 모름 인정", test_factual_unknown_admits),
        ("위치 → env 양보", test_factual_location_yields_to_env),
        ("causal 반사실", test_causal_what_if_real_answer),
        ("past 회상", test_past_recall_response),
        ("짧은 가르침 학습", test_short_teaching_works),
        ("긴 가르침 거부", test_long_teaching_rejected),
        ("ResponsePlan 반환", test_returns_responseplan),
        ("미지원 상황 None", test_returns_none_for_unhandled),
        ("기존 회귀", test_existing_no_regression),
        ("라운드 29 회귀", test_round29_still_works),
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
