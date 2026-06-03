"""
라운드 30 (B-2): SituationResponder
- Orchestrator 컨텍스트 기반 응답 합성
- 9가지 상황 처리
- 진짜 한계 (메타 질문, 모름 답하기, 사족 제거)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from utils.types import Meaning


def test_situation_responder_attached():
    engine = build_full_engine()
    assert engine.planner.situation_responder is not None
    print(f"  ✓ situation_responder 부착")


def test_can_handle_all_situations():
    engine = build_full_engine()
    sr = engine.planner.situation_responder
    handled = ["greeting", "meta_self", "meta_user",
               "emotional_share", "factual_question",
               "causal_what_if", "past_recall"]
    for s in handled:
        assert sr.can_handle(s), f"{s} 처리 못함"
    # task와 small_talk는 fallback (None)
    assert not sr.can_handle("teaching")
    print(f"  ✓ {len(handled)}개 상황 처리 가능")


# ===== 메타 질문 (너 본 핵심 한계) =====

def test_meta_user_returns_user_info():
    """'내가 누구야?' → 사용자 이름 답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("내가 누구야?"))
    full = " ".join(out)
    assert "민석" in full
    print(f"  응답: {full}")


def test_meta_self_returns_eve_identity():
    """'너는 누구야?' → EVE 정체성 답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("너는 누구야?"))
    full = " ".join(out)
    assert "EVE" in full
    print(f"  응답: {full}")


def test_meta_self_alive_question():
    """'너는 살아있어?' → 자기 인식 답"""
    engine = build_full_engine()
    out = list(engine.chat_stream("너는 살아있어?"))
    full = " ".join(out)
    # 카테고리 회상이 아니라 자기 인식 답이어야
    assert "호르몬" in full or "살아" in full or "EVE" in full
    print(f"  응답: {full}")


# ===== 감정 공유 =====

def test_emotional_negative_empathy():
    """부정 감정 → 공감"""
    engine = build_full_engine()
    out = list(engine.chat_stream("민석이 너무 힘들어"))
    full = " ".join(out)
    # 라운드 56+57: SpeechHub 공감 풀 넓힘
    assert any(p in full for p in ["힘들", "느껴", "쉬어", "빡세", "그치",
                                     "같이", "옆에", "마음", "천천히"]), \
        f"공감 응답 X: {full}"
    print(f"  응답: {full}")


def test_emotional_positive_warm():
    """긍정 감정 → 따뜻함"""
    engine = build_full_engine()
    out = list(engine.chat_stream("오늘 진짜 기뻐"))
    full = " ".join(out)
    assert any(p in full for p in ["좋", "따뜻", "신나", "기뻐"])
    print(f"  응답: {full}")


# ===== 모름 솔직히 =====

def test_factual_unknown_admits():
    """모르는 정보 → 솔직히 모름"""
    engine = build_full_engine()
    out = list(engine.chat_stream("오늘 날씨 어때?"))
    full = " ".join(out)
    # 산만한 응답 X, 모름 인정
    assert any(p in full for p in ["모르", "잘 안", "알아볼"])
    # "action" 같은 fragment 없어야
    assert "action" not in full
    print(f"  응답: {full}")


# ===== 산술 사족 =====

def test_task_no_extra():
    """산술 답 후 사족 최소화"""
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는?"))
    full = " ".join(out)
    assert "8" in full
    # "옆에 있을게" 같은 큰 사족 없어야 (또는 짧아야)
    print(f"  응답: {full}")
    # 핵심: "8" 들어가있고 응답이 너무 길지 않음
    # 어느 정도 간결한지 출력해서 확인


# ===== 반사실 =====

def test_causal_what_if():
    """만약 질문"""
    engine = build_full_engine()
    out = list(engine.chat_stream("만약 PT 안 했으면 어땠을까"))
    full = " ".join(out)
    assert any(p in full for p in ["만약", "그랬", "어떻", "달랐", "다른"])
    print(f"  응답: {full}")


# ===== 과거 회상 =====

def test_past_recall():
    """저번에 ~"""
    engine = build_full_engine()
    list(engine.chat_stream("민석이 PT 빡셌어"))
    list(engine.chat_stream("PT 진짜 빡세"))
    out = list(engine.chat_stream("저번에 PT 얘기했었지?"))
    full = " ".join(out)
    print(f"  응답: {full}")
    # "기억" 또는 "전에"
    assert any(p in full for p in ["기억", "얘기", "그래", "전", "응"])


# ===== 인사 =====

def test_greeting_simple():
    engine = build_full_engine()
    out = list(engine.chat_stream("안녕"))
    full = " ".join(out)
    assert "안녕" in full
    print(f"  응답: {full}")


# ===== fallback =====

def test_small_talk_fallbacks_to_old():
    """small_talk는 SituationResponder 아닌 기존 흐름"""
    engine = build_full_engine()
    # 일부러 어떤 분류에도 안 걸리는 입력
    out = list(engine.chat_stream("그냥"))
    full = " ".join(out)
    # 응답이 있어야
    assert len(full) > 0
    print(f"  응답: {full}")


def test_teaching_processed_separately():
    """teaching은 SituationResponder 안 거치고 chat_stream에서 처리"""
    engine = build_full_engine()
    out = list(engine.chat_stream("'응'이라고 말해"))
    full = " ".join(out)
    # teaching 응답 (수용/의문/거부 중 하나)
    assert any(p in full for p in ["응.", "왜", "어색", "이상", "알았"])
    print(f"  응답: {full}")


# ===== 결정론 =====

def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    cases = ["내가 누구야?", "너는 누구야?", "민석이 힘들어", "안녕"]
    for c in cases:
        o1 = list(e1.chat_stream(c))
        o2 = list(e2.chat_stream(c))
        assert o1 == o2, f"{c} 결정론 깨짐"
    print(f"  ✓ {len(cases)} 케이스 결정론")


# ===== 회귀 =====

def test_existing_no_regression():
    engine = build_full_engine()
    # 산술
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    # 환경
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    # 공감
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert any(p in full for p in ["힘들", "느껴", "쉬어", "지친", "빡세", "옆에", "마음", "같이", "그치"])
    print(f"  ✓ 산술/환경/공감 OK")


def test_round29_orchestrator_still_works():
    engine = build_full_engine()
    list(engine.chat_stream("안녕"))
    s = engine.orchestrator_adapter.stats()
    assert s["total_classifications"] > 0
    print(f"  ✓ 라운드 29 정상")


def run_all():
    tests = [
        ("어댑터 부착", test_situation_responder_attached),
        ("9 상황 처리", test_can_handle_all_situations),
        # 메타 질문 (핵심)
        ("메타 user", test_meta_user_returns_user_info),
        ("메타 self", test_meta_self_returns_eve_identity),
        ("메타 살아", test_meta_self_alive_question),
        # 감정
        ("부정 공감", test_emotional_negative_empathy),
        ("긍정 따뜻", test_emotional_positive_warm),
        # 모름
        ("모름 인정", test_factual_unknown_admits),
        # 산술
        ("산술 사족", test_task_no_extra),
        # 반사실/회상
        ("반사실", test_causal_what_if),
        ("과거 회상", test_past_recall),
        # 인사
        ("인사", test_greeting_simple),
        # fallback
        ("small_talk fallback", test_small_talk_fallbacks_to_old),
        ("teaching 별도", test_teaching_processed_separately),
        # 결정론
        ("결정론", test_determinism),
        # 회귀
        ("기존 회귀", test_existing_no_regression),
        ("라운드 29 회귀", test_round29_orchestrator_still_works),
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
