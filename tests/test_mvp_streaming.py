"""
MVP 검증:
- chat("민석아 힘들다") 호출하면 4단계 이상 chunk 나옴
- 동일 입력 → 동일 출력 (결정론)
- LLM/random 의존 0
- 각 chunk에 의미 있는 텍스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.streaming import StreamingEngine


def make_engine():
    """테스트는 sleep 없이 빠르게."""
    return StreamingEngine(delays={"stage1": 0, "core": 0, "expand": 0, "final": 0})


def test_fatigue_4_stage_response():
    """eve1.md 첫 예시: '힘들다' -> 4단계 응답."""
    engine = make_engine()
    chunks = list(engine.chat_stream("민석아 힘들다"))

    assert len(chunks) >= 4, f"4단계 이상 나와야 함, 실제 {len(chunks)}: {chunks}"
    print(f"  ✓ chunks={len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"    [{i}] {c}")

    # stage1 확인 (망설임)
    assert chunks[0] in ("음...", "어...", "잠깐,", "오?"), f"stage1 이상: {chunks[0]}"

    # 핵심 메시지에 "지친" 또는 "에너지" 들어있어야 함
    assert any("지친" in c or "에너지" in c for c in chunks[1:3]), \
        f"core에 fatigue 표현 없음: {chunks}"


def test_determinism():
    """같은 입력 두 번 → 같은 출력 (random/LLM 없음 검증)."""
    e1 = make_engine()
    e2 = make_engine()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\nout1={out1}\nout2={out2}"
    print(f"  ✓ 결정론 확인 ({len(out1)} chunks 동일)")


def test_no_emotion_fallback():
    """감정 없는 입력도 죽지 않고 무언가 응답."""
    engine = make_engine()
    chunks = list(engine.chat_stream("민석아 안녕"))
    assert len(chunks) >= 2, f"최소 stage1+core: {chunks}"
    print(f"  ✓ no-emotion: {chunks}")


def test_question_intent():
    engine = make_engine()
    chunks = list(engine.chat_stream("이거 왜 이래?"))
    assert len(chunks) >= 2
    full = " ".join(chunks)
    print(f"  ✓ question: {full}")


def test_multiple_emotions_picks_strongest():
    engine = make_engine()
    # fatigue (0.8) > sadness (0.8) — fatigue가 코드 순서상 먼저
    # 둘 중 하나의 sub_points가 나와야 함
    chunks = list(engine.chat_stream("힘들고 슬프다"))
    full = " ".join(chunks)
    fatigue_hit = "지친" in full or "에너지" in full or "운동" in full or "피로" in full
    sadness_hit = "가라앉" in full or "마음 무거" in full or "혼자 안고" in full
    assert fatigue_hit or sadness_hit, f"강한 감정 둘 다 무시됨: {chunks}"
    print(f"  ✓ multi-emotion: {full}")


def test_history_recorded():
    engine = make_engine()
    list(engine.chat_stream("민석아 힘들다"))
    list(engine.chat_stream("응 알겠어"))
    assert len(engine.history) == 2
    assert engine.history[0].input_text == "민석아 힘들다"
    print(f"  ✓ 학습 기록 {len(engine.history)}개")


def run_all():
    tests = [
        ("4단계 응답", test_fatigue_4_stage_response),
        ("결정론", test_determinism),
        ("감정 없음 fallback", test_no_emotion_fallback),
        ("질문 의도", test_question_intent),
        ("복수 감정", test_multiple_emotions_picks_strongest),
        ("학습 기록", test_history_recorded),
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
            failed.append(name)
    print(f"\n{'='*40}")
    print(f"{passed}/{len(tests)} pass")
    if failed:
        print(f"실패: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
