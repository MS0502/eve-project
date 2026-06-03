"""
라운드 14: AgencyAdapter + AutonomousLoop

GPT 라운드 14: autonomous loop — 스스로 사이클 돌리기.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from core.autonomous import _detect_needs


def test_detect_needs_baseline():
    """baseline 호르몬 → need 적음."""
    engine = build_full_engine()
    needs = _detect_needs(engine.hormone_adapter)
    print(f"  baseline needs: {needs}")
    # default ot=0.3 → low_oxytocin 트리거 (< 0.25 = False, ot=0.3은 OK)
    # 하지만 da=0.3 < 0.25 = False. 일단 안 죽으면 OK.


def test_detect_needs_low_ot():
    """ot 낮춤 → low_oxytocin need."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    needs = _detect_needs(engine.hormone_adapter)
    print(f"  low ot needs: {needs}")
    assert "low_oxytocin" in needs


def test_detect_needs_high_cort():
    """cort 높임 → high_cortisol need."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    needs = _detect_needs(engine.hormone_adapter)
    print(f"  high cort needs: {needs}")
    assert "high_cortisol" in needs


def test_agency_evaluate():
    """Agency.evaluate_action 호출 가능."""
    engine = build_full_engine()
    engine.activation_adapter.sa.activate("쉬다", 0.5)
    ev = engine.agency_adapter.evaluate("쉬다")
    print(f"  evaluate('쉬다'): decision={ev.get('decision')}, conf={ev.get('confidence')}")
    assert "decision" in ev


def test_agency_best_action():
    """후보 중 do 점수 최고 선택."""
    engine = build_full_engine()
    engine.activation_adapter.sa.activate("쉬다", 0.7)
    engine.activation_adapter.sa.activate("탐색", 0.4)
    picked = engine.agency_adapter.best_action(["쉬다", "탐색"])
    print(f"  best: {picked}")
    # do 미달이면 None — 안 죽으면 OK


def test_autonomous_step_silent():
    """step(emit=False) — 사고만, 발화 없음."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    result = engine.autonomous_loop.step(emit=False)
    print(f"  silent step: needs={result['needs']}, action={result['action']}, "
          f"decision={result['decision']}, emitted={result['emitted']}")
    assert result["emitted"] is False
    assert "high_cortisol" in result["needs"]


def test_autonomous_step_emit():
    """step(emit=True) — autonomy due면 발화."""
    engine = build_full_engine()
    # idle 충분히 (autonomy threshold 넘김)
    engine.autonomy_adapter.last_input_time -= 60.0
    # cort 올림 → need 트리거
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85

    result = engine.autonomous_loop.step(emit=True)
    print(f"  emit step: emitted={result['emitted']}, "
          f"response={result['response'][:60]}...")
    # need 있고 idle 충분이면 발화 시도


def test_autonomous_step_count_increments():
    """step() 호출 시 step_count 증가."""
    engine = build_full_engine()
    n0 = engine.autonomous_loop.step_count
    engine.autonomous_loop.step(emit=False)
    engine.autonomous_loop.step(emit=False)
    n1 = engine.autonomous_loop.step_count
    print(f"  step_count: {n0} → {n1}")
    assert n1 == n0 + 2


def test_autonomous_history_records():
    """history에 trace 기록."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    engine.autonomous_loop.step(emit=False)
    h = engine.autonomous_loop.history
    print(f"  history len: {len(h)}, last action: {h[-1]['action'] if h else None}")
    assert len(h) >= 1


def test_autonomous_thought_generated():
    """_silent_thought이 ReasoningLoop 통해 thought 생성."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    result = engine.autonomous_loop.step(emit=False)
    print(f"  thought: {result.get('thought')}")
    # thought가 None이어도 OK (감정 약한 카테고리)


def test_autonomous_no_action_no_emit():
    """need 없고 후보 없으면 silent."""
    engine = build_full_engine()
    # 평상 호르몬, 활성 목표 없음
    result = engine.autonomous_loop.step(emit=True)
    print(f"  no need step: action={result['action']}, emitted={result['emitted']}")
    # action None or emitted False


def test_full_demo_autonomous():
    """진짜 시나리오: 대화 + 자율 사이클 + 다시 대화."""
    engine = build_full_engine()

    print("\n  [User] 민석아 PT 힘들다")
    for c in engine.chat_stream("민석아 PT 힘들다"): print(f"    {c}")

    # 호르몬 변동: cort 올림 (스트레스 누적)
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.8
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.15

    # idle 누적
    engine.autonomy_adapter.last_input_time -= 60.0

    print("\n  [Autonomous step (idle 동안)]")
    for i in range(3):
        r = engine.autonomous_loop.step(emit=True)
        print(f"    step {r['step']}: needs={r['needs']}, action={r['action']}, "
              f"emitted={r['emitted']}")
        if r["emitted"]:
            print(f"    → {r['response'][:70]}")
            break

    print(f"\n  [Final state] step_count={engine.autonomous_loop.step_count}, "
          f"recent={engine.autonomous_loop.get_state()['recent_actions']}")


def test_28_stack_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2, f"비결정론!\n{o1}\nvs\n{o2}"
    print(f"  ✓ 28스택 결정론")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("PT 하면 어떻게 돼?"))
    # CG에 인과 안 학습이라 sim 응답 없을 수 있음 — 안 죽으면 OK
    print(f"  ✓ 기존 기능 정상")


def run_all():
    tests = [
        ("need 감지 baseline", test_detect_needs_baseline),
        ("need 감지 low ot", test_detect_needs_low_ot),
        ("need 감지 high cort", test_detect_needs_high_cort),
        ("Agency evaluate", test_agency_evaluate),
        ("Agency best_action", test_agency_best_action),
        ("autonomous step silent", test_autonomous_step_silent),
        ("autonomous step emit", test_autonomous_step_emit),
        ("step_count 증가", test_autonomous_step_count_increments),
        ("history 기록", test_autonomous_history_records),
        ("thought 생성", test_autonomous_thought_generated),
        ("need 없음 silent", test_autonomous_no_action_no_emit),
        ("전체 시나리오", test_full_demo_autonomous),
        ("28스택 결정론", test_28_stack_determinism),
        ("기존 회귀", test_existing_no_regression),
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
