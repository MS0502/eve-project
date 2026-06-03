"""
라운드 6: TaskSolver + Tools (eve1.md 12절 코드 생성 루프)

확인:
1. Calculator 안전 평가 (eval/exec 0)
2. 위험한 식 → ToolError
3. LU가 산술 표현 추출
4. chat_stream "3 + 5는 뭐야" → "8이야" 응답
5. 한국어 연산자 ("3 더하기 5")
6. StringLength 도구
7. Solver: 도구 없으면 fallback, 있으면 실행
8. 결정론 유지
9. task 없는 일반 입력은 그대로 동작 (회귀 없음)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import Calculator, StringLength, ToolError, safe_eval_arith, ToolRegistry
from learning.code_synthesis import TaskSolver
from language.understanding import LanguageUnderstanding, extract_task
from language.streaming import StreamingEngine


def make_engine():
    return StreamingEngine(delays={"stage1": 0, "core": 0, "expand": 0, "final": 0})


def test_safe_eval_basic():
    assert safe_eval_arith("3 + 5") == 8
    assert safe_eval_arith("10 - 4 * 2") == 2
    assert safe_eval_arith("2 ** 10") == 1024
    assert safe_eval_arith("-5 + 3") == -2
    print(f"  ✓ basic arith")


def test_safe_eval_blocks_unsafe():
    """이름/함수/속성/문자열 다 막힘."""
    for bad in ["__import__('os')", "open('x')", "x + 1", "'abc'", "[1,2,3]"]:
        try:
            safe_eval_arith(bad)
            assert False, f"{bad} 막혀야 함"
        except ToolError:
            pass
    print(f"  ✓ unsafe expressions all blocked")


def test_calculator_tool():
    calc = Calculator()
    assert calc.can_handle({"type": "arithmetic", "expr": "3 + 5"})
    out = calc.run({"type": "arithmetic", "expr": "3 + 5"})
    assert "8" in out, f"결과 없음: {out}"
    print(f"  ✓ Calculator: '{out}'")


def test_calculator_handles_zerodiv():
    calc = Calculator()
    try:
        calc.run({"type": "arithmetic", "expr": "5 / 0"})
        assert False
    except ToolError as e:
        print(f"  ✓ 0/0: {e}")


def test_string_length_tool():
    sl = StringLength()
    assert sl.can_handle({"type": "string_length", "target": "안녕"})
    out = sl.run({"type": "string_length", "target": "안녕"})
    assert "2" in out
    print(f"  ✓ StringLength: '{out}'")


def test_extract_arith_korean():
    """LU가 한국어 연산자 인식."""
    t = extract_task("3 더하기 5는 뭐야")
    assert t == {"type": "arithmetic", "expr": "3 + 5"}, f"{t}"
    t = extract_task("10 곱하기 4")
    assert t == {"type": "arithmetic", "expr": "10 * 4"}, f"{t}"
    t = extract_task("그냥 안녕")
    assert t is None
    print(f"  ✓ 한국어 연산자")


def test_task_solver_finds_tool():
    solver = TaskSolver()
    r = solver.solve({"type": "arithmetic", "expr": "3 + 5"})
    assert r["ok"] is True
    assert "8" in r["result"]
    assert r["tool"] == "calculator"
    assert r["attempts"] == 1
    print(f"  ✓ solver: {r}")


def test_task_solver_no_tool():
    solver = TaskSolver()
    r = solver.solve({"type": "unknown_thing"})
    assert r["ok"] is False
    assert r["tool"] is None
    print(f"  ✓ solver fallback: {r}")


def test_chat_stream_arith():
    """진짜 시나리오: '3 + 5는 뭐야' → '8이야' 끼임."""
    engine = make_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    full = " ".join(out)
    print(f"  arith chat: {out}")
    assert "8" in full, f"결과 없음: {full}"


def test_chat_stream_normal_no_task_regression():
    """task 없는 입력 정상 동작 (회귀 없음)."""
    engine = make_engine()
    out = list(engine.chat_stream("민석아 힘들다"))
    full = " ".join(out)
    print(f"  normal: {out}")
    assert "지친" in full or "에너지" in full


def test_arith_determinism():
    e1 = make_engine()
    e2 = make_engine()
    o1 = list(e1.chat_stream("3 더하기 5"))
    o2 = list(e2.chat_stream("3 더하기 5"))
    assert o1 == o2
    print(f"  ✓ 결정론: {o1}")


def test_full_stack_with_task():
    """11 어댑터 풀스택에서도 task 작동."""
    from main import build_full_engine
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    full = " ".join(out)
    print(f"  full stack arith: {out}")
    assert "8" in full


def run_all():
    tests = [
        ("safe_eval basic", test_safe_eval_basic),
        ("safe_eval blocks unsafe", test_safe_eval_blocks_unsafe),
        ("Calculator tool", test_calculator_tool),
        ("Calculator zerodiv", test_calculator_handles_zerodiv),
        ("StringLength tool", test_string_length_tool),
        ("LU 한국어 연산자", test_extract_arith_korean),
        ("Solver 도구 매칭", test_task_solver_finds_tool),
        ("Solver fallback", test_task_solver_no_tool),
        ("chat 산술", test_chat_stream_arith),
        ("normal 회귀", test_chat_stream_normal_no_task_regression),
        ("산술 결정론", test_arith_determinism),
        ("풀스택+task", test_full_stack_with_task),
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
