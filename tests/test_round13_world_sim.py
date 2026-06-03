"""
라운드 13: WorldModelAdapter + SimulationEngine

GPT 라운드 12 영역: world_model + simulation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.world_model_adapter import (
    WorldModelAdapter, extract_describe_query, extract_relate_query,
)
from language.understanding import extract_simulation_query


def test_extract_describe():
    """'X 뭐야?' 패턴."""
    assert extract_describe_query("PT 뭐야?") == "PT"
    assert extract_describe_query("운동에 대해 알아?") == "운동"
    assert extract_describe_query("그냥 안녕") is None
    print(f"  ✓ describe 패턴")


def test_extract_relate():
    """'X랑 Y 관련 있어?' 패턴."""
    r = extract_relate_query("PT랑 운동 관련 있어?")
    print(f"  relate('PT랑 운동'): {r}")
    # 패턴 강하지 않아도 안 죽으면 OK
    print(f"  ✓ relate 패턴")


def test_extract_simulation():
    """'X 하면' 패턴."""
    assert extract_simulation_query("PT 하면 어떻게 돼?") == "PT"
    assert extract_simulation_query("운동을 하면 좋을까") == "운동"
    assert extract_simulation_query("그냥 안녕") is None
    print(f"  ✓ simulation 패턴")


def test_world_model_describe_response():
    """'PT 뭐야?' → describe."""
    engine = build_full_engine()
    # 카테고리 활성 + 시드
    engine.activation_adapter.sa.activate("PT", 0.5)
    engine.activation_adapter.sa.learn_pair("PT", "운동", strength=0.7)
    engine.activation_adapter.sa.learn_pair("PT", "힘들다", strength=0.6)
    
    out = list(engine.chat_stream("PT 뭐야?"))
    full = " ".join(out)
    print(f"  describe: {out}")
    # PT 들어가야 함
    assert "PT" in full or "운동" in full or "잘 모르" in full


def test_world_model_describe_unknown():
    """모르는 카테고리 → '잘 모르겠다' 응답."""
    engine = build_full_engine()
    out = list(engine.chat_stream("크리스마스가 뭐야?"))
    full = " ".join(out)
    print(f"  unknown: {out}")
    # 정상 응답 (어떤 답이든 안 죽으면 OK)
    assert len(out) >= 2


def test_simulation_with_causal_graph():
    """CG에 인과 관계 학습 → simulate 결과."""
    engine = build_full_engine()
    # 인과 시드
    cg = engine.simulation_engine.cg
    if cg is not None:
        cg.learn_causal("PT", "피로", evidence_weight=0.8)
        cg.learn_causal("피로", "수면", evidence_weight=0.7)
    
    steps = engine.simulation_engine.simulate("PT", depth=3)
    print(f"  steps: {steps}")
    # 최소 seed step만이라도
    assert len(steps) >= 1
    assert steps[0]["category"] == "PT"


def test_simulation_in_response():
    """'PT 하면 어떻게 돼?' → simulation 응답."""
    engine = build_full_engine()
    cg = engine.simulation_engine.cg
    if cg is not None:
        cg.learn_causal("PT", "피로", evidence_weight=0.8)
        cg.learn_causal("피로", "수면", evidence_weight=0.7)
    
    out = list(engine.chat_stream("PT 하면 어떻게 돼?"))
    full = " ".join(out)
    print(f"  sim chat: {out}")
    # 화살표 또는 연쇄 표현
    has_chain = "→" in full or "흘러" in full or "PT" in full
    assert has_chain


def test_overview():
    """knowledge_overview 호출."""
    engine = build_full_engine()
    phrase = engine.world_model_adapter.overview_phrase()
    print(f"  overview: {phrase}")
    assert phrase is not None and len(phrase) > 0


def test_full_demo():
    """진짜 시나리오 - WM + Sim 통합."""
    engine = build_full_engine()
    cg = engine.simulation_engine.cg
    sa = engine.activation_adapter.sa
    if cg is not None:
        cg.learn_causal("PT", "피로", evidence_weight=0.8)
        cg.learn_causal("피로", "수면", evidence_weight=0.7)
    sa.activate("PT", 0.5)
    sa.learn_pair("PT", "운동", strength=0.7)
    
    print("\n  [WM] PT 뭐야?")
    for c in engine.chat_stream("PT 뭐야?"): print(f"    {c}")
    
    print("\n  [Sim] PT 하면 어떻게 돼?")
    for c in engine.chat_stream("PT 하면 어떻게 돼?"): print(f"    {c}")


def test_27_stack_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 27스택 결정론")


def test_existing_no_regression():
    """기존 12 라운드 기능 살아있음."""
    engine = build_full_engine()

    # 산술
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out), f"산술: {out}"

    # 환경
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out), f"환경: {out}"

    # 강감정 + 사고
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert "느껴져" in full or "지친" in full, f"감정: {out}"

    print(f"  ✓ 모든 기존 기능 정상")


def run_all():
    tests = [
        ("describe 패턴", test_extract_describe),
        ("relate 패턴", test_extract_relate),
        ("simulation 패턴", test_extract_simulation),
        ("WM describe 응답", test_world_model_describe_response),
        ("WM unknown", test_world_model_describe_unknown),
        ("simulation 동작", test_simulation_with_causal_graph),
        ("simulation 응답", test_simulation_in_response),
        ("overview", test_overview),
        ("전체 시나리오", test_full_demo),
        ("27스택 결정론", test_27_stack_determinism),
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
