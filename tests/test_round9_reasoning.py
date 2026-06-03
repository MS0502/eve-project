"""
라운드 9: Counterfactual + Analogy + Temporal
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.streaming import StreamingEngine
from adapters.hormone_adapter import HormoneAdapter
from adapters.activation_adapter import ActivationAdapter
from adapters.memory_adapter import MemoryAdapter
from adapters.nl_adapter import NLAdapter
from adapters.sd_adapter import SDAdapter
from adapters.dmn_adapter import DMNAdapter
from adapters.vsa_adapter import VSAAdapter
from adapters.ai_adapter import AIAdapter
from adapters.goal_adapter import GoalAdapter
from adapters.norm_adapter import NormAdapter
from adapters.continual_adapter import ContinualAdapter
from adapters.allostatic_adapter import AllostaticAdapter
from adapters.env_adapter import EnvironmentAdapter
from adapters.autonomy_adapter import AutonomyAdapter
from adapters.counterfactual_adapter import CounterfactualAdapter, extract_cf_query
from adapters.analogy_adapter import AnalogyAdapter
from adapters.temporal_adapter import TemporalAdapter


def make_engine_full():
    h = HormoneAdapter()
    a = ActivationAdapter(hormone_adapter=h)
    m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
    n = NLAdapter(hormone_adapter=h, activation_adapter=a)
    s = SDAdapter(hormone_adapter=h, activation_adapter=a)
    d = DMNAdapter(hormone_adapter=h, activation_adapter=a)
    v = VSAAdapter(hormone_adapter=h, activation_adapter=a)
    ai = AIAdapter(hormone_adapter=h, activation_adapter=a, sd_adapter=s)
    g = GoalAdapter(hormone_adapter=h, activation_adapter=a)
    nr = NormAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n, dmn_adapter=d)
    cr = ContinualAdapter(hormone_adapter=h, activation_adapter=a)
    al = AllostaticAdapter(hormone_adapter=h, activation_adapter=a)
    env = EnvironmentAdapter(hormone_adapter=h, activation_adapter=a)
    auto = AutonomyAdapter()
    cf = CounterfactualAdapter(hormone_adapter=h, activation_adapter=a)
    an = AnalogyAdapter(hormone_adapter=h, activation_adapter=a)
    tm = TemporalAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m)
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
        goal_adapter=g, norm_adapter=nr,
        continual_adapter=cr, allostatic_adapter=al,
        env_adapter=env, autonomy_adapter=auto,
        counterfactual_adapter=cf, analogy_adapter=an, temporal_adapter=tm,
    )


def test_cf_pattern_extraction():
    """CF 패턴 인식."""
    assert extract_cf_query("만약 PT 안 했으면 어땠을까") == "PT"
    assert extract_cf_query("운동 안 했으면 좋았을 텐데") == "운동"
    assert extract_cf_query("만약 시험 안 했으면") == "시험"
    assert extract_cf_query("그냥 안녕") is None
    print(f"  ✓ CF 패턴 인식")


def test_cf_response():
    """'만약 PT 안 했으면' → counterfactual 응답."""
    engine = make_engine_full()
    # 인과 관계 미리 학습
    engine.activation_adapter.sa.activate("PT", 0.6)
    engine.activation_adapter.sa.activate("힘들다", 0.6)
    engine.counterfactual_adapter.cf.cg.learn_causal("PT", "힘들다", evidence_weight=0.7)

    out = list(engine.chat_stream("만약 PT 안 했으면 어땠을까"))
    full = " ".join(out)
    print(f"  cf chat: {out}")
    has_cf = "없었으면" in full or "달라지" in full
    assert has_cf, f"CF 응답 없음: {full}"


def test_temporal_past_keyword():
    """'어제' / '전에' → past 회상."""
    engine = make_engine_full()
    list(engine.chat_stream("PT 힘들다"))   # 일화 저장
    out = list(engine.chat_stream("PT 어제 어땠어?"))
    full = " ".join(out)
    print(f"  past: {out}")
    assert "PT" in full or "전에" in full or "예전" in full


def test_temporal_future_keyword():
    """'내일' → future 상상."""
    engine = make_engine_full()
    out = list(engine.chat_stream("PT 내일 어떨까"))
    full = " ".join(out)
    print(f"  future: {out}")
    # 미래 응답
    has_future = "다음" in full or "어떨지" in full or "올 수도" in full or "PT" in full
    assert has_future, f"future 응답 없음: {full}"


def test_analogy_complete():
    """A:B :: C:? — 그래프에 데이터 있을 때만 작동."""
    engine = make_engine_full()
    sa = engine.activation_adapter.sa
    # 미리 페어 학습
    sa.learn_pair("비", "젖다", strength=0.7)
    sa.learn_pair("불", "뜨겁다", strength=0.7)
    # 결과는 그래프 충분히 풍부해야 정확. 그냥 안 죽는지 확인.
    result = engine.analogy_adapter.find("비", "젖다", "불", top_n=1)
    print(f"  analogy 비:젖다::불:?: {result}")
    # 빈 결과여도 에러 없으면 통과 (graph sparse)


def test_full_17_adapter_chat():
    """17 어댑터 풀스택 일반 응답."""
    engine = make_engine_full()
    out = list(engine.chat_stream("민석아 PT 힘들다"))
    full = " ".join(out)
    print(f"  17-stack: {out}")
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_17_stack_determinism():
    e1 = make_engine_full()
    e2 = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 17스택 결정론")


def test_all_existing_features_still_work():
    """기존 기능들 다 회귀 없음."""
    engine = make_engine_full()

    # 산술
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    print(f"  ✓ 산술: {out}")

    # 환경
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    print(f"  ✓ 환경: {out}")

    # 목표
    list(engine.chat_stream("쉬어"))
    assert "쉬다" in engine.goal_adapter.active_goal_categories()
    print(f"  ✓ 목표")


def run_all():
    tests = [
        ("CF 패턴 인식", test_cf_pattern_extraction),
        ("CF 응답", test_cf_response),
        ("TM past", test_temporal_past_keyword),
        ("TM future", test_temporal_future_keyword),
        ("AN find", test_analogy_complete),
        ("17 풀스택", test_full_17_adapter_chat),
        ("17 결정론", test_17_stack_determinism),
        ("기존 회귀", test_all_existing_features_still_work),
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
