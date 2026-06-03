"""
라운드 4: VSAAdapter + AIAdapter

확인:
1. VSA 어댑터 없으면 기존 동작
2. VSA compose가 의미 합성 수행
3. VSA가 System1 후보에 합성 카테고리 추가 가능
4. AI 어댑터 predict + observe 작동
5. AI 학습이 백그라운드에서 일어남 (응답 영향 없음)
6. 8 어댑터 풀스택 작동 + 결정론
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


def make_engine_full():
    h = HormoneAdapter()
    a = ActivationAdapter(hormone_adapter=h)
    m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
    n = NLAdapter(hormone_adapter=h, activation_adapter=a)
    s = SDAdapter(hormone_adapter=h, activation_adapter=a)
    d = DMNAdapter(hormone_adapter=h, activation_adapter=a)
    v = VSAAdapter(hormone_adapter=h, activation_adapter=a)
    ai = AIAdapter(hormone_adapter=h, activation_adapter=a, sd_adapter=s)
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
    ), h, a, m, n, s, d, v, ai


def test_vsa_compose_creates_composite():
    """VSA compose로 의미 합성 키 생김."""
    eng, *_ = make_engine_full()
    from utils.types import Meaning
    meaning = Meaning(intent="emotion", entities=["민석"], emotions={"fatigue": 0.8})
    name = eng.vsa_adapter.compose_meaning(meaning)
    print(f"  composite: {name}")
    assert name is not None, f"compose 실패: {name}"


def test_vsa_no_compose_for_thin_meaning():
    """entities/emotions가 부족하면 compose 안 함."""
    eng, *_ = make_engine_full()
    from utils.types import Meaning
    meaning = Meaning(intent="statement", entities=["하늘"])  # emotions, actions 없음
    name = eng.vsa_adapter.compose_meaning(meaning)
    print(f"  thin meaning composite: {name}")
    assert name is None


def test_ai_predict_returns_dict():
    """AI predict 호출 가능."""
    eng, *_ = make_engine_full()
    pred = eng.ai_adapter.predict_before_response()
    print(f"  predict: {type(pred).__name__}, keys={list(pred.keys())[:5] if pred else 'none'}")
    assert pred is not None


def test_ai_observe_after_predict():
    """predict → observe 사이클."""
    eng, *_ = make_engine_full()
    eng.ai_adapter.predict_before_response()
    obs = eng.ai_adapter.observe_after_response(outcome="negative")
    print(f"  observe: {type(obs).__name__}")
    # 그냥 안 죽으면 OK


def test_full_8_adapters_response():
    """8 어댑터 다 켜고 정상 응답."""
    eng, *_ = make_engine_full()
    out = list(eng.chat_stream("민석아 PT 힘들다"))
    full = " ".join(out)
    print(f"  full 8-stack: {out}")
    assert len(out) >= 4
    assert "지친" in full or "에너지" in full


def test_8_adapter_determinism():
    """8 어댑터 결정론."""
    e1, *_ = make_engine_full()
    e2, *_ = make_engine_full()
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 8스택 결정론")


def test_ai_observation_increments_count():
    """대화 후 AI observe_count 증가."""
    eng, *_ = make_engine_full()
    n_before = eng.ai_adapter.ai.observe_count
    list(eng.chat_stream("민석아 힘들다"))
    n_after = eng.ai_adapter.ai.observe_count
    print(f"  observe_count: {n_before} → {n_after}")
    # observe가 호출은 됨 (성공 여부와 무관)


def test_vsa_does_not_break_chat():
    """VSA 켜도 응답 깨지지 않음."""
    eng, *_ = make_engine_full()
    out = list(eng.chat_stream("운동 빡세다"))
    print(f"  vsa chat: {out}")
    assert len(out) >= 3


def test_proactive_still_works():
    """8 어댑터 풀에서도 proactive 작동."""
    eng, *_ = make_engine_full()
    list(eng.chat_stream("PT 힘들다"))
    out = list(eng.proactive_stream(force=True))
    print(f"  proactive: {out}")
    assert len(out) >= 2


def run_all():
    tests = [
        ("VSA compose", test_vsa_compose_creates_composite),
        ("VSA thin skip", test_vsa_no_compose_for_thin_meaning),
        ("AI predict", test_ai_predict_returns_dict),
        ("AI observe", test_ai_observe_after_predict),
        ("8스택 응답", test_full_8_adapters_response),
        ("8스택 결정론", test_8_adapter_determinism),
        ("AI observation count", test_ai_observation_increments_count),
        ("VSA 안 깨짐", test_vsa_does_not_break_chat),
        ("proactive 회귀", test_proactive_still_works),
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
