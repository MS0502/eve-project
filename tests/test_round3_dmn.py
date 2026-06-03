"""
라운드 3: DMNAdapter (자발 발화)

확인:
1. DMN 어댑터 없으면 proactive_stream() 빈 yield
2. DMN 켜면 force_spontaneous로 자발 활성 발생
3. proactive_stream() force=True → 응답 chunk 나옴
4. 응답 형식: "그러고 보니 ~ 생각이 났어" 같은 자기 reference
5. 6 어댑터 다 켜도 결정론
6. 자발 발화도 EM에 일화로 저장됨
7. chat_stream도 여전히 작동 (회귀 없음)
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


def make_full_engine(with_dmn=True):
    h = HormoneAdapter()
    a = ActivationAdapter(hormone_adapter=h)
    m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
    n = NLAdapter(hormone_adapter=h, activation_adapter=a)
    s = SDAdapter(hormone_adapter=h, activation_adapter=a)
    d = DMNAdapter(hormone_adapter=h, activation_adapter=a) if with_dmn else None
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
    )


def test_no_dmn_proactive_empty():
    """DMN 없으면 proactive_stream 빈."""
    engine = make_full_engine(with_dmn=False)
    out = list(engine.proactive_stream(force=True))
    assert out == [], f"DMN 없는데 발화: {out}"
    print(f"  ✓ DMN 없을 때 빈 stream")


def test_dmn_seeds_commonsense():
    """commonsense seed로 SA에 페어들 생김."""
    engine = make_full_engine()
    sa = engine.activation_adapter.sa
    # COMMONSENSE_TRIPLES에서 알려진 페어 하나 검사
    w = sa.get_weight("배고프다", "먹다")
    assert w > 0, f"commonsense seed 안 됨, w={w}"
    print(f"  ✓ '배고프다' ↔ '먹다' weight={w:.3f}")


def test_dmn_force_spontaneous_works():
    """DMN.activate_one(force=True) → 자발 활성 결과 나옴."""
    engine = make_full_engine()
    # 카테고리 좀 미리 활성 (DMN이 그 주변으로 wandering)
    engine.activation_adapter.sa.activate("PT", 0.5)
    engine.activation_adapter.sa.activate("운동", 0.5)
    engine.activation_adapter.sa.activate("힘들다", 0.5)

    result = engine.dmn_adapter.force_spontaneous()
    print(f"  force_spontaneous: {result}")
    assert result is not None, "DMN이 카테고리 못 만듦"
    assert "category" in result and "mode" in result


def test_proactive_stream_yields_chunks():
    """proactive_stream(force=True) → 자발 발화 chunk."""
    engine = make_full_engine()
    # 입력 한 번 줘서 SA에 활성 카테고리 깔기
    list(engine.chat_stream("PT 힘들다"))

    out = list(engine.proactive_stream(force=True))
    print(f"  proactive: {out}")
    assert len(out) >= 2, f"자발 발화 너무 짧음: {out}"

    full = " ".join(out)
    # proactive marker — 새 풀에서 어떤 패턴이라도 OK
    proactive_markers = [
        "생각이 났어", "기억이 떠오르네", "떠올랐어",
        "맴돌", "신경 쓰", "스쳐 지나갔어", "곱씹", "끌리네",
        "기억나", "그때가", "생각해보고", "하고 싶",
        # 라운드 70: 통합 자율발화 marker
        "생각나네", "생각나", "떠올라",
    ]
    has_marker = any(m in full for m in proactive_markers)
    assert has_marker, f"자발 발화 marker 없음: {full}"


def test_proactive_records_episode():
    """자발 발화도 EM 일화로 저장."""
    engine = make_full_engine()
    list(engine.chat_stream("PT 힘들다"))
    n_before = len(engine.memory_adapter.em.episodes)

    list(engine.proactive_stream(force=True))
    n_after = len(engine.memory_adapter.em.episodes)
    assert n_after > n_before, f"자발 발화 EM 안 저장: {n_before} → {n_after}"
    print(f"  ✓ 일화 {n_before} → {n_after}")


def test_chat_stream_still_works():
    """DMN 켜도 chat_stream 회귀 없음."""
    engine = make_full_engine()
    out = list(engine.chat_stream("민석아 힘들다"))
    full = " ".join(out)
    assert "지친" in full or "에너지" in full
    print(f"  ✓ chat_stream 정상: {out}")


def test_proactive_determinism():
    """자발 발화도 결정론 (같은 SA 상태 → 같은 출력)."""
    e1 = make_full_engine()
    list(e1.chat_stream("PT 힘들다"))
    out1 = list(e1.proactive_stream(force=True))

    e2 = make_full_engine()
    list(e2.chat_stream("PT 힘들다"))
    out2 = list(e2.proactive_stream(force=True))

    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 자발 발화 결정론: {out1}")


def test_full_demo_scenario():
    """전체 시나리오: 대화 → 시간 흘림 → 자발 발화."""
    engine = make_full_engine()

    print("\n  [Turn 1] 민석아 PT 힘들다")
    out1 = list(engine.chat_stream("민석아 PT 힘들다"))
    for c in out1: print(f"    {c}")

    # 호르몬 시뮬: ot 살짝 올림 (친밀감)
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.7

    print("\n  [Proactive] (외부 입력 없이 자발 발화)")
    out2 = list(engine.proactive_stream(force=True))
    for c in out2: print(f"    {c}")
    assert len(out2) >= 2


def run_all():
    tests = [
        ("DMN 없을 때 빈", test_no_dmn_proactive_empty),
        ("commonsense seed", test_dmn_seeds_commonsense),
        ("DMN 강제 활성", test_dmn_force_spontaneous_works),
        ("proactive yield", test_proactive_stream_yields_chunks),
        ("자발도 EM 저장", test_proactive_records_episode),
        ("chat_stream 회귀", test_chat_stream_still_works),
        ("자발 결정론", test_proactive_determinism),
        ("전체 시나리오", test_full_demo_scenario),
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
