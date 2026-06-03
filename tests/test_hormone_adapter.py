"""
HormoneAdapter 통합 검증.

확인:
1. 어댑터 없을 때 = 기존 동작 (하위호환)
2. 어댑터 있을 때 호르몬 dict에 26종
3. 평상 호르몬 → 응답 동일
4. cort↑ ot↓ (스트레스) → calm 톤 강제 + 망설임 "잠깐,"
5. ot↑ cort↓ (따뜻함) → warm 톤 강제 + 망설임 "어..."
6. 에너지 낮음 (da, ne 낮음) → sub_points 감소
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.streaming import StreamingEngine
from adapters.hormone_adapter import HormoneAdapter


def make_engine(adapter=None):
    return StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0},
        hormone_adapter=adapter,
    )


def test_no_adapter_backward_compat():
    """어댑터 없으면 기존 v41 동작 유지."""
    eng = make_engine()
    chunks = list(eng.chat_stream("민석아 힘들다"))
    assert chunks[0] == "음...", f"기본 stage1: {chunks[0]}"
    print(f"  ✓ no-adapter: {chunks}")


def test_adapter_exposes_26_hormones():
    """어댑터 켜면 워크스페이스 호르몬 dict에 26종 다 있음."""
    adapter = HormoneAdapter()
    eng = make_engine(adapter)
    eng.chat_stream("테스트").__next__()  # workspace.update 트리거
    h = eng.workspace.hormone
    # v40 26종 + v41 합성 키 (stress/energy/curiosity)
    assert "cortisol" in h and "oxytocin" in h and "dopamine" in h
    assert "stress" in h and "energy" in h
    assert len(h) >= 26
    print(f"  ✓ {len(h)}개 호르몬 노출")


def test_baseline_hormones_no_change():
    """평상 호르몬(default 0.3 등)일 때 응답이 어댑터 없을 때와 거의 같음."""
    eng_off = make_engine()
    eng_on = make_engine(HormoneAdapter())
    out_off = list(eng_off.chat_stream("민석아 힘들다"))
    out_on = list(eng_on.chat_stream("민석아 힘들다"))
    # core message 같아야 함
    assert out_off[1] == out_on[1], f"평상시 core 다름:\n off={out_off}\n on={out_on}"
    print(f"  ✓ 평상시 core 동일: {out_on[1]}")


def test_stress_forces_calm_tone():
    """cort↑↑ ot↓↓ → tone calm + stage1 '잠깐,'"""
    adapter = HormoneAdapter()
    adapter.hs.hormones["cortisol"].level = 0.85
    adapter.hs.hormones["oxytocin"].level = 0.10
    eng = make_engine(adapter)
    chunks = list(eng.chat_stream("민석아 안녕"))   # 감정 없는 입력
    assert chunks[0] == "잠깐,", f"stress stage1: {chunks[0]}"
    final = chunks[-1]
    # calm 톤의 finalize: "조금 쉬어가도 괜찮아."
    assert "쉬어가도 괜찮아" in final, f"stress finalize: {final}"
    print(f"  ✓ 스트레스: {chunks}")


def test_warmth_forces_warm_tone():
    """ot↑↑ cort↓↓ → tone warm + stage1 '어...'"""
    adapter = HormoneAdapter()
    adapter.hs.hormones["oxytocin"].level = 0.85
    adapter.hs.hormones["cortisol"].level = 0.20
    eng = make_engine(adapter)
    chunks = list(eng.chat_stream("민석아 안녕"))
    assert chunks[0] == "어...", f"warmth stage1: {chunks[0]}"
    final = chunks[-1]
    assert "옆에 있을게" in final, f"warmth finalize: {final}"
    print(f"  ✓ 따뜻함: {chunks}")


def test_low_energy_shortens_sub_points():
    """cort 높음 (스트레스) → 길이 줄어듦 (length_decider 결정).
    
    이전엔 hormone sub_point_budget이 직접 자름. 이제는 length_decider가
    cort 0.7 넘으면 -1줄 보정.
    """
    adapter = HormoneAdapter()
    adapter.hs.hormones["cortisol"].level = 0.85
    adapter.hs.hormones["oxytocin"].level = 0.10
    eng = make_engine(adapter)

    # fatigue 자극 (sub_points 원래 2개 → length_decider가 cort↑로 줄임)
    chunks = list(eng.chat_stream("민석아 힘들다"))
    print(f"  high cort + fatigue: {len(chunks)} chunks: {chunks}")
    # cort 높음 → 짧아짐. 평상보다 적어야 함.
    assert len(chunks) <= 4, f"high cort 인데 안 줄음: {len(chunks)}"


def test_normal_energy_keeps_sub_points():
    """평상 에너지 → sub_points 유지 (fatigue=2개)."""
    adapter = HormoneAdapter()  # default: da=0.3, ne=0.3
    eng = make_engine(adapter)
    chunks = list(eng.chat_stream("민석아 힘들다"))
    assert len(chunks) == 5, f"normal: {chunks}"
    print(f"  ✓ 평상 에너지: {len(chunks)} chunks")


def test_determinism_with_adapter():
    """어댑터 켜도 결정론 유지."""
    a1 = HormoneAdapter()
    a1.hs.hormones["cortisol"].level = 0.85
    a1.hs.hormones["oxytocin"].level = 0.10
    a2 = HormoneAdapter()
    a2.hs.hormones["cortisol"].level = 0.85
    a2.hs.hormones["oxytocin"].level = 0.10
    out1 = list(make_engine(a1).chat_stream("민석아 힘들다"))
    out2 = list(make_engine(a2).chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 어댑터 결정론 유지")


def run_all():
    tests = [
        ("hormone-off 회귀", test_no_adapter_backward_compat),
        ("26종 노출", test_adapter_exposes_26_hormones),
        ("평상 호르몬 동일", test_baseline_hormones_no_change),
        ("스트레스 → calm", test_stress_forces_calm_tone),
        ("따뜻함 → warm", test_warmth_forces_warm_tone),
        ("저에너지 → 짧아짐", test_low_energy_shortens_sub_points),
        ("평상 에너지 유지", test_normal_energy_keeps_sub_points),
        ("결정론 유지", test_determinism_with_adapter),
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
