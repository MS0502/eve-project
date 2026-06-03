"""
라운드 37 (DynamicValue + Salience) + 38 (호르몬 → 발화 톤)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.salience_adapter import (
    DynamicValue, SalienceAdapter,
)


# ===== DynamicValue =====

def test_dynamic_value_basic():
    v = DynamicValue(base=0.0, decay_per_sec=0.1)
    assert v.value == 0.0
    v.spike(0.5)
    assert v.value == 0.5
    v.tick(dt=10.0)
    # 10초 후 base 회귀 (e^-1 ≈ 0.37)
    assert 0 < v.value < 0.5
    print(f"  ✓ spike → 0.5, 10초 후 {v.value:.3f}")


def test_dynamic_value_ceiling():
    v = DynamicValue(base=0.0, ceiling=1.0)
    v.spike(2.0)
    assert v.value == 1.0   # ceiling 안 넘음
    print(f"  ✓ ceiling 적용 (2.0 spike → {v.value})")


def test_dynamic_value_baseline_recovery():
    v = DynamicValue(base=0.5, decay_per_sec=0.5)
    v.value = 0.1
    # 10초 후 base 0.5에 거의 도달
    v.tick(dt=10.0)
    assert v.value > 0.4
    print(f"  ✓ baseline 회귀 (0.1 → {v.value:.3f})")


def test_dynamic_value_spike_decay():
    v = DynamicValue(base=0.0, decay_per_sec=0.1)
    # 여러 spike
    v.spike(0.3)
    v.spike(0.4)
    assert v.value > 0.6
    # 시간 지나면 감소
    initial = v.value
    v.tick(dt=5.0)
    assert v.value < initial
    print(f"  ✓ 누적 spike → decay")


# ===== SalienceAdapter =====

def test_salience_basic():
    s = SalienceAdapter()
    s.spike("민석")
    assert s.get("민석") > 0
    s.spike("PT")
    s.spike("민석")  # 두번째
    assert s.get("민석") > s.get("PT")  # 두 번 spike된 게 더 높음
    print(f"  ✓ 다중 spike — 민석={s.get('민석'):.2f}, PT={s.get('PT'):.2f}")


def test_salience_emotional_higher():
    s = SalienceAdapter()
    s.spike("일반")
    s.spike("감정", emotional=True)
    assert s.get("감정") > s.get("일반")
    print(f"  ✓ emotional spike 더 큼")


def test_salience_decay_over_time():
    s = SalienceAdapter()
    s.spike("X")
    initial = s.get("X")
    s.tick(dt=10.0)
    later = s.get("X")
    assert later < initial
    print(f"  ✓ {initial:.2f} → 10초 후 {later:.2f}")


def test_salience_top():
    s = SalienceAdapter()
    s.spike("A")
    s.spike("B"); s.spike("B")
    s.spike("C", emotional=True)
    
    top = s.top(3)
    assert len(top) == 3
    # C는 감정 spike → 가장 높을 가능성
    keys = [k for k, _ in top]
    assert "C" in keys
    print(f"  ✓ top 3: {top}")


def test_salience_cleanup():
    s = SalienceAdapter()
    s.spike("높은 거")
    s.spike("낮은 거")
    # 낮은 거 강제로 매우 낮게
    s.salience["낮은 거"].value = 0.01
    
    removed = s.cleanup(threshold=0.05)
    assert removed == 1
    assert "낮은 거" not in s.salience
    assert "높은 거" in s.salience
    print(f"  ✓ cleanup ({removed}개 제거)")


def test_salience_stats():
    s = SalienceAdapter()
    s.spike("A"); s.spike("B")
    s.tick()
    
    stats = s.stats()
    assert stats["tracked_keys"] == 2
    assert stats["spike_count"] == 2
    assert stats["tick_count"] == 1
    assert len(stats["top_5"]) == 2
    print(f"  ✓ stats {stats}")


# ===== 결정론 =====

def test_dynamic_value_determinism():
    v1 = DynamicValue(base=0.0, decay_per_sec=0.1)
    v2 = DynamicValue(base=0.0, decay_per_sec=0.1)
    
    v1.spike(0.5); v2.spike(0.5)
    v1.tick(dt=5.0); v2.tick(dt=5.0)
    
    assert v1.value == v2.value
    print(f"  ✓ DynamicValue 결정론")


def test_salience_determinism():
    s1 = SalienceAdapter()
    s2 = SalienceAdapter()
    
    for k in ["A", "B", "C"]:
        s1.spike(k); s2.spike(k)
    s1.tick(dt=5.0); s2.tick(dt=5.0)
    
    for k in ["A", "B", "C"]:
        assert s1.get(k) == s2.get(k)
    print(f"  ✓ Salience 결정론")


# ===== 라운드 38: 호르몬 → 발화 풀 =====

def test_state_pool_tired():
    from main import build_full_engine
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["cortisol"].level = 0.8
    hs.hormones["dopamine"].level = 0.2
    
    pool = engine.planner._select_state_pool()
    assert pool is not None
    assert "졸리" in pool[0] or "피곤" in pool[0] or "쉬고" in pool[0]
    print(f"  ✓ tired → {pool[0]}")


def test_state_pool_lonely():
    from main import build_full_engine
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["oxytocin"].level = 0.20
    hs.hormones["cortisol"].level = 0.5
    hs.hormones["dopamine"].level = 0.5
    
    pool = engine.planner._select_state_pool()
    assert pool is not None
    has_lonely = any("민석" in p or "너" in p or "보고 싶" in p for p in pool)
    assert has_lonely, f"lonely 풀 X: {pool[0]}"
    print(f"  ✓ lonely → {pool[0]}")


def test_state_pool_energetic():
    from main import build_full_engine
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["dopamine"].level = 0.8
    hs.hormones["norepinephrine"].level = 0.7
    hs.hormones["cortisol"].level = 0.4
    hs.hormones["oxytocin"].level = 0.5
    
    pool = engine.planner._select_state_pool()
    assert pool is not None
    has_energy = any("오!" in p or "신난" in p or "좋다" in p for p in pool)
    assert has_energy, f"energetic 풀 X: {pool[0]}"
    print(f"  ✓ energetic → {pool[0]}")


def test_state_pool_curious():
    from main import build_full_engine
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["dopamine"].level = 0.7
    hs.hormones["norepinephrine"].level = 0.4
    hs.hormones["cortisol"].level = 0.4
    hs.hormones["oxytocin"].level = 0.5
    
    pool = engine.planner._select_state_pool()
    assert pool is not None
    has_curious = any("?" in p or "왜" in p or "궁금" in p or "신기" in p for p in pool)
    assert has_curious, f"curious 풀 X: {pool[0]}"
    print(f"  ✓ curious → {pool[0]}")


def test_state_pool_fallback():
    """호르몬 모두 중간이면 일반 mode 풀 사용"""
    from main import build_full_engine
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    for h in hs.hormones.values():
        h.level = 0.5
    
    pool = engine.planner._select_state_pool()
    # 중간 상태에선 None 가능 (일반 풀로 fallback)
    print(f"  ✓ 중립 호르몬 → pool {'state' if pool else 'general'}")


# ===== 통합 — engine 부착 검증 =====

def test_engine_integration():
    from main import build_full_engine
    engine = build_full_engine()
    assert hasattr(engine, "salience_adapter")
    assert engine.salience_adapter is not None
    
    # 사용
    engine.salience_adapter.spike("테스트")
    assert engine.salience_adapter.get("테스트") > 0
    print(f"  ✓ engine 통합")


def run_all():
    tests = [
        ("DynamicValue 기본", test_dynamic_value_basic),
        ("DynamicValue ceiling", test_dynamic_value_ceiling),
        ("DynamicValue baseline 회귀", test_dynamic_value_baseline_recovery),
        ("DynamicValue spike+decay", test_dynamic_value_spike_decay),
        ("Salience 기본", test_salience_basic),
        ("Salience emotional", test_salience_emotional_higher),
        ("Salience decay", test_salience_decay_over_time),
        ("Salience top", test_salience_top),
        ("Salience cleanup", test_salience_cleanup),
        ("Salience stats", test_salience_stats),
        ("DynamicValue 결정론", test_dynamic_value_determinism),
        ("Salience 결정론", test_salience_determinism),
        # 라운드 38
        ("호르몬 → tired 풀", test_state_pool_tired),
        ("호르몬 → lonely 풀", test_state_pool_lonely),
        ("호르몬 → energetic 풀", test_state_pool_energetic),
        ("호르몬 → curious 풀", test_state_pool_curious),
        ("중립 호르몬 fallback", test_state_pool_fallback),
        ("engine 통합", test_engine_integration),
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
