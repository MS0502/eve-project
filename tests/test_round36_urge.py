"""
라운드 36: UrgeAdapter (발화 충동) + DMN 문장 다양화
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.urge_adapter import (
    UrgeAdapter, URGE_THRESHOLD,
    W_DOPAMINE, W_NOREPI, W_LONELINESS, W_TIME_SINCE,
)


# ===== UrgeAdapter 단독 =====

def test_urge_no_engine():
    u = UrgeAdapter(None)
    urge = u.compute_urge(0)
    assert urge == 0.5, f"engine 없으면 중립 0.5, 실제: {urge}"
    print(f"  ✓ engine 없으면 중립 (0.5)")


def test_urge_with_hormones():
    """호르몬 변화 시 urge 변화"""
    from main import build_full_engine
    engine = build_full_engine()
    u = UrgeAdapter(engine)
    
    # 도파민 높이면 urge 증가
    engine.hormone_adapter.hs.hormones["dopamine"].level = 0.9
    urge_high_da = u.compute_urge(0)
    
    # 도파민 낮으면 urge 감소
    engine.hormone_adapter.hs.hormones["dopamine"].level = 0.1
    urge_low_da = u.compute_urge(0)
    
    assert urge_high_da > urge_low_da, \
        f"도파민 높이면 urge 더 큼: {urge_high_da} vs {urge_low_da}"
    print(f"  ✓ 도파민 높음→urge 높음 ({urge_high_da:.2f} > {urge_low_da:.2f})")


def test_urge_loneliness():
    """oxytocin 낮으면 외로움 → urge 증가"""
    from main import build_full_engine
    engine = build_full_engine()
    u = UrgeAdapter(engine)
    
    # 친구 옆 있음 (oxytocin 높음)
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.9
    urge_with_friend = u.compute_urge(0)
    
    # 외로움 (oxytocin 낮음)
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.1
    urge_lonely = u.compute_urge(0)
    
    assert urge_lonely > urge_with_friend, \
        f"외로우면 urge 더 큼: {urge_lonely} vs {urge_with_friend}"
    print(f"  ✓ 외로움→urge 높음 ({urge_lonely:.2f} > {urge_with_friend:.2f})")


def test_urge_time_pressure():
    """시간 누적 → urge 증가"""
    from main import build_full_engine
    engine = build_full_engine()
    u = UrgeAdapter(engine)
    u.last_emit_ts = 0
    
    # 1초 후
    urge1 = u.compute_urge(1.0)
    # 60초 후
    urge60 = u.compute_urge(60.0)
    
    assert urge60 > urge1, f"시간 더 지나면 urge 더 큼"
    print(f"  ✓ 시간 누적→urge 증가 ({urge1:.2f} → {urge60:.2f})")


def test_should_emit():
    """should_emit boolean"""
    from main import build_full_engine
    engine = build_full_engine()
    u = UrgeAdapter(engine)
    
    # 모든 호르몬 0.5 (중립) → urge ≈ 0.5
    for h in engine.hormone_adapter.hs.hormones.values():
        h.level = 0.5
    
    # 시간 60초 누적 (충분한 압박)
    u.last_emit_ts = 0
    
    assert u.should_emit(60.0) is True
    print(f"  ✓ should_emit (충분한 시간 + 중립 호르몬)")


def test_mark_emitted():
    u = UrgeAdapter(None)
    assert u.last_emit_ts == 0
    u.mark_emitted(100.0)
    assert u.last_emit_ts == 100.0
    print(f"  ✓ mark_emitted")


def test_stats():
    from main import build_full_engine
    engine = build_full_engine()
    u = UrgeAdapter(engine)
    u.compute_urge(0)
    u.compute_urge(60)
    s = u.stats()
    assert s["urge_checks"] == 2
    assert "last_urge" in s
    assert "last_components" in s
    print(f"  ✓ stats {s}")


# ===== DMN 문장 다양화 =====

def test_proactive_prefix_variety():
    """_PROACTIVE_PREFIXES_BY_MODE 풀에서 다양 선택"""
    from language.planner import ResponsePlanner
    p = ResponsePlanner()
    
    # mind_wandering 8개 변형
    pool = p._PROACTIVE_PREFIXES_BY_MODE["mind_wandering"]
    assert len(pool) >= 5, f"mind_wandering 풀 ≥ 5, 실제 {len(pool)}"
    
    total = sum(len(v) for v in p._PROACTIVE_PREFIXES_BY_MODE.values())
    assert total >= 20, f"전체 풀 ≥ 20, 실제 {total}"
    print(f"  ✓ proactive 패턴 풀 {total}개 (mind_wandering {len(pool)}개)")


def test_proactive_rotation():
    """카테고리 다르면 다른 패턴 (다양성). 같은 카테고리 = 같은 (결정론)."""
    from language.planner import ResponsePlanner
    from utils.types import Meaning
    
    p = ResponsePlanner()
    
    # 다양한 카테고리 — 다양한 패턴
    cats = ["친구", "공부", "민석", "PT", "맛있다", "날씨", "고양이", "음악"]
    seen_prefixes = []
    for cat in cats:
        m = Meaning(intent="proactive", entities=[], emotions={})
        plan = p._build_proactive(m, cat, "mind_wandering")
        seen_prefixes.append(plan.core_message)
    
    unique = len(set(seen_prefixes))
    assert unique >= 3, f"다양한 카테고리에서 다양 표현, 실제: {unique}"
    
    # 결정론 — 같은 카테고리 = 같은 패턴
    m1 = Meaning(intent="proactive", entities=[], emotions={})
    plan1 = p._build_proactive(m1, "친구", "mind_wandering")
    plan2 = p._build_proactive(m1, "친구", "mind_wandering")
    assert plan1.core_message == plan2.core_message, "결정론 깨짐"
    
    print(f"  ✓ {len(cats)}개 카테고리 → {unique}개 표현 + 결정론 보존")


# ===== 통합 (engine + live_loop urge 사용) =====

def test_live_loop_uses_urge():
    """live_loop가 urge 체크 후 발화"""
    from main import build_full_engine
    engine = build_full_engine()
    
    # urge 임계 못 넘게 — 모든 호르몬 0.0
    for h in engine.hormone_adapter.hs.hormones.values():
        h.level = 0.0
    engine.urge_adapter.last_emit_ts = time.time()  # 막 발화한 셈
    
    urge = engine.urge_adapter.compute_urge(time.time())
    assert urge < URGE_THRESHOLD, f"호르몬 0이면 urge < 임계, 실제 {urge}"
    
    print(f"  ✓ 호르몬 0 + 막 발화 → urge {urge:.2f} < 임계 {URGE_THRESHOLD}")


# ===== 결정론 =====

def test_determinism():
    from main import build_full_engine
    e1 = build_full_engine()
    e2 = build_full_engine()
    
    # 같은 호르몬 상태
    for h in e1.hormone_adapter.hs.hormones.values():
        h.level = 0.5
    for h in e2.hormone_adapter.hs.hormones.values():
        h.level = 0.5
    
    u1 = UrgeAdapter(e1)
    u2 = UrgeAdapter(e2)
    
    urge1 = u1.compute_urge(30.0)
    urge2 = u2.compute_urge(30.0)
    assert urge1 == urge2
    print(f"  ✓ 결정론 (urge {urge1})")


def run_all():
    tests = [
        ("engine 없으면 중립", test_urge_no_engine),
        ("호르몬 변화→urge", test_urge_with_hormones),
        ("외로움→urge", test_urge_loneliness),
        ("시간 누적→urge", test_urge_time_pressure),
        ("should_emit", test_should_emit),
        ("mark_emitted", test_mark_emitted),
        ("stats", test_stats),
        ("DMN 패턴 풀", test_proactive_prefix_variety),
        ("DMN 회전", test_proactive_rotation),
        ("live_loop urge 사용", test_live_loop_uses_urge),
        ("결정론", test_determinism),
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
