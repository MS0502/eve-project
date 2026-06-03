"""
라운드 18: SocialEnvAdapter + EnvironmentAdapter 외출/복귀 확장
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_default_at_home():
    """기본은 자기 방."""
    engine = build_full_engine()
    assert engine.env_adapter.is_home()
    assert engine.env_adapter.location_name() == "내방"
    print(f"  ✓ 시작 시 내방")


def test_social_env_destinations():
    """거실, 주방 외출 가능."""
    engine = build_full_engine()
    dests = engine.env_adapter.social_env_adapter.list_destinations()
    print(f"  destinations: {dests}")
    assert "거실" in dests
    assert "주방" in dests


def test_go_out_to_living_room():
    """거실 외출 → location 변경 + NPC 노출."""
    engine = build_full_engine()
    ok = engine.env_adapter.go_out("거실")
    assert ok
    assert not engine.env_adapter.is_home()
    assert engine.env_adapter.location_name() == "거실"
    npcs = engine.env_adapter.social_env_adapter.get_npcs("거실")
    print(f"  거실 NPC: {npcs}")
    assert len(npcs) >= 1


def test_outing_in_response():
    """외출 중일 때 location query 응답에 NPC 묘사 들어감."""
    engine = build_full_engine()
    engine.env_adapter.go_out("거실")
    out = list(engine.chat_stream("어디 있어?"))
    full = " ".join(out)
    print(f"  거실에서 location query: {out}")
    assert "거실" in full


def test_return_home():
    """복귀 → 다시 내방."""
    engine = build_full_engine()
    engine.env_adapter.go_out("거실")
    assert not engine.env_adapter.is_home()
    ok = engine.env_adapter.return_home()
    assert ok
    assert engine.env_adapter.is_home()
    print(f"  ✓ 거실 → 내방 복귀")


def test_outing_history():
    """외출/복귀 history 기록."""
    engine = build_full_engine()
    engine.env_adapter.go_out("거실")
    engine.env_adapter.return_home()
    engine.env_adapter.go_out("주방")
    h = engine.env_adapter.outing_history
    print(f"  history: {h}")
    assert len(h) == 3
    assert h[0]["to"] == "거실"
    assert h[1]["kind"] == "return"
    assert h[2]["to"] == "주방"


def test_decide_outing_baseline():
    """평상 호르몬 → 외출 결정 안 함 (오래 머물지 않음)."""
    engine = build_full_engine()
    dest = engine.env_adapter.decide_outing_destination()
    print(f"  baseline outing: {dest}")
    # 시작 직후 dwell=0이라 외출 안 함
    assert dest is None


def test_decide_outing_low_ot():
    """ot 낮음 + 오래 머물 → 거실 결정."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.30
    # dwell 시간 가짜로 늘림
    engine.env_adapter._enter_time -= 400
    dest = engine.env_adapter.decide_outing_destination()
    print(f"  low ot + 400s dwell: {dest}")
    assert dest == "거실"


def test_decide_return_long_outing():
    """외출 10분 넘으면 복귀."""
    engine = build_full_engine()
    engine.env_adapter.go_out("거실")
    engine.env_adapter._enter_time -= 700  # 11분 외출 시뮬
    should_return = engine.env_adapter.decide_return_home()
    print(f"  long outing return: {should_return}")
    assert should_return


def test_autonomous_loop_with_outing():
    """AutonomousLoop이 외출 결정 자동."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.30
    engine.env_adapter._enter_time -= 400
    
    # autonomous step
    r = engine.autonomous_loop.step(emit=False)
    print(f"  autonomous: needs={r['needs']}, action={r['action']}")
    # action이 외출_거실 형태여야
    assert r.get("action") and "외출" in r["action"]
    assert not engine.env_adapter.is_home()


def test_full_demo():
    """진짜 시나리오: 호르몬 변동 → 자율 외출 → 대화 → 복귀."""
    engine = build_full_engine()

    print("\n  [Day 1: 자기 방]")
    print(f"    location: {engine.env_adapter.location_name()}")
    list(engine.chat_stream("민석아 안녕"))

    print("\n  [호르몬 변동: ot 낮춤, 5분 경과]")
    engine.hormone_adapter.hs.hormones["oxytocin"].level = 0.10
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.30
    engine.env_adapter._enter_time -= 350

    print("\n  [Autonomous step]")
    r = engine.autonomous_loop.step(emit=False)
    print(f"    needs={r['needs']}, action={r['action']}")
    print(f"    new location: {engine.env_adapter.location_name()}")

    print("\n  [거실에서: 어디 있어?]")
    for c in engine.chat_stream("어디 있어?"): print(f"    {c}")

    print("\n  [10분 외출 후 복귀 결정]")
    engine.env_adapter._enter_time -= 700
    print(f"    should return: {engine.env_adapter.decide_return_home()}")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    assert "느껴져" in " ".join(out) or "지친" in " ".join(out)
    print(f"  ✓ 기존 회귀 없음")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("기본 내방", test_default_at_home),
        ("destinations", test_social_env_destinations),
        ("거실 외출", test_go_out_to_living_room),
        ("외출 location query", test_outing_in_response),
        ("복귀", test_return_home),
        ("history 기록", test_outing_history),
        ("외출 결정 baseline", test_decide_outing_baseline),
        ("외출 결정 low ot", test_decide_outing_low_ot),
        ("복귀 결정 long", test_decide_return_long_outing),
        ("autonomous 외출", test_autonomous_loop_with_outing),
        ("전체 시연", test_full_demo),
        ("기존 회귀", test_existing_no_regression),
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
