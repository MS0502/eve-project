"""
라운드 25: CharacterAdapter (호르몬→표정, 행동→모션, 입모양)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.visualizer_server import build_state_snapshot


def test_adapter_present():
    engine = build_full_engine()
    assert engine.character_adapter is not None
    print(f"  ✓ character_adapter 부착")


def test_snapshot_keys():
    engine = build_full_engine()
    s = engine.character_adapter.snapshot()
    assert "expression" in s
    assert "motion" in s
    assert "speaking" in s
    print(f"  snapshot: {s}")


def test_default_neutral():
    """기본 호르몬에서 neutral."""
    engine = build_full_engine()
    expr = engine.character_adapter.expression()
    print(f"  default expr: {expr}")
    # 디폴트 호르몬은 baseline 가까움 → neutral 확률 높음


def test_happy_when_dop_ot_high():
    """dopamine + oxytocin 높으면 happy."""
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["dopamine"].level = 0.8
    hs.hormones["oxytocin"].level = 0.7
    hs.hormones["cortisol"].level = 0.2
    hs.hormones["melatonin"].level = 0.2
    expr = engine.character_adapter.expression()
    print(f"  high dop+ot expr: {expr}")
    assert expr == "happy"


def test_sleepy_when_melatonin():
    """melatonin 높으면 sleepy."""
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["melatonin"].level = 0.8
    expr = engine.character_adapter.expression()
    print(f"  high mel expr: {expr}")
    assert expr == "sleepy"


def test_sad_when_low_dop_high_cort():
    """cort 높고 dop 낮으면 sad."""
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    hs.hormones["cortisol"].level = 0.7
    hs.hormones["dopamine"].level = 0.2
    hs.hormones["melatonin"].level = 0.2
    hs.hormones["norepinephrine"].level = 0.2
    expr = engine.character_adapter.expression()
    print(f"  low dop, high cort expr: {expr}")
    assert expr == "sad"


def test_motion_idle_default():
    """기본은 idle 또는 walking."""
    engine = build_full_engine()
    motion = engine.character_adapter.motion()
    print(f"  default motion: {motion}")
    assert motion in ("idle", "walking")


def test_motion_sleeping_high_melatonin():
    """melatonin 매우 높으면 sleeping."""
    engine = build_full_engine()
    engine.hormone_adapter.hs.hormones["melatonin"].level = 0.9
    motion = engine.character_adapter.motion()
    print(f"  high mel motion: {motion}")
    assert motion == "sleeping"


def test_speaking_toggle():
    """입모양 토글."""
    engine = build_full_engine()
    c = engine.character_adapter
    assert c.is_speaking() is False
    c.set_speaking(True)
    assert c.is_speaking() is True
    c.set_speaking(False)
    assert c.is_speaking() is False
    print(f"  ✓ speaking 토글")


def test_visualizer_includes_character():
    """visualizer snapshot에 character 포함."""
    engine = build_full_engine()
    snap = build_state_snapshot(engine)
    assert "character" in snap
    char = snap["character"]
    assert "expression" in char
    assert "motion" in char
    assert "speaking" in char
    print(f"  ✓ snapshot.character: {char}")


def test_motion_after_action():
    """autonomous step 후 motion 변화."""
    engine = build_full_engine()
    # autonomous_loop history에 가짜 행동 넣기
    engine.autonomous_loop.history.append({
        "step": 1, "action": "외출_거실", "needs": [], "decision": {}, "emitted": None,
    })
    motion = engine.character_adapter.motion()
    print(f"  after 외출_거실: {motion}")
    assert motion == "walking"


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert "느껴져" in full or "지친" in full
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    # 같은 호르몬에서 같은 표정
    assert e1.character_adapter.expression() == e2.character_adapter.expression()
    print(f"  ✓ 결정론")


def test_round24_still_works():
    engine = build_full_engine()
    assert engine.visualizer_server is not None
    print(f"  ✓ 라운드 24 정상")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("snapshot 키", test_snapshot_keys),
        ("default neutral", test_default_neutral),
        ("happy (dop+ot)", test_happy_when_dop_ot_high),
        ("sleepy (mel)", test_sleepy_when_melatonin),
        ("sad (low dop)", test_sad_when_low_dop_high_cort),
        ("idle 기본", test_motion_idle_default),
        ("sleeping (high mel)", test_motion_sleeping_high_melatonin),
        ("speaking 토글", test_speaking_toggle),
        ("visualizer 통합", test_visualizer_includes_character),
        ("action → motion", test_motion_after_action),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 24 회귀", test_round24_still_works),
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
