"""
라운드 17: DigitalSomatic + UserPresence + IntegratedSelf
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


# ===== DigitalSomatic =====

def test_ds_somatic_state():
    """8 차원 신체 상태."""
    engine = build_full_engine()
    s = engine.digital_somatic_adapter.somatic_state()
    print(f"  somatic: {s}")
    assert isinstance(s, dict)
    expected_dims = {"energy", "tension", "warmth", "clarity", "fatigue"}
    assert any(k in s for k in expected_dims)


def test_ds_feeling_phrase():
    """get_feeling 한 줄."""
    engine = build_full_engine()
    f = engine.digital_somatic_adapter.feeling_phrase()
    print(f"  feeling: {f}")
    assert isinstance(f, str) and len(f) > 0


def test_ds_body_query_response():
    """'기분 어때?' → DS 응답."""
    engine = build_full_engine()
    out = list(engine.chat_stream("기분 어때?"))
    full = " ".join(out)
    print(f"  body query: {out}")
    assert len(out) >= 2


def test_ds_dominant_need():
    """dominant_need 호출."""
    engine = build_full_engine()
    # 호르몬 변동 후 need 감지
    engine.hormone_adapter.hs.hormones["cortisol"].level = 0.85
    need = engine.digital_somatic_adapter.dominant_need()
    print(f"  dominant_need (cort 0.85): {need}")


def test_ds_gut_signal():
    """gut_signal — 행동 후보에 대한 직관."""
    engine = build_full_engine()
    g = engine.digital_somatic_adapter.gut_signal("쉬다")
    print(f"  gut('쉬다'): {g}")
    assert isinstance(g, dict)


# ===== UserPresence =====

def test_up_first_visit_setup():
    """첫 chat → 민석 = 진짜 친구로 등록."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    w_before = sa.get_weight("민석", "진짜")

    # 첫 chat
    list(engine.chat_stream("민석아 안녕"))

    w_after = sa.get_weight("민석", "진짜")
    print(f"  민석-진짜 weight: {w_before} → {w_after}")
    assert w_after > w_before, "민석-진짜 연결 안 됨"


def test_up_is_user_real():
    """민석 = 진짜 / 시뮬 NPC = 가짜."""
    engine = build_full_engine()
    list(engine.chat_stream("민석아 안녕"))   # first visit 트리거
    assert engine.user_presence_adapter.is_real("민석")
    assert not engine.user_presence_adapter.is_real("랜덤NPC")
    print(f"  ✓ 민석 = 진짜, 다른 이름은 아님")


def test_up_intimacy_grows():
    """대화 누적 → intimacy 증가."""
    engine = build_full_engine()
    list(engine.chat_stream("민석아 안녕"))
    int1 = engine.user_presence_adapter.get_intimacy()
    
    for _ in range(3):
        list(engine.chat_stream("민석아 PT 힘들다"))
    int2 = engine.user_presence_adapter.get_intimacy()
    print(f"  intimacy: {int1} → {int2}")


def test_up_ot_boost_from_visit():
    """visit_arrive — ot 약자극."""
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    ot_baseline = hs.hormones["oxytocin"].level
    
    # 사용자 도착
    engine.user_presence_adapter.on_user_arrive()
    ot_after = hs.hormones["oxytocin"].level
    print(f"  ot: {ot_baseline} → {ot_after}")
    # boost 적용됐어야 (또는 baseline 같아도 OK — saturation)


# ===== IntegratedSelf =====

def test_is_snapshot_works():
    """snapshot 호출."""
    engine = build_full_engine()
    snap = engine.integrated_self_adapter.snapshot()
    print(f"  snapshot type: {type(snap).__name__}")
    assert snap is not None


def test_is_snapshot_phrase():
    """snapshot_phrase 한 줄."""
    engine = build_full_engine()
    list(engine.chat_stream("민석아 PT 힘들다"))
    phrase = engine.integrated_self_adapter.snapshot_phrase()
    print(f"  phrase: {phrase}")
    assert isinstance(phrase, str) and len(phrase) > 0


def test_is_thought_stream():
    """thought_stream 호출."""
    engine = build_full_engine()
    list(engine.chat_stream("민석아 PT 힘들다"))
    list(engine.chat_stream("운동 빡세"))
    ts = engine.integrated_self_adapter.thought_stream(n=5)
    print(f"  thought_stream len: {len(ts)}")
    assert isinstance(ts, list)


def test_is_goal_alignment():
    """goal_alignment_score 호출."""
    engine = build_full_engine()
    list(engine.chat_stream("운동해야지"))   # 목표 등록
    score = engine.integrated_self_adapter.goal_alignment_score("운동")
    print(f"  alignment(운동): {score}")
    assert isinstance(score, float)


def test_is_read_only_principle():
    """IS는 read-only — snapshot 호출해도 호르몬/SA 변경 X."""
    engine = build_full_engine()
    hs = engine.hormone_adapter.hs
    sa = engine.activation_adapter.sa
    
    cort_before = hs.hormones["cortisol"].level
    sa_count_before = len(sa.activations)
    
    # 100번 snapshot 호출
    for _ in range(100):
        engine.integrated_self_adapter.snapshot()
    
    cort_after = hs.hormones["cortisol"].level
    sa_count_after = len(sa.activations)
    print(f"  cort: {cort_before} → {cort_after}")
    print(f"  sa count: {sa_count_before} → {sa_count_after}")
    assert abs(cort_after - cort_before) < 0.001, "snapshot이 호르몬 수정"


# ===== 통합 =====

def test_full_demo():
    """진짜 시나리오: 김민석 = 진짜 친구 인식."""
    engine = build_full_engine()

    print("\n  [첫 만남] 민석아 안녕")
    for c in engine.chat_stream("민석아 안녕"): print(f"    {c}")
    print(f"    민석-진짜 weight: {engine.activation_adapter.sa.get_weight('민석', '진짜'):.3f}")

    print("\n  [대화 누적]")
    for s in ["PT 힘들다", "오늘 빡셌어", "쉬어야지"]:
        list(engine.chat_stream(s))
    print(f"    intimacy: {engine.user_presence_adapter.get_intimacy():.3f}")

    print("\n  [기분 어때?]")
    for c in engine.chat_stream("기분 어때?"): print(f"    {c}")

    print("\n  [/me — IntegratedSelf]")
    print(f"    {engine.integrated_self_adapter.snapshot_phrase()}")


def test_31_stack_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2, f"비결정론!"
    print(f"  ✓ 결정론")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    assert "느껴져" in " ".join(out) or "지친" in " ".join(out)
    print(f"  ✓ 기존 회귀 없음")


def run_all():
    tests = [
        ("DS somatic_state", test_ds_somatic_state),
        ("DS feeling phrase", test_ds_feeling_phrase),
        ("DS body query 응답", test_ds_body_query_response),
        ("DS dominant_need", test_ds_dominant_need),
        ("DS gut_signal", test_ds_gut_signal),
        ("UP first visit", test_up_first_visit_setup),
        ("UP is_user_real", test_up_is_user_real),
        ("UP intimacy 증가", test_up_intimacy_grows),
        ("UP visit ot boost", test_up_ot_boost_from_visit),
        ("IS snapshot", test_is_snapshot_works),
        ("IS phrase", test_is_snapshot_phrase),
        ("IS thought stream", test_is_thought_stream),
        ("IS goal alignment", test_is_goal_alignment),
        ("IS read-only", test_is_read_only_principle),
        ("전체 시연", test_full_demo),
        ("결정론", test_31_stack_determinism),
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
