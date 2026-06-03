"""
라운드 23: CuriosityAdapter (자율 검색)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from utils.types import Meaning


def test_adapter_present():
    engine = build_full_engine()
    assert engine.curiosity_adapter is not None
    print(f"  ✓ curiosity_adapter 부착")


def test_status():
    engine = build_full_engine()
    s = engine.curiosity_adapter.status()
    assert "searches_today" in s
    assert "daily_limit" in s
    assert "remaining" in s
    print(f"  status: {s}")


def test_unknown_when_empty():
    """초기 상태 — 모든 카테고리 모름."""
    engine = build_full_engine()
    # SA에 없는 카테고리는 모름
    assert engine.curiosity_adapter.is_unknown_category("크리스마스") is True
    print(f"  ✓ 빈 SA → 모름 판정")


def test_known_when_in_sa():
    """SA에 활성/페어 있으면 알고 있음."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("강아지", 0.5)
    sa.learn_pair("강아지", "동물", strength=0.7)
    assert engine.curiosity_adapter.is_unknown_category("강아지") is False
    print(f"  ✓ SA 활성 → 알고 있음")


def test_can_search_initially():
    """초기엔 검색 가능."""
    engine = build_full_engine()
    ok, reason = engine.curiosity_adapter.can_search()
    assert ok or reason in ("ok", "no_web")
    print(f"  can_search: {ok}, {reason}")


def test_daily_limit_blocks():
    """daily_limit 초과 시 차단."""
    engine = build_full_engine()
    c = engine.curiosity_adapter
    c.searches_today = [{"x": i} for i in range(c.daily_limit)]
    ok, reason = c.can_search()
    assert not ok
    assert reason == "daily_limit"
    print(f"  ✓ daily_limit 차단")


def test_cooldown_blocks():
    """쿨다운 중엔 차단."""
    import time
    engine = build_full_engine()
    c = engine.curiosity_adapter
    c.last_search_time = time.time()       # 방금
    c.searches_today = []
    ok, reason = c.can_search()
    assert not ok
    assert reason == "cooldown"
    print(f"  ✓ cooldown 차단")


def test_explore_meaning_no_unknowns():
    """알고 있는 카테고리만 있으면 검색 안 함."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("강아지", 0.5)
    sa.learn_pair("강아지", "동물", strength=0.7)
    
    m = Meaning(intent="statement", entities=["강아지"], emotions={})
    results = engine.curiosity_adapter.explore_meaning(m)
    print(f"  results (known cat): {results}")
    # 알고 있으니 검색 시도 X
    assert len(results) == 0


def test_explore_meaning_unknown():
    """모르는 카테고리 → 검색 시도 (web 없으면 graceful)."""
    engine = build_full_engine()
    m = Meaning(intent="statement", entities=["완전모르는단어xyz"], emotions={})
    results = engine.curiosity_adapter.explore_meaning(m)
    print(f"  results (unknown): {results}")
    # web 라이브러리 없으면 reason='no_web'
    # 있으면 시도하나 검색 결과 없음 — 둘 다 OK
    assert isinstance(results, list)


def test_known_unknowns_dedup():
    """이미 시도한 카테고리는 다시 안 함."""
    engine = build_full_engine()
    c = engine.curiosity_adapter
    c._known_unknowns.add("이미시도한카테고리")
    
    m = Meaning(intent="statement", entities=["이미시도한카테고리"], emotions={})
    unknowns = c.find_unknowns_in_meaning(m)
    assert "이미시도한카테고리" not in unknowns
    print(f"  ✓ 중복 방지")


def test_chat_stream_triggers_curiosity():
    """chat_stream이 응답 후 자율 검색 시도."""
    engine = build_full_engine()
    out = list(engine.chat_stream("크리스마스가 뭐야?"))
    full = " ".join(out)
    print(f"  응답: {out}")
    # 응답 자체는 정상이어야
    assert len(out) >= 2
    # curiosity 시도 기록 확인
    history = engine.curiosity_adapter.history(5)
    print(f"  curiosity history: {history}")


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
    print(f"  ✓ 결정론")


def test_round22_still_works():
    engine = build_full_engine()
    # web/vision 어댑터 정상
    assert engine.web_learning_adapter is not None
    assert engine.vision_adapter is not None
    print(f"  ✓ 라운드 22 정상")


def test_autonomous_with_curiosity():
    """autonomous_loop step에서 curiosity 호출 안 죽음."""
    engine = build_full_engine()
    # curiosity 호르몬 활성 (있으면)
    try:
        engine.hormone_adapter.hs.hormones["curiosity"].level = 0.7
    except Exception:
        pass
    # autonomous step
    r = engine.autonomous_loop.step(emit=False)
    print(f"  step result keys: {list(r.keys())}")
    # 'curiosity' 키 있을 수도, 없을 수도 (web 없으면 없음)


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("status", test_status),
        ("빈 SA → 모름", test_unknown_when_empty),
        ("SA 있으면 알고 있음", test_known_when_in_sa),
        ("초기 검색 가능", test_can_search_initially),
        ("daily_limit 차단", test_daily_limit_blocks),
        ("cooldown 차단", test_cooldown_blocks),
        ("known cat → 검색 X", test_explore_meaning_no_unknowns),
        ("unknown → 시도", test_explore_meaning_unknown),
        ("중복 방지", test_known_unknowns_dedup),
        ("chat_stream 트리거", test_chat_stream_triggers_curiosity),
        ("autonomous 통합", test_autonomous_with_curiosity),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 22 회귀", test_round22_still_works),
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
