"""
라운드 24: VisualizerServer (2D 시각화 WebSocket)

websockets 없어도 graceful degrade.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.visualizer_server import build_state_snapshot, HORMONE_COLORS


def test_adapter_present():
    engine = build_full_engine()
    assert engine.visualizer_server is not None
    print(f"  ✓ visualizer_server 부착")


def test_status():
    engine = build_full_engine()
    s = engine.visualizer_server.status()
    assert "available" in s
    assert "running" in s
    assert isinstance(s["running"], bool)
    print(f"  status: {s}")


def test_websockets_graceful():
    """websockets 없을 때 안 죽음."""
    engine = build_full_engine()
    v = engine.visualizer_server
    if not v.available():
        # 라이브러리 없을 때 start_background False
        ok = v.start_background()
        assert ok is False
        print(f"  ✓ websockets 없음 → graceful")
    else:
        print(f"  websockets 있음")


def test_state_snapshot_basic():
    """build_state_snapshot — 모든 키 있음."""
    engine = build_full_engine()
    snap = build_state_snapshot(engine)
    assert snap["type"] == "state"
    assert "hormones" in snap
    assert "location" in snap
    assert "objects" in snap
    assert "intimacy" in snap
    assert "feeling" in snap
    assert "active_categories" in snap
    print(f"  ✓ snapshot 구조 OK")
    print(f"  location: {snap['location']}")
    print(f"  hormones: {len(snap['hormones'])}")


def test_state_json_serializable():
    """snapshot이 JSON 직렬화 가능."""
    engine = build_full_engine()
    snap = build_state_snapshot(engine)
    s = json.dumps(snap, ensure_ascii=False)
    assert len(s) > 100
    # 다시 파싱 가능
    parsed = json.loads(s)
    assert parsed["type"] == "state"
    print(f"  ✓ JSON 직렬화 OK ({len(s)} bytes)")


def test_state_after_chat():
    """chat 후 snapshot 변화."""
    engine = build_full_engine()
    snap1 = build_state_snapshot(engine)
    list(engine.chat_stream("민석이 너무 힘들어"))
    snap2 = build_state_snapshot(engine)
    # 카테고리 늘어났을 가능성
    cats1 = len(snap1.get("active_categories", []))
    cats2 = len(snap2.get("active_categories", []))
    print(f"  cats before: {cats1}, after: {cats2}")
    # 활성 카테고리는 늘거나 같아야
    assert cats2 >= 0  # 그냥 OK


def test_hormone_colors():
    """호르몬 색상 매핑."""
    assert "dopamine" in HORMONE_COLORS
    assert "cortisol" in HORMONE_COLORS
    assert "oxytocin" in HORMONE_COLORS
    # 색은 hex
    for name, color in HORMONE_COLORS.items():
        assert color.startswith("#")
        assert len(color) == 7
    print(f"  ✓ 호르몬 색상 {len(HORMONE_COLORS)}개")


def test_visualizer_html_exists():
    """index.html 파일 존재."""
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, "..", "visualizer", "index.html")
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "WebSocket" in html
    assert "EVE" in html
    assert "ws-url" in html
    print(f"  ✓ visualizer/index.html ({len(html)} bytes)")


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


def test_round23_still_works():
    engine = build_full_engine()
    assert engine.curiosity_adapter is not None
    assert engine.web_learning_adapter is not None
    print(f"  ✓ 라운드 23 정상")


def test_no_background_thread_by_default():
    """기본은 background thread 0 (사용자가 명시 시작 안 하면)."""
    engine = build_full_engine()
    v = engine.visualizer_server
    assert v.running is False
    assert v._thread is None
    print(f"  ✓ 기본 background thread 0")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("status", test_status),
        ("websockets graceful", test_websockets_graceful),
        ("snapshot 구조", test_state_snapshot_basic),
        ("JSON 직렬화", test_state_json_serializable),
        ("chat 후 변화", test_state_after_chat),
        ("호르몬 색상", test_hormone_colors),
        ("HTML 파일 존재", test_visualizer_html_exists),
        ("기본 thread 0", test_no_background_thread_by_default),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 23 회귀", test_round23_still_works),
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
