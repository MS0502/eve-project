"""
라운드 22: WebLearningAdapter + VisionAdapter

requests/wikipedia/youtube-transcript-api/transformers 없어도 graceful degrade.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


# ===== WebLearning =====

def test_web_adapter_present():
    engine = build_full_engine()
    assert engine.web_learning_adapter is not None
    print(f"  ✓ web_learning_adapter 부착")


def test_web_status():
    engine = build_full_engine()
    s = engine.web_learning_adapter.status()
    print(f"  status: {s}")
    assert "requests" in s
    assert "wikipedia" in s
    assert "youtube_transcript" in s


def test_fetch_url_graceful():
    """requests 없을 때 빈 문자열."""
    engine = build_full_engine()
    if not engine.web_learning_adapter.status()["requests"]:
        text = engine.web_learning_adapter.fetch_url("https://example.com")
        assert text == ""
        print(f"  ✓ requests 없음 → graceful")
    else:
        print(f"  requests 있음 (실제 fetch 가능)")


def test_fetch_and_learn_graceful():
    """학습 시도 — 실패해도 dict 반환."""
    engine = build_full_engine()
    r = engine.web_learning_adapter.fetch_and_learn("https://nonexistent-domain-xyz123.com")
    assert isinstance(r, dict)
    assert "url" in r
    assert "learned" in r
    print(f"  ✓ fetch_and_learn graceful: {r}")


def test_wiki_graceful():
    engine = build_full_engine()
    r = engine.web_learning_adapter.wiki_and_learn("파이썬")
    assert isinstance(r, dict)
    print(f"  ✓ wiki graceful: {r}")


def test_youtube_id_extraction():
    """video_id 추출 패턴."""
    engine = build_full_engine()
    w = engine.web_learning_adapter
    # 직접 ID
    assert w._extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # URL 패턴
    assert w._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert w._extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # 잘못된
    assert w._extract_video_id("not a url") is None
    print(f"  ✓ video_id 추출")


def test_youtube_graceful():
    engine = build_full_engine()
    r = engine.web_learning_adapter.youtube_and_learn("invalidID00")
    assert isinstance(r, dict)
    print(f"  ✓ youtube graceful")


def test_strip_html():
    """HTML 제거 가벼운 함수."""
    from adapters.web_learning_adapter import _strip_html
    html = "<html><body><script>alert('x')</script><p>안녕 <b>민석</b>아</p></body></html>"
    text = _strip_html(html)
    assert "안녕" in text
    assert "민석" in text
    assert "alert" not in text
    assert "<" not in text
    print(f"  ✓ strip_html: {text!r}")


def test_safety_whitelist():
    """allowed_domains 화이트리스트."""
    from adapters.web_learning_adapter import WebLearningAdapter
    engine = build_full_engine()
    w = WebLearningAdapter(engine, allowed_domains={"ko.wikipedia.org"})
    
    # 허용
    assert w._is_allowed("https://ko.wikipedia.org/wiki/Python")
    # 차단
    assert not w._is_allowed("https://evil.com/page")
    print(f"  ✓ safety whitelist")


# ===== Vision =====

def test_vision_adapter_present():
    engine = build_full_engine()
    assert engine.vision_adapter is not None
    print(f"  ✓ vision_adapter 부착")


def test_vision_status():
    engine = build_full_engine()
    s = engine.vision_adapter.status()
    print(f"  status: {s}")
    assert "clip" in s
    assert "labels_count" in s
    assert s["labels_count"] > 0


def test_vision_classify_graceful():
    """CLIP 없을 때 빈 결과."""
    engine = build_full_engine()
    if not engine.vision_adapter.available():
        result = engine.vision_adapter.classify_image("/nonexistent.jpg")
        assert result == []
        print(f"  ✓ CLIP 없음 → graceful")
    else:
        print(f"  CLIP 있음 (실제 분류 가능)")


def test_vision_see_and_learn_graceful():
    engine = build_full_engine()
    r = engine.vision_adapter.see_and_learn("/nonexistent.jpg")
    assert isinstance(r, dict)
    assert "labels" in r
    print(f"  ✓ see_and_learn graceful")


def test_vision_add_label():
    """카테고리 추가."""
    engine = build_full_engine()
    n_before = len(engine.vision_adapter.labels)
    engine.vision_adapter.add_label("새카테고리")
    n_after = len(engine.vision_adapter.labels)
    assert n_after == n_before + 1
    # 중복 방지
    engine.vision_adapter.add_label("새카테고리")
    assert len(engine.vision_adapter.labels) == n_after
    print(f"  ✓ add_label")


# ===== 회귀 =====

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


def test_round20_21_still_work():
    engine = build_full_engine()
    s = engine.sensory_adapter.status()
    assert isinstance(s, dict)
    v = engine.voice_loop_adapter.status()
    assert isinstance(v, dict)
    print(f"  ✓ 라운드 20/21 정상")


def run_all():
    tests = [
        ("web 어댑터 부착", test_web_adapter_present),
        ("web status", test_web_status),
        ("fetch graceful", test_fetch_url_graceful),
        ("fetch+learn graceful", test_fetch_and_learn_graceful),
        ("wiki graceful", test_wiki_graceful),
        ("youtube id 추출", test_youtube_id_extraction),
        ("youtube graceful", test_youtube_graceful),
        ("strip_html", test_strip_html),
        ("safety whitelist", test_safety_whitelist),
        ("vision 어댑터 부착", test_vision_adapter_present),
        ("vision status", test_vision_status),
        ("vision graceful", test_vision_classify_graceful),
        ("vision learn graceful", test_vision_see_and_learn_graceful),
        ("vision add_label", test_vision_add_label),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 20/21 회귀", test_round20_21_still_work),
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
