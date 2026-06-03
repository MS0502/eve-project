"""
라운드 26: OpenAIServerAdapter (AIRI/외부 OpenAI 클라이언트 연결용)
"""

import sys, os, json, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.openai_server_adapter import (
    _make_chat_response, _make_chat_chunk, OpenAIServerAdapter
)


def test_adapter_present():
    engine = build_full_engine()
    assert engine.openai_server_adapter is not None
    print(f"  ✓ openai_server_adapter 부착")


def test_status():
    engine = build_full_engine()
    s = engine.openai_server_adapter.status()
    assert "running" in s
    assert "host" in s
    assert "port" in s
    assert "endpoints" in s
    assert s["running"] is False     # 기본은 안 돌아감
    print(f"  status: {s}")


def test_make_chat_response_format():
    """OpenAI 호환 non-streaming 응답 포맷."""
    r = _make_chat_response("안녕")
    assert "id" in r
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"]["content"] == "안녕"
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert r["choices"][0]["finish_reason"] == "stop"
    print(f"  ✓ chat.completion 포맷")


def test_make_chat_chunk_format():
    """OpenAI 호환 streaming chunk 포맷."""
    c = _make_chat_chunk("음...")
    assert c["object"] == "chat.completion.chunk"
    assert c["choices"][0]["delta"]["content"] == "음..."
    assert c["choices"][0]["finish_reason"] is None
    # done
    done = _make_chat_chunk("", done=True)
    assert done["choices"][0]["finish_reason"] == "stop"
    print(f"  ✓ chat.completion.chunk 포맷")


def test_handle_chat_no_messages():
    """messages 없으면 400."""
    engine = build_full_engine()
    a = engine.openai_server_adapter
    status, body, is_stream = a._handle_chat_completions({})
    assert status == 400
    assert is_stream is False
    print(f"  ✓ 빈 요청 → 400")


def test_handle_chat_no_user_content():
    """user 메시지 없으면 400."""
    engine = build_full_engine()
    a = engine.openai_server_adapter
    status, body, _ = a._handle_chat_completions({
        "messages": [{"role": "system", "content": "..."}]
    })
    assert status == 400
    print(f"  ✓ user 없음 → 400")


def test_handle_chat_non_streaming():
    """non-streaming 응답."""
    engine = build_full_engine()
    a = engine.openai_server_adapter
    status, body, is_stream = a._handle_chat_completions({
        "messages": [{"role": "user", "content": "민석아 안녕"}],
        "stream": False,
    })
    assert status == 200
    assert is_stream is False
    assert "choices" in body
    content = body["choices"][0]["message"]["content"]
    assert len(content) > 0
    print(f"  응답: {content[:50]}")


def test_handle_chat_streaming():
    """streaming generator 반환."""
    engine = build_full_engine()
    a = engine.openai_server_adapter
    status, gen, is_stream = a._handle_chat_completions({
        "messages": [{"role": "user", "content": "안녕"}],
        "stream": True,
    })
    assert status == 200
    assert is_stream is True
    chunks = list(gen)
    assert len(chunks) >= 2     # 적어도 1 chunk + done + [DONE]
    # 마지막은 [DONE]
    assert "data: [DONE]" in chunks[-1]
    # 각 chunk SSE 포맷
    for c in chunks[:-1]:
        assert c.startswith("data: ")
    print(f"  ✓ SSE chunks: {len(chunks)}개")


def test_handle_models():
    engine = build_full_engine()
    a = engine.openai_server_adapter
    status, body, _ = a._handle_models()
    assert status == 200
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "eve"
    print(f"  ✓ /v1/models")


def test_server_start_stop():
    """진짜 HTTP 서버 시작/중지."""
    engine = build_full_engine()
    # 충돌 방지 — 다른 포트 사용
    a = OpenAIServerAdapter(engine, port=18765)
    ok = a.start_background()
    assert ok is True
    assert a.wait_until_ready(timeout=0.5)
    
    try:
        # /v1/models 호출
        with urllib.request.urlopen(f"http://127.0.0.1:18765/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
            assert data["data"][0]["id"] == "eve"
        print(f"  ✓ /v1/models 응답")
        
        # /v1/chat/completions non-streaming
        req = urllib.request.Request(
            "http://127.0.0.1:18765/v1/chat/completions",
            data=json.dumps({
                "model": "eve",
                "messages": [{"role": "user", "content": "민석아 힘들다"}],
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0
        print(f"  ✓ chat.completions non-stream: {content[:50]}")
        
        # streaming
        req = urllib.request.Request(
            "http://127.0.0.1:18765/v1/chat/completions",
            data=json.dumps({
                "model": "eve",
                "messages": [{"role": "user", "content": "안녕"}],
                "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8")
            assert "data:" in body
            assert "[DONE]" in body
        print(f"  ✓ chat.completions streaming")
        
    finally:
        a.stop()


def test_cors():
    """OPTIONS preflight."""
    engine = build_full_engine()
    a = OpenAIServerAdapter(engine, port=18766)
    assert a.start_background()
    assert a.wait_until_ready(timeout=0.5)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:18766/v1/chat/completions",
            method="OPTIONS",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 204
            assert r.headers.get("Access-Control-Allow-Origin") == "*"
        print(f"  ✓ CORS preflight OK")
    finally:
        a.stop()


def test_health():
    engine = build_full_engine()
    a = OpenAIServerAdapter(engine, port=18767)
    assert a.start_background()
    assert a.wait_until_ready(timeout=0.5)
    try:
        with urllib.request.urlopen("http://127.0.0.1:18767/health", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
            assert data["status"] == "ok"
        print(f"  ✓ /health")
    finally:
        a.stop()




def test_server_port_conflict_does_not_mark_running():
    engine = build_full_engine()
    first = OpenAIServerAdapter(engine, port=18768)
    second = OpenAIServerAdapter(engine, port=18768)
    try:
        assert first.start_background() is True
        assert first.wait_until_ready(timeout=0.5)
        assert second.start_background() is False
        assert second.status()["running"] is False
    finally:
        first.stop()
        second.stop()


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


def test_round25_still_works():
    engine = build_full_engine()
    s = engine.character_adapter.snapshot()
    assert "expression" in s
    print(f"  ✓ 라운드 25 정상")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("status", test_status),
        ("응답 포맷", test_make_chat_response_format),
        ("chunk 포맷", test_make_chat_chunk_format),
        ("빈 요청 400", test_handle_chat_no_messages),
        ("user 없음 400", test_handle_chat_no_user_content),
        ("non-streaming", test_handle_chat_non_streaming),
        ("streaming generator", test_handle_chat_streaming),
        ("/v1/models", test_handle_models),
        ("HTTP 서버 start/stop + 호출", test_server_start_stop),
        ("CORS", test_cors),
        ("/health", test_health),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 25 회귀", test_round25_still_works),
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
