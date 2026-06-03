"""
라운드 21: VoiceLoopAdapter (자동 음성 대화)

sounddevice/silero-vad/webrtcvad 없어도 graceful degrade.
EVE 코어 절대 안 깨짐.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_voice_loop_adapter_present():
    """voice_loop_adapter 부착됨."""
    engine = build_full_engine()
    assert engine.voice_loop_adapter is not None
    print(f"  ✓ voice_loop_adapter 존재")


def test_status_works():
    """status() — 라이브러리 없어도 안 죽음."""
    engine = build_full_engine()
    s = engine.voice_loop_adapter.status()
    print(f"  status: {s}")
    assert "audio_io" in s
    assert "vad" in s
    assert "stt" in s
    assert "tts" in s
    assert isinstance(s["audio_io"], bool)
    assert isinstance(s["vad"], bool)


def test_audio_graceful_degrade():
    """sounddevice 없을 때 안 죽음."""
    engine = build_full_engine()
    v = engine.voice_loop_adapter
    if not v.audio_available():
        # 라이브러리 없을 때 record가 None 반환
        result = v.record_until_silence()
        assert result is None
        print(f"  ✓ audio 라이브러리 없음 → graceful")
    else:
        print(f"  audio 가능 (sounddevice 설치됨)")


def test_vad_graceful_degrade():
    """VAD 없을 때 안 죽음."""
    engine = build_full_engine()
    v = engine.voice_loop_adapter
    available = v.vad_available()
    print(f"  VAD available: {available}")
    # available 여부와 무관하게 안 죽음


def test_play_wav_graceful():
    """play_wav — 파일 없어도 안 죽음."""
    engine = build_full_engine()
    result = engine.voice_loop_adapter.play_wav("/nonexistent.wav")
    assert isinstance(result, bool)
    print(f"  ✓ play_wav graceful")


def test_chat_once_graceful():
    """chat_once — 어떤 단계 실패해도 dict 반환."""
    engine = build_full_engine()
    result = engine.voice_loop_adapter.chat_once()
    assert isinstance(result, dict)
    assert "recorded" in result
    assert "transcribed" in result
    assert "response" in result
    print(f"  ✓ chat_once graceful: {result}")


def test_chat_stream_voice_yields_dict():
    """chat_stream_voice — 실패해도 dict yield."""
    engine = build_full_engine()
    items = list(engine.voice_loop_adapter.chat_stream_voice())
    assert len(items) >= 1
    for item in items:
        assert isinstance(item, dict)
        assert "kind" in item
    print(f"  ✓ stream graceful: {len(items)} items")


def test_existing_no_regression():
    """기존 EVE 회귀."""
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


def test_sensory_round20_still_works():
    """라운드 20 회귀."""
    engine = build_full_engine()
    s = engine.sensory_adapter.status()
    assert isinstance(s, dict)
    print(f"  ✓ 라운드 20 sensory 정상")


def run_all():
    tests = [
        ("어댑터 부착", test_voice_loop_adapter_present),
        ("status 동작", test_status_works),
        ("audio graceful", test_audio_graceful_degrade),
        ("VAD graceful", test_vad_graceful_degrade),
        ("play_wav graceful", test_play_wav_graceful),
        ("chat_once graceful", test_chat_once_graceful),
        ("stream graceful", test_chat_stream_voice_yields_dict),
        ("기존 회귀", test_existing_no_regression),
        ("결정론", test_determinism),
        ("라운드 20 회귀", test_sensory_round20_still_works),
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
