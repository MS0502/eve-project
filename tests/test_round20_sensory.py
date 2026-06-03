"""
라운드 20: SensoryAdapter (시각/청각 가벼운 버전)

Whisper/Piper/Tesseract 라이브러리 없어도 graceful degrade.
EVE 코어 절대 안 깨짐.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_sensory_adapter_present():
    """sensory_adapter 부착됨."""
    engine = build_full_engine()
    assert engine.sensory_adapter is not None
    print(f"  ✓ sensory_adapter 존재")


def test_status_works_without_libs():
    """status() — 라이브러리 없어도 안 죽음."""
    engine = build_full_engine()
    s = engine.sensory_adapter.status()
    print(f"  status: {s}")
    assert "stt" in s and "tts" in s and "ocr" in s
    # bool 값들이어야
    assert isinstance(s["stt"], bool)
    assert isinstance(s["tts"], bool)
    assert isinstance(s["ocr"], bool)


def test_stt_graceful_degrade():
    """STT 라이브러리 없을 때 빈 문자열 반환."""
    engine = build_full_engine()
    if not engine.sensory_adapter.stt_available():
        result = engine.sensory_adapter.transcribe("/nonexistent.wav")
        assert result == ""
        print(f"  ✓ STT 라이브러리 없음 → graceful degrade")
    else:
        print(f"  STT 가능 (Whisper 설치됨)")


def test_tts_graceful_degrade():
    """TTS 라이브러리 없을 때 None 반환."""
    engine = build_full_engine()
    if not engine.sensory_adapter.tts_available():
        result = engine.sensory_adapter.synthesize("안녕")
        assert result is None
        print(f"  ✓ TTS 라이브러리 없음 → graceful degrade")
    else:
        print(f"  TTS 가능 (Piper 설치됨)")


def test_ocr_graceful_degrade():
    """OCR 라이브러리 없을 때 빈 문자열 반환."""
    engine = build_full_engine()
    if not engine.sensory_adapter.ocr_available():
        result = engine.sensory_adapter.read_image("/nonexistent.jpg")
        assert result == ""
        print(f"  ✓ OCR 라이브러리 없음 → graceful degrade")
    else:
        print(f"  OCR 가능 (Tesseract 설치됨)")


def test_learn_integration():
    """listen_and_learn / read_and_learn — corpus 통합."""
    engine = build_full_engine()
    # 라이브러리 없어도 안 죽고 ""만 반환
    r = engine.sensory_adapter.listen_and_learn("/nonexistent.wav")
    assert r == ""
    r = engine.sensory_adapter.read_and_learn("/nonexistent.jpg")
    assert r == ""
    print(f"  ✓ 학습 통합 - 실패해도 안 죽음")


def test_voice_chat_graceful():
    """voice_chat — STT 실패해도 빈 결과 반환."""
    engine = build_full_engine()
    user_text, chunks, wav = engine.sensory_adapter.voice_chat("/nonexistent.wav")
    assert user_text == ""
    assert chunks == []
    assert wav is None
    print(f"  ✓ voice_chat graceful")


def test_existing_no_regression():
    """기존 EVE 기능 회귀 확인."""
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


def test_mock_text_through_corpus():
    """STT 시뮬: 텍스트를 직접 corpus 학습 → EVE 인식."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    n_before = len(sa.activations)
    
    # STT가 있다면 transcribe 결과를 corpus로 흘림
    # 라이브러리 없어도 corpus는 작동하니 직접 시뮬:
    fake_transcript = "PT 진짜 힘들었어 오늘"
    if engine.corpus_adapter:
        engine.corpus_adapter.learn_text(fake_transcript)
    
    n_after = len(sa.activations)
    print(f"  카테고리: {n_before} → {n_after}")
    assert n_after > n_before


def run_all():
    tests = [
        ("어댑터 부착", test_sensory_adapter_present),
        ("status 동작", test_status_works_without_libs),
        ("STT graceful", test_stt_graceful_degrade),
        ("TTS graceful", test_tts_graceful_degrade),
        ("OCR graceful", test_ocr_graceful_degrade),
        ("학습 통합", test_learn_integration),
        ("voice_chat graceful", test_voice_chat_graceful),
        ("STT → corpus 시뮬", test_mock_text_through_corpus),
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
