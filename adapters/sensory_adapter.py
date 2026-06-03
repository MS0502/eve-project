"""
SensoryAdapter — 라운드 20 가벼운 시각/청각

EVE 코어는 LLM 0 유지. 입출력 모듈은 인식/합성 모델 사용 OK (LLM 아님).

지원:
- STT (Whisper) — 한국어 음성 → 텍스트
- TTS (Piper) — 텍스트 → 한국어 음성
- OCR (Tesseract) — 이미지 속 텍스트
- (미래) CLIP — 이미지 → 카테고리 (라운드 21+)

원칙:
- Lazy load — 사용할 때만 로드 (EVE 코어 시작 무겁지 않게)
- Optional dependencies — 라이브러리 없으면 graceful degrade
- 결과는 corpus_adapter로 흘러서 EVE 학습 (텍스트 = 카테고리 페어)
- EVE 결정론 안 깸 (인식만, 사고는 EVE가)

사용:
    sensory = SensoryAdapter(engine)
    text = sensory.listen()  # 마이크 → STT
    sensory.speak("안녕")     # TTS → 스피커
    text = sensory.read_image("path.jpg")  # OCR
"""

from typing import Optional


class SensoryAdapter:

    def __init__(self, engine):
        self.engine = engine
        # Lazy: 처음 사용 시 로드
        self._stt = None
        self._tts = None
        self._ocr = None

    # ===== STT (Whisper) =====

    def _load_stt(self, model_name: str = "small"):
        """Whisper.cpp 또는 faster-whisper. 둘 다 시도."""
        if self._stt is not None:
            return self._stt
        try:
            # 우선 faster-whisper (가벼움)
            from faster_whisper import WhisperModel
            self._stt = WhisperModel(model_name, device="cpu", compute_type="int8")
            self._stt_kind = "faster_whisper"
            return self._stt
        except ImportError:
            pass
        try:
            # fallback: openai-whisper
            import whisper
            self._stt = whisper.load_model(model_name)
            self._stt_kind = "whisper"
            return self._stt
        except ImportError:
            self._stt = "unavailable"
            return None

    def stt_available(self) -> bool:
        """STT 가능 여부 (라이브러리 있는지)."""
        if self._stt is None:
            self._load_stt()
        return self._stt is not None and self._stt != "unavailable"

    def transcribe(self, audio_path: str) -> str:
        """오디오 파일 → 텍스트 (한국어)."""
        if not self.stt_available():
            return ""
        try:
            if self._stt_kind == "faster_whisper":
                segments, _ = self._stt.transcribe(audio_path, language="ko")
                return " ".join(s.text for s in segments).strip()
            else:
                result = self._stt.transcribe(audio_path, language="ko")
                return result.get("text", "").strip()
        except Exception as e:
            return ""

    # ===== TTS (Piper) =====

    def _load_tts(self, model_path: Optional[str] = None):
        """Piper TTS. 한국어 voice 모델 경로 필요."""
        if self._tts is not None:
            return self._tts
        try:
            from piper import PiperVoice
            # 기본 경로 (사용자가 다운로드 후 지정)
            default = model_path or "ko_KR-default-x_low.onnx"
            self._tts = PiperVoice.load(default)
            return self._tts
        except (ImportError, FileNotFoundError, Exception):
            self._tts = "unavailable"
            return None

    def tts_available(self) -> bool:
        if self._tts is None:
            self._load_tts()
        return self._tts is not None and self._tts != "unavailable"

    def synthesize(self, text: str, output_path: str = "/tmp/eve_speech.wav") -> Optional[str]:
        """텍스트 → wav 파일. 경로 반환."""
        if not self.tts_available():
            return None
        try:
            import wave
            with wave.open(output_path, "wb") as f:
                self._tts.synthesize(text, f)
            return output_path
        except Exception:
            return None

    # ===== OCR (Tesseract) =====

    def _load_ocr(self):
        """pytesseract. tesseract 바이너리 + 한국어 데이터 필요."""
        if self._ocr is not None:
            return self._ocr
        try:
            import pytesseract
            self._ocr = pytesseract
            return self._ocr
        except ImportError:
            self._ocr = "unavailable"
            return None

    def ocr_available(self) -> bool:
        if self._ocr is None:
            self._load_ocr()
        return self._ocr is not None and self._ocr != "unavailable"

    def read_image(self, image_path: str, lang: str = "kor+eng") -> str:
        """이미지 → 텍스트 (한국어 + 영어)."""
        if not self.ocr_available():
            return ""
        try:
            from PIL import Image
            img = Image.open(image_path)
            return self._ocr.image_to_string(img, lang=lang).strip()
        except Exception:
            return ""

    # ===== EVE 학습 통합 =====

    def listen_and_learn(self, audio_path: str) -> str:
        """음성 → 텍스트 → EVE corpus_adapter 학습."""
        text = self.transcribe(audio_path)
        if text and self.engine.corpus_adapter is not None:
            self.engine.corpus_adapter.learn_text(text)
        return text

    def read_and_learn(self, image_path: str) -> str:
        """이미지 → OCR → EVE corpus_adapter 학습."""
        text = self.read_image(image_path)
        if text and self.engine.corpus_adapter is not None:
            self.engine.corpus_adapter.learn_text(text)
        return text

    # ===== 통합 음성 대화 (한 사이클) =====

    def voice_chat(self, audio_path: str) -> tuple[str, list[str], Optional[str]]:
        """
        마이크 녹음 파일 → STT → EVE chat → TTS 합성.
        Returns: (입력 텍스트, EVE 응답 chunks, 합성 wav 경로)
        """
        user_text = self.transcribe(audio_path)
        if not user_text:
            return ("", [], None)
        chunks = list(self.engine.chat_stream(user_text))
        full_response = " ".join(chunks)
        wav_path = self.synthesize(full_response)
        return (user_text, chunks, wav_path)

    # ===== 상태 =====

    def status(self) -> dict:
        """어떤 기능이 가능한지 보고."""
        return {
            "stt": self.stt_available(),
            "tts": self.tts_available(),
            "ocr": self.ocr_available(),
        }
