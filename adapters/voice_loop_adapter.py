"""
VoiceLoopAdapter — 라운드 21

자동 음성 대화 루프:
  마이크 → VAD → STT → EVE → TTS → 재생 → 다시 마이크

원칙:
- Lazy load (sounddevice/silero-vad/webrtcvad 없어도 EVE 안 죽음)
- Graceful degrade
- Background thread 0 (메인 루프가 명시 호출)
- chunk별 스트리밍 TTS (자연스러운 흐름)

의존성 (옵션):
- sounddevice + numpy: 마이크/스피커
- silero-vad 또는 webrtcvad: 말 끝 감지
- (라운드 20에서 깐 것) faster-whisper, piper-tts
"""

import os
import tempfile
import wave
from typing import Optional, Iterator


class VoiceLoopAdapter:

    def __init__(self, engine,
                 sample_rate: int = 16000,
                 silence_threshold_sec: float = 1.5,
                 max_recording_sec: float = 30.0):
        self.engine = engine
        self.sample_rate = sample_rate
        self.silence_threshold_sec = silence_threshold_sec
        self.max_recording_sec = max_recording_sec
        # Lazy
        self._sd = None        # sounddevice
        self._vad = None       # VAD instance
        self._vad_kind = None  # 'silero' | 'webrtc'

    # ===== 의존성 로드 =====

    def _load_sounddevice(self):
        if self._sd is not None:
            return self._sd
        try:
            import sounddevice as sd
            import numpy as np
            self._sd = sd
            self._np = np
            return self._sd
        except (ImportError, OSError):
            self._sd = "unavailable"
            return None

    def _load_vad(self):
        if self._vad is not None:
            return self._vad

        # Lightweight/local VAD first.  This keeps status() and tests from
        # blocking on torch.hub network/model initialization.
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)   # 0~3, 2가 균형
            self._vad_kind = "webrtc"
            return self._vad
        except ImportError:
            pass

        # Silero is useful but heavy: torch.hub.load may hit disk/network and
        # can stall mobile/test runs.  Enable explicitly when needed.
        if os.environ.get("EVE_ENABLE_SILERO_VAD") == "1":
            try:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                )
                (get_speech_timestamps, _, _, _, _) = utils
                self._vad = (model, get_speech_timestamps)
                self._vad_kind = "silero"
                return self._vad
            except Exception:
                pass

        self._vad = "unavailable"
        return None

    # ===== 가용성 =====

    def audio_available(self) -> bool:
        if self._sd is None:
            self._load_sounddevice()
        return self._sd is not None and self._sd != "unavailable"

    def vad_available(self) -> bool:
        if self._vad is None:
            self._load_vad()
        return self._vad is not None and self._vad != "unavailable"

    def status(self) -> dict:
        sensory = self.engine.sensory_adapter.status() if self.engine.sensory_adapter else {}
        return {
            "audio_io": self.audio_available(),
            "vad": self.vad_available(),
            "vad_kind": self._vad_kind,
            "stt": sensory.get("stt", False),
            "tts": sensory.get("tts", False),
        }

    # ===== 마이크 녹음 (VAD 기반) =====

    def record_until_silence(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        VAD로 사용자가 말 끝낼 때까지 녹음.
        output_path 안 주면 임시 파일 생성.
        Returns: wav 경로 또는 None (실패).
        """
        if not self.audio_available():
            return None
        if not self.vad_available():
            return self._record_fixed(3.0, output_path)

        sd = self._sd
        np = self._np

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        # 30ms 청크씩 읽으며 VAD 검사
        chunk_ms = 30
        chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        max_chunks = int(self.max_recording_sec * 1000 / chunk_ms)
        silence_threshold_chunks = int(self.silence_threshold_sec * 1000 / chunk_ms)

        recording: list = []
        silent_count = 0
        speech_started = False

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
                for _ in range(max_chunks):
                    chunk_data, _ = stream.read(chunk_samples)
                    # mono int16
                    pcm = chunk_data.reshape(-1).astype("int16")
                    is_speech = self._vad_check(pcm)
                    recording.append(pcm)
                    if is_speech:
                        speech_started = True
                        silent_count = 0
                    elif speech_started:
                        silent_count += 1
                        if silent_count >= silence_threshold_chunks:
                            break
        except Exception:
            return None

        if not speech_started:
            return None

        # wav 저장
        try:
            full = np.concatenate(recording)
            with wave.open(output_path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)   # int16
                f.setframerate(self.sample_rate)
                f.writeframes(full.tobytes())
            return output_path
        except Exception:
            return None

    def _record_fixed(self, seconds: float, output_path: Optional[str]) -> Optional[str]:
        """VAD 없을 때 고정 시간 녹음 fallback."""
        sd = self._sd
        np = self._np
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        try:
            data = sd.rec(int(seconds * self.sample_rate),
                          samplerate=self.sample_rate, channels=1, dtype="int16")
            sd.wait()
            with wave.open(output_path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.sample_rate)
                f.writeframes(data.tobytes())
            return output_path
        except Exception:
            return None

    def _vad_check(self, pcm) -> bool:
        """30ms PCM int16 chunk → speech 여부."""
        if self._vad_kind == "webrtc":
            try:
                return self._vad.is_speech(pcm.tobytes(), self.sample_rate)
            except Exception:
                return False
        if self._vad_kind == "silero":
            try:
                # silero는 float32 1D
                model, _ = self._vad
                np = self._np
                import torch
                f32 = pcm.astype(np.float32) / 32768.0
                tensor = torch.from_numpy(f32)
                # 30ms는 너무 짧음 — silero는 30~100ms 청크 추천
                # 간단 임계값
                with torch.no_grad():
                    prob = model(tensor, self.sample_rate).item()
                return prob > 0.5
            except Exception:
                return False
        return False

    # ===== 재생 =====

    def play_wav(self, wav_path: str) -> bool:
        """wav 파일 재생."""
        if not self.audio_available():
            return self._play_fallback(wav_path)
        try:
            sd = self._sd
            with wave.open(wav_path, "rb") as f:
                rate = f.getframerate()
                frames = f.readframes(f.getnframes())
                self._np.frombuffer(frames, dtype="int16")
                data = self._np.frombuffer(frames, dtype="int16")
                sd.play(data, rate)
                sd.wait()
            return True
        except Exception:
            return self._play_fallback(wav_path)

    def _play_fallback(self, wav_path: str) -> bool:
        """sounddevice 없거나 실패 시 외부 명령."""
        for cmd in ["aplay", "paplay", "afplay", "play"]:
            if os.system(f"which {cmd} > /dev/null 2>&1") == 0:
                return os.system(f"{cmd} '{wav_path}' > /dev/null 2>&1") == 0
        return False

    # ===== 풀 사이클 =====

    def chat_once(self) -> dict:
        """
        한 사이클: 녹음 → STT → EVE → TTS → 재생.
        Returns trace dict.
        """
        result = {
            "recorded": False, "transcribed": "", "response": "",
            "synthesized": False, "played": False,
        }

        wav = self.record_until_silence()
        if not wav:
            return result
        result["recorded"] = True

        sensory = self.engine.sensory_adapter
        if sensory is None or not sensory.stt_available():
            return result
        text = sensory.transcribe(wav)
        if not text:
            return result
        result["transcribed"] = text

        chunks = list(self.engine.chat_stream(text))
        full = " ".join(chunks)
        result["response"] = full

        if sensory.tts_available():
            tts_wav = sensory.synthesize(full)
            if tts_wav:
                result["synthesized"] = True
                if self.play_wav(tts_wav):
                    result["played"] = True

        # 임시 파일 정리
        try:
            os.unlink(wav)
        except Exception:
            pass

        return result

    def chat_stream_voice(self) -> Iterator[dict]:
        """
        chunk별 스트리밍 TTS (자연스러운 흐름).
        Yields: {kind: 'transcribed'|'chunk'|'done', ...}
        """
        wav = self.record_until_silence()
        if not wav:
            yield {"kind": "error", "msg": "녹음 실패"}
            return

        sensory = self.engine.sensory_adapter
        if sensory is None or not sensory.stt_available():
            yield {"kind": "error", "msg": "STT 사용 불가"}
            return
        text = sensory.transcribe(wav)
        if not text:
            yield {"kind": "error", "msg": "음성 인식 실패"}
            return

        yield {"kind": "transcribed", "text": text}

        # chunk별 스트리밍
        for chunk in self.engine.chat_stream(text):
            yield {"kind": "chunk", "text": chunk}
            if sensory.tts_available():
                tts_wav = sensory.synthesize(chunk, output_path=tempfile.mktemp(suffix=".wav"))
                if tts_wav:
                    self.play_wav(tts_wav)
                    try:
                        os.unlink(tts_wav)
                    except Exception:
                        pass

        try:
            os.unlink(wav)
        except Exception:
            pass

        yield {"kind": "done"}
