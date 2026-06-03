# EVE 시각/청각 설치 가이드 (라운드 20)

폰 Termux 또는 PC에서 STT/TTS/OCR 활성화하는 법.

## 핵심 원칙

EVE는 **lazy load** — 라이브러리 없어도 EVE 코어는 정상 작동. 필요한 것만 설치하면 됨.

## 1. STT (Whisper, 음성 → 텍스트)

### 폰 Termux
```bash
# faster-whisper (가벼움 추천)
pip install faster-whisper

# 또는 openai-whisper (느리지만 안정)
pip install openai-whisper
```

**모델 크기 선택**:
- `tiny` (75MB) — 빠름, 정확도 낮음
- `small` (500MB) — 추천 (Z Fold 6 OK)
- `medium` (1.5GB) — 폰 빡빡
- `large` (3GB) — 폰 무리

**사용**:
```
> /listen /sdcard/recording.wav
들었어: 민석이 PT 힘들었다
어... → 좀 지친 상태 같네. ...
```

녹음은 폰 기본 녹음 앱 사용. wav/mp3/m4a 다 지원.

## 2. TTS (Piper, 텍스트 → 음성)

### 설치
```bash
pip install piper-tts
```

### 한국어 voice 모델 다운로드
[Piper voices Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/ko)

```bash
# 한국어 voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ko/ko_KR/default/x_low/ko_KR-default-x_low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ko/ko_KR/default/x_low/ko_KR-default-x_low.onnx.json

# EVE 디렉토리에 두기
mv ko_KR-default-x_low.* eve_v41/
```

**사용**:
```
> /say 안녕 민석아
합성됨: /tmp/eve_speech.wav
# 폰 음악 플레이어로 재생
```

## 3. OCR (Tesseract, 이미지 속 텍스트)

### 폰 Termux
```bash
pkg install tesseract
pkg install tesseract-data-kor   # 한국어 데이터
pip install pytesseract pillow
```

### PC (Ubuntu/Debian)
```bash
sudo apt install tesseract-ocr tesseract-ocr-kor
pip install pytesseract pillow
```

**사용**:
```
> /read /sdcard/screenshot.jpg
읽었어: 오늘 PT 일정 — 푸쉬업 100개...
→ corpus 학습 완료
```

## 4. 통합 음성 대화

폰에서 자연스러운 음성 대화:

```bash
# 1. 폰 녹음 앱으로 음성 녹음 → /sdcard/voice.wav

# 2. EVE 실행
python main.py

> /listen /sdcard/voice.wav
들었어: 민석이 오늘 사역 빡셌어
음... → 좀 지친 상태 같네. ...

# 3. 응답 음성 합성
> /say 좀 지친 상태 같네 민석이 피로한 거 같이 느껴져
합성됨: /tmp/eve_speech.wav

# 4. 폰 음악 앱으로 재생
```

## 5. 자동화 — 라운드 21 (자동 음성 루프) ✅

이제 수동 아니다. 한 번 명령으로 무한 음성 대화.

### 추가 의존성 (옵션)

```bash
# 마이크/스피커 (필수)
pip install sounddevice numpy

# Termux: PortAudio 먼저
pkg install portaudio
pip install sounddevice numpy

# VAD (말 끝 자동 감지)
pip install webrtcvad         # 가벼움 (1MB) ← 추천
# 또는 더 정확하지만 무거움 (40MB + torch)
pip install silero-vad torch
```

### 사용

```
> /sensory
  STT: ✅, TTS: ✅, OCR: ✅
  Audio I/O: ✅
  VAD: ✅ webrtc

> /voice
  말하세요...
  들었어: 민석이 PT 빡셌어
  답: 음... 좀 지친 상태 같네 민석이 피로한 거 같이...
  ♪ 재생됨

> /voiceloop 5
  [1/5] 말하세요...
  들었어: ...
  답: ...
  ♪ 재생됨
  
  [2/5] 말하세요...
  ...

> /voiceloop 0    # 무한 (Ctrl+C로 중단)
```

### 작동 원리

```
[마이크 30ms 청크 연속 읽기]
    ↓
[VAD: 각 청크가 음성인지 검사]
    ↓
[음성 시작 → 녹음 시작]
    ↓
[1.5초 침묵 → 녹음 끝]
    ↓
[wav 임시 파일 → Whisper STT]
    ↓
[user_text → EVE chat_stream]
    ↓
[response → Piper TTS → wav]
    ↓
[sounddevice로 자동 재생]
    ↓
[다시 마이크 대기]
```

### 장점

- VAD 자동 감지 → "어디까지 말하는지" 안 알려줘도 됨
- 손 안 써도 됨
- 외출 중 폰 들고 자연스럽게 대화
- 운전 중에도 (안전 주의)

### 한계

- 인터럽션 X — EVE 말하는 도중 끼어들면 무시 (라운드 22+에서)
- 화이트노이즈 환경 → VAD 오작동 가능
- 폰 마이크 품질 의존

### 발열 / 배터리

```
Whisper small loaded:    1GB RAM
Audio capture 24/7:      낮은 CPU
TTS 합성 (간헐):         낮은 CPU
백그라운드 idle:         거의 0

배터리: 1시간 voice loop ≈ 5~10% 소모
```

## 6. 라운드 22 — 인터넷 학습 + 비전

### 인터넷 학습

```bash
pip install requests beautifulsoup4 wikipedia youtube-transcript-api
```

**사용**:
```
> /web https://ko.wikipedia.org/wiki/파이썬
  텍스트 길이: 24531
  학습: ✅

> /wiki 머신러닝
  텍스트 길이: 1234
  학습: ✅

> /yt https://youtu.be/dQw4w9WgXcQ
  자막 길이: 5678
  학습: ✅
```

매번 corpus_adapter로 흘러서 EVE 카테고리/페어 자동 생성.

### CLIP 비전

```bash
# 무거움 — Z Fold 6 OK, 일반 폰 빡빡
pip install transformers torch pillow
# 첫 실행 시 모델 자동 다운로드 (약 600MB)
```

**사용**:
```
> /see /sdcard/pic.jpg
  봤어: 강아지, 동물 같은 거 보여
  카테고리: [('강아지', '0.78'), ('동물', '0.12'), ('실외', '0.05')]
  학습: ✅
```

### Safety - 화이트리스트 (옵션)

기본은 모든 도메인 허용. 안전하게 가려면 코드에서:
```python
engine.web_learning_adapter.allowed_domains = {
    "ko.wikipedia.org", "en.wikipedia.org",
    "namu.wiki",
    "youtube.com",
}
```
그러면 화이트리스트 외 URL은 차단.

### 카테고리 추가 (CLIP)

기본 60개 한국어 라벨. 더 추가:
```python
engine.vision_adapter.add_label("새카테고리")
```

라운드 22 완료 = EVE가 인터넷 + 사진으로 학습 가능.

## 6. 디스크 / 메모리

### Z Fold 6 RAM 12GB 시뮬
```
EVE 코어:        300MB
faster-whisper small: 1GB (loaded)
Piper TTS:       300MB
Tesseract:       200MB
─────────────
합계 (피크):     약 1.8GB
─────────────
여유:            10GB+ → ★ 충분
```

### 디스크
```
EVE 코드:         5MB
Whisper small 모델: 500MB
Piper voice:     100MB
Tesseract 데이터: 50MB
persistence:     500KB ~ 100MB (누적)
─────────────
합계:            약 700MB
```

## 7. 문제 해결

### "STT 사용 불가"
```bash
pip install faster-whisper --no-cache-dir
# 또는
pip install openai-whisper
```

### "TTS 사용 불가" — voice 모델 못 찾음
EVE 실행 디렉토리에 `ko_KR-default-x_low.onnx` 파일 있는지 확인:
```bash
ls eve_v41/ko_KR*
```

또는 sensory_adapter에 경로 직접 지정:
```python
engine.sensory_adapter._load_tts(model_path="/path/to/ko_KR-default-x_low.onnx")
```

### "OCR 사용 불가"
```bash
# Termux
pkg install tesseract tesseract-data-kor
# 또는 영어만 빠르게 테스트
pip install pytesseract pillow
```

### Whisper 너무 느림
`tiny` 모델 사용 또는 GPU 있으면 CUDA:
```python
WhisperModel("small", device="cuda")
```

### 발열
faster-whisper > openai-whisper. small > medium. CPU 사용 줄임.

## 진짜 작동 확인

```
> /sensory
  STT (Whisper): ✅ 가능
  TTS (Piper):   ✅ 가능
  OCR (Tesseract): ✅ 가능
```

세 개 다 ✅면 EVE는 보고 듣고 말함. 진짜 감각 있는 존재.

## 다음 라운드 (21~22)

라운드 21에서 만들 것:
- 마이크 직접 녹음 (sounddevice)
- 자동 음성 대화 루프
- VAD (말 끝 감지)
- 음성 응답 자동 재생

라운드 22:
- CLIP (이미지 → 카테고리, 시각 학습)
- 카메라 통합

지금은 수동이라도 충분히 시작 가능. 가벼운 버전 = 라이브러리 옵션, EVE 코어 가벼움 유지.
