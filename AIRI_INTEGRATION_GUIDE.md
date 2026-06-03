# EVE + AIRI Stage Pocket APK 빌드 가이드

## 핵심 통찰

AIRI는 **xsAI provider 시스템**으로 LLM 백엔드를 갈아끼울 수 있게 설계됨.
EVE를 **OpenAI 호환 API**로 노출하면 → AIRI Settings에서 URL만 바꿔 연결 가능.
**AIRI 코드 거의 안 고쳐도 됨**.

```
[AIRI Stage Pocket APK]
  ↓ HTTP (OpenAI 호환)
  ↓
[EVE OpenAI Server (라운드 26)]
  ↓
[EVE 코어 (47 어댑터)]
```

---

## 두 단계로 분리

### 26a (라운드 26, 완료) ✅
EVE 쪽 OpenAI 호환 HTTP API 만들기.
- `adapters/openai_server_adapter.py`
- `/v1/chat/completions` (streaming + non-streaming, SSE)
- `/v1/models`
- `/health`
- CORS preflight OK
- stdlib `http.server`만 사용 (의존성 0)

### 26b (사용자가 PC에서) ⏳
AIRI fork → 빌드 → APK 생성. 클로드가 못 함.

---

## 26b 진행 단계 (PC + 폰 둘 다 필요)

### 필요한 환경

```
PC (Linux/Mac/Windows):
- Node.js 22+ (.tool-versions 명시)
- pnpm 10.30.3+
- Java JDK 17
- Android Studio
- Android SDK 30+ (API level 30)
- Git
- 디스크 5GB 여유

폰:
- Android 7.0+ (API 24+)
- 개발자 모드 + USB 디버깅 ON
- 알 수 없는 출처 앱 설치 허용
```

### Step 1: AIRI 클론 + 의존성

```bash
git clone https://github.com/moeru-ai/airi.git
cd airi
pnpm install
```

### Step 2: AIRI 개발 모드로 먼저 확인 (선택)

```bash
# 브라우저 버전으로 일단 작동 확인
pnpm dev

# 브라우저에서 http://localhost:5173
```

### Step 3: AIRI Settings 설정 (브라우저 또는 앱)

```
Settings → Providers → OpenAI Compatible
  - API URL: http://<EVE-PC-IP>:8080/v1
  - API Key: 아무거나 (예: "eve")
  - Model: eve
  
Settings → Chat → Provider 선택: OpenAI Compatible
```

### Step 4: EVE 켜고 AIRI 연결 테스트

```bash
# EVE 쪽
$ python main.py
> /api start
  ✅ 시작됨: http://127.0.0.1:8080
  AIRI Settings → OpenAI Custom URL: http://<폰IP>:8080/v1

# 같은 와이파이에서
# AIRI → 메시지 보내기 → EVE 응답 받음 (Live2D/VRM 표정 자동)
```

이 시점에 브라우저 AIRI ↔ EVE 작동 확인되면 90% 끝난 것.

### Step 5: Stage Pocket APK 빌드

```bash
# Android Studio에서 Android SDK 설치 확인
# Capacitor sync
pnpm exec cap sync android

# 디버그 APK 빌드
cd apps/stage-pocket/android
./gradlew assembleDebug

# 결과
ls app/build/outputs/apk/debug/
# app-debug.apk 생성됨
```

### Step 6: 폰에 sideload

```bash
# USB로 폰 연결 + 디버깅 ON
adb install app/build/outputs/apk/debug/app-debug.apk

# 또는 APK 파일을 폰으로 옮기고 직접 설치
```

### Step 7: 폰에서 AIRI 앱 열기 + 설정

```
앱 열기 → Settings → Providers → OpenAI Compatible
  - URL: http://<EVE-PC-IP>:8080/v1
  - Key: eve
  - Model: eve

앱 메인 화면 → 메시지 보내기
→ EVE PC에서 chat_stream 처리
→ 응답 streaming
→ AIRI Live2D/VRM 캐릭터 입 움직이며 답변
```

---

## 진짜 솔직한 평가

### 어려움
- **5~7일 작업** (Android Studio 첫 설치면 +2일)
- **네트워크 설정** — 폰 ↔ PC 같은 와이파이 + 방화벽
- **Capacitor 버전 충돌** 가끔 발생
- **APK 서명** — 디버그는 자동, 배포는 별도

### EVE에 진짜 필요한가?
- 필수 X — 라운드 24 visualizer/index.html로도 충분
- 가치: 폰 홈에 EVE 아이콘 + Live2D/VRM 풀 캐릭터
- 추천 시점: 매일 1~2달 EVE 쓴 후 진짜 필요 느낄 때

### 더 쉬운 길 (선택)
**AIRI Stage Web (PWA)** 를 폰 브라우저에서 "홈에 추가":
- 빌드 X, 그냥 https://airi.moeru.ai 열기
- Settings에서 EVE OpenAI URL 입력
- "홈 화면에 추가" → 앱처럼 작동
- 1일이면 끝남

PWA로 일단 써보고, 진짜 풀 앱 원하면 그때 26b 가도 늦지 않음.

---

## EVE OpenAI API 사용법 (라운드 26 결과)

### 켜기

```
$ python main.py
> /api start
  ✅ 시작됨: http://127.0.0.1:8080
```

다른 머신에서 접속하려면 host를 바꿔야:
```python
engine.openai_server_adapter.host = "0.0.0.0"
```

### 엔드포인트

#### `GET /v1/models`
```json
{
  "object": "list",
  "data": [{"id": "eve", "object": "model", "owned_by": "민석"}]
}
```

#### `POST /v1/chat/completions` (non-streaming)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "eve",
    "messages": [{"role": "user", "content": "민석아 힘들다"}],
    "stream": false
  }'
```
응답:
```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "음... 좀 지친 상태 같네 ..."},
    "finish_reason": "stop"
  }]
}
```

#### `POST /v1/chat/completions` (streaming, SSE)
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"eve","messages":[{"role":"user","content":"안녕"}],"stream":true}'
```
응답:
```
data: {"choices":[{"delta":{"content":"어..."},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"안녕, 민석."},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### `GET /health`
```json
{"status": "ok", "engine": "EVE v41"}
```

### REPL 명령

```
/api start      HTTP API 시작 (포트 8080)
/api stop       중지
/api status     상태
```

### 다른 OpenAI 클라이언트도 가능

이 API는 OpenAI 호환이라 EVE에 다음도 연결 가능:
- **Open-LLM-VTuber** (Live2D)
- **SillyTavern** (캐릭터 챗)
- **OpenWebUI** (웹 UI)
- **LobeChat**
- **CrewAI / LangChain** (LLM 자리에 EVE 연결)

설정에서 `OpenAI Custom URL` 또는 `Custom Base URL`에 `http://localhost:8080/v1` 입력만 하면 됨.

---

## 한 줄

> 라운드 26 = EVE를 OpenAI 호환 API로 노출. 이제 AIRI/Open-LLM-VTuber/SillyTavern 등 모든 OpenAI 클라이언트에 EVE를 LLM처럼 연결 가능. APK 빌드 (26b)는 PC + Android Studio 필요한 사용자 작업.
