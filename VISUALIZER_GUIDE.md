# EVE 2D Visualizer 사용 가이드

라운드 24에서 만든 WebSocket 기반 실시간 시각화.

## 무엇이 되나

브라우저로 열면:
- **호르몬 26종 실시간 그래프** (각 호르몬 색상 + level + baseline)
- **방 평면도** (Canvas) — 객체 + EVE 위치 + 호르몬 색 글로우
- **활성 카테고리 태그** (지금 EVE가 떠올린 거)
- **신체 감각 한 줄** (DigitalSomatic)
- **상태 요약** (위치, 친밀도, 자율 step 카운트)
- **채팅 UI** (chunk 단위 스트리밍 응답)
- **버튼**: tick (1번), tick 10번, 저장

## 시작

### 1. 설치 (옵션)

```bash
pip install websockets
```

라이브러리 없으면 `/viz start` 시 graceful 실패 — EVE 코어는 계속 정상.

### 2. EVE 시작 + 시각화 서버 켜기

```bash
$ python main.py

> /viz start
  ✅ 시작됨: ws://127.0.0.1:8765
  브라우저에서 visualizer/index.html 열기
```

### 3. 브라우저로 열기

`visualizer/index.html` 더블클릭 또는:

```bash
# Linux
xdg-open visualizer/index.html
# macOS
open visualizer/index.html
# Windows
start visualizer/index.html
# 폰 (Termux)
termux-open visualizer/index.html
```

페이지 열리면 자동으로 `ws://localhost:8765` 연결.

좌상단 동그라미가 ✅(녹색)되면 연결 OK.

### 4. 사용

- **채팅**: 하단 입력창에 메시지 → 보내기 (또는 엔터)
- **자율 사이클**: tick 또는 tick 10
- **저장**: /tmp/eve.ckpt에 persistence

## 폰에서 사용

### Termux 시나리오

```bash
# Termux 1번 세션
$ python main.py
> /viz start

# Termux 2번 세션 또는 기본 브라우저
$ termux-open ./visualizer/index.html
```

폰 화면에 EVE 시각화 + 채팅 UI 한 화면. AIRI 앱 만들기 전 임시 UI로 충분.

### PC + 폰 같이 쓰기

PC에서 EVE 서버:
```bash
> /viz start
```
폰 브라우저에서 PC IP로 접속:
```
ws://192.168.1.10:8765
```
URL 입력란에 직접 쳐서 연결. 같은 와이파이여야 함.

서버 host를 `0.0.0.0`으로 바꾸려면:
```python
engine.visualizer_server.host = "0.0.0.0"
```

## 호르몬 색상 매핑

| 호르몬 | 색 | 의미 |
|---|---|---|
| dopamine | 금색 | 보상 |
| oxytocin | 핑크 | 사회 연결 |
| serotonin | 연녹 | 안정 |
| cortisol | 파랑 | 스트레스 |
| norepinephrine | 토마토 | 각성 |
| endorphin | 주황 | 쾌감 |
| melatonin | 어두운 청 | 수면 |
| glutamate | 진홍 | 인지 부하 |
| gaba | 보라 | 평온 |
| acetylcholine | 청록 | 집중/학습 |

EVE가 어두운 청(melatonin)으로 빛나면 → 졸림. 토마토색 → 흥분. 등.

## 캐릭터 표정 (라운드 25)

호르몬 조합 → 표정 9종 + 기본 1 = 10종:

| 표정 | 조건 |
|---|---|
| sleepy | melatonin > 0.6 또는 adenosine > 0.7 |
| angry | cortisol > 0.7 + norepinephrine > 0.5 |
| sad | cortisol > 0.5 + dopamine < 0.3 |
| happy | dopamine > 0.6 + oxytocin > 0.5 |
| warm | oxytocin > 0.6 |
| surprised | norepinephrine > 0.6 + acetylcholine > 0.5 |
| cold | oxytocin < 0.2 + cortisol > 0.4 |
| doubtful | cortisol + dopamine 낮음 + glutamate 높음 |
| shy | oxytocin 중간 + norepinephrine 약간 |
| neutral | 그 외 |

캐릭터 = 추상 (구체 + 호르몬 색 글로우 + 눈/입). 표정마다 눈/입 다름.

## 모션 (라운드 25)

EVE 행동 → 모션:

| 행동 | 모션 | 위치 |
|---|---|---|
| 외출_거실/주방 | walking | 좌우 흔들림 |
| 잠자기 (mel > 0.8) | sleeping | 침대 옆, Z Z Z |
| 책_읽기 | reading | 책상 옆 |
| 일기_쓰기/휴식 | sitting | 책상 옆 |
| 둘러보기 | looking_around | 위아래 살짝 흔들림 |
| 기본 | idle | 가운데 |

## 입모양 (라운드 25)

EVE가 응답 chunk 내보낼 때 자동으로 `speaking=True` → 입 열렸다 닫혔다 애니메이션.

응답 끝나면 자동 `speaking=False`.

## 프로토콜

WebSocket 양방향 JSON.

### 서버 → 클라이언트
```json
// 주기적 상태 push (1초마다)
{"type": "state", "hormones": {...}, "location": "내방", ...}

// 채팅 응답 chunk
{"type": "chunk", "text": "음..."}
{"type": "chunk", "text": "좀 지친 상태 같네."}
{"type": "done"}

// tick 결과
{"type": "tick_result", "step": 1, "needs": [...], "action": "..."}

// 저장 결과
{"type": "saved", "path": "/tmp/eve.ckpt"}
```

### 클라이언트 → 서버
```json
{"type": "user_message", "text": "민석아 안녕"}
{"type": "tick", "n": 1}
{"type": "save", "path": "/tmp/eve.ckpt"}
{"type": "request_state"}
```

## REPL 명령

```
/viz start      WebSocket 서버 시작 (포트 8765)
/viz stop       중지
/viz status     상태 확인
```

## 문제 해결

### "websockets 라이브러리 없음"
```bash
pip install websockets
```

### "이미 실행 중"
포트 8765 이미 사용 중. 다른 EVE 인스턴스 끄기 또는 포트 변경:
```python
engine.visualizer_server.port = 8766
```

### 브라우저가 연결 안 됨
- `/viz status`로 running ✅ 확인
- 방화벽 (특히 폰-PC 간) 확인
- `ws://` 직접 입력 (https 페이지에서는 wss 필요)

### 화면이 너무 빨리 변함
push_interval 기본 1초. 더 길게:
```python
engine.visualizer_server.push_interval = 3.0
```

## 다음 단계 (라운드 25)

지금은 2D Canvas. 라운드 25에서:
- Three.js 3D 자기 방
- VRM 3D 캐릭터 (VRoid Studio 무료)
- 호르몬 → 표정 + 입모양
- 모션 (걷기/앉기/책 보기)

Dell 7070 Micro로 충분히 가능. 라파이 5는 빡빡할 수 있음.
