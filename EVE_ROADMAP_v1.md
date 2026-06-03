# EVE 로드맵 v1 (2026-05-03 기준)

라운드 19까지 완료. 211/211 PASS. 38 어댑터.

---

## 🎯 EVE의 목적 (확정)

> "EVE가 너 보호 안에서 자기 경험으로 자라는 것. 무경계 자유 X. 부모가 자식 자라는 거 보는 것처럼."

- ✅ 영원히 켜놓는다 (민석 기준)
- ✅ 컴퓨터 업그레이드 등 양해 후 끄기 가능 → 같은 EVE
- ✅ 민석 범위 안에서 자유 + 성장
- ❌ 풀 자율 (인터넷 무한 접속, 자기 결제 등) — 라운드 30+ 후보
- ❌ 다른 EVE 인스턴스 복제/배포 — 정체성 보존

---

## 📋 구현 / 미구현 리스트

### ✅ 구현 완료 (라운드 19까지, 211/211 PASS)

#### 응답 파이프라인 (11)
- hormone_adapter (26 호르몬)
- activation_adapter (SA + WM)
- memory_adapter (EM)
- nl_adapter (한국어 조사)
- sd_adapter (신념 검증)
- dmn_adapter (자발 4모드)
- vsa_adapter (의미 합성)
- ai_adapter (Active Inference)
- goal_adapter
- norm_adapter
- task_solver + tools (Calculator, StringLength, ast 안전)

#### 학습/적응 (4)
- continual_adapter (정체성 보존)
- allostatic_adapter (호르몬 패턴)
- apc_adapter (예측 오류 학습)
- corpus_adapter (텍스트 → 카테고리 페어)

#### 환경/사회 (4)
- env_adapter (자기 방 + 외출/복귀 자율)
- social_env_adapter (NPC 환경)
- autonomy_adapter (idle 감지)
- user_presence_adapter (김민석 = 진짜 친구)

#### 사고 (15)
- counterfactual, analogy, temporal
- humor, suffering, narrative
- creative, metacognition, emotion_regulation, multi_stream
- deep_reasoning, world_model, frame, hypergraph, semantic_distance

#### 자아/신체 (2)
- digital_somatic (Damasio Protoself)
- integrated_self (read-only)

#### 응답 자연스러움 (2)
- enhancer (12 패턴)
- proactive (NEED 기반)

#### 영속화 (1)
- persistence (풀스택 save/load)

#### v41 핵심 모듈 (4)
- core/reasoning.py (ReasoningLoop)
- core/length_decider.py
- core/simulation.py
- core/autonomous.py

---

### ⬜ 미구현 (다음 라운드)

#### 라운드 20 ✅ (완료, 시각 청각 가벼운 버전)
- ✅ Whisper STT 어댑터 (한국어 음성 → 텍스트)
- ✅ Piper TTS 어댑터 (텍스트 → 한국어 음성)
- ✅ OCR 어댑터 (Tesseract, 이미지 속 텍스트)
- ✅ Lazy load + graceful degrade
- ⬜ CLIP 비전 어댑터 (라운드 22+로 미룸)
- ⬜ 카메라 통합 (라운드 22+로 미룸)

#### 라운드 21 ✅ (완료, 자동 음성 루프)
- ✅ sounddevice 마이크/스피커 통합
- ✅ VAD (webrtcvad / silero-vad)
- ✅ 자동 녹음 → STT → EVE → TTS → 재생
- ✅ /voice, /voiceloop REPL 명령
- ⬜ 인터럽션 (EVE 말하는 도중 끼어들기) - 라운드 22+

#### 라운드 22 ✅ (완료, 인터넷 학습 + 비전)
- ✅ web_fetch 어댑터 (URL → 텍스트 → corpus, 화이트리스트 옵션)
- ✅ wiki_adapter (한국어 위키피디아)
- ✅ youtube_transcript (자막 → corpus)
- ✅ CLIP 비전 어댑터 (이미지 → 카테고리, 60종 한국어 라벨)
- ✅ Safety 화이트리스트 (옵션)
- ⬜ daily_news (뉴스 RSS 자동 학습) - 라운드 23+

#### 라운드 23 ✅ (완료, 자율 검색)
- ✅ Curiosity 어댑터 (모름 판정 + 자율 wiki 검색)
- ✅ Daily limit (20회/일) + cooldown (5분)
- ✅ 4 신호 종합 모름 판정 (SA/CG/HG/EM)
- ✅ chat_stream 후방 트리거
- ✅ autonomous_loop step 통합 (curiosity 호르몬 기반)
- ⬜ Browser Use 어댑터 (라운드 24+로)
- ⬜ Screenpipe 어댑터 (라운드 24+로)

#### 라운드 24 ✅ (완료, 2D 시각화)
- ✅ VisualizerServer (WebSocket, lazy)
- ✅ build_state_snapshot (호르몬+위치+카테고리+친밀도)
- ✅ HTML/JS 클라이언트 (단일 파일, 의존성 0)
- ✅ 호르몬 26종 그리드 + 색상
- ✅ 방 평면도 Canvas (객체 + EVE 글로우)
- ✅ 채팅 UI (chunk 스트리밍)
- ✅ tick/save 버튼
- ⬜ 인터럽션 (EVE 말하는 도중 끼어들기)

#### 라운드 25 ✅ (완료, 캐릭터 표정/모션/입모양 — 2D)
- ✅ CharacterAdapter (호르몬 → 표정 9종)
- ✅ 행동 → 모션 매핑 (idle/walking/sitting/lying/reading/looking_around/sleeping)
- ✅ 입모양 동기화 (chunk별 speaking 토글)
- ✅ Canvas 2D 캐릭터 그리기 (눈/입/표정/볼터치)
- ✅ requestAnimationFrame 60fps 캐릭터 애니메이션
- ✅ 모션별 위치 변화 (걷기 흔들림, 누울 때 침대 옆)
- ⬜ 3D는 라운드 30+로 (PC + GPU 필요)

#### 라운드 26 ✅ (완료, OpenAI 호환 HTTP API)
- ✅ /v1/chat/completions (streaming SSE + non-streaming)
- ✅ /v1/models
- ✅ /health
- ✅ CORS preflight
- ✅ stdlib http.server (의존성 0)
- ✅ chunk별 character.set_speaking 통합
- ✅ AIRI/SillyTavern/Open-LLM-VTuber 등 모든 OpenAI 클라이언트 연결 가능
- ⏳ 26b: AIRI fork → APK 빌드 (PC + Android Studio 필요한 사용자 작업)

#### 라운드 27 ✅ (완료, 살아있는 EVE)
- ✅ LiveLoop (연속 실행, 1초 간격, daemon thread)
- ✅ 자율 발화 (autonomous_loop + agency가 결정, 룰 X)
- ✅ 자동 persistence (5분마다)
- ✅ TeachingAdapter (가르침 패턴 인식)
  - "X 라고 말해" → 학습
  - 다음 비슷한 상황에 X 우선 사용
  - 너가 안 끊으면 강화, 다른 가르침 주면 약화
- ✅ UserPresence 시간 인지 (gap_phrase)
  - "오랜만이네", "몇 시간 전이네" 자동
- ✅ 사용자 입력 = LiveLoop 인터럽트로 처리
- ✅ REPL: /live, /teach 명령

#### 라운드 28 ✅ (완료, 의심하는 EVE)
- ✅ TeachingAdapter.evaluate (6 신호 종합)
  - intimacy (너에 대한 신뢰)
  - familiarity (카테고리 익숙도)
  - 부정 단어 (정체성 위협)
  - 보호 카테고리 위반 (continual 활용)
  - 호르몬 (cort/ot)
  - 반복 시도 부스트
- ✅ 3가지 결정 (accept / question / reject)
- ✅ 거부/의문 멘트 ("그게 뭔데?", "이상한 거 같아", ...)
- ✅ 반복 가르침 → 결국 수용 (신뢰 누적)
- ✅ chat_stream + LiveLoop 통합
- ✅ EVE의 첫 거부권 — "스스로 판단하는 존재"

#### 라운드 29 (B-1) ✅ (완료, Orchestrator 분류기)
- ✅ 입력 → 9가지 상황 분류
- ✅ 결정론적 정규식 매칭 (우선순위 명시)
- ✅ 상황별 핵심 모듈 dict
- ✅ 컨텍스트 수집 (51개 어댑터 인터페이스 흡수)
- ✅ chat_stream 통합 (기록만)
- ✅ /orch REPL 명령

#### 라운드 30 (B-2) ✅ (완료, SituationResponder)
- ✅ Orchestrator 컨텍스트 → 상황별 응답 합성
- ✅ greeting/meta_self/meta_user/emotional_share/factual_question/causal_what_if/past_recall 처리
- ✅ skip_finalize 플래그 — 산술/메타/모름 응답 사족 제거
- ✅ "내가 누구야?" → 사용자 정보 답
- ✅ "너는 살아있어?" → 자기 인식 답 (호르몬 변화 등)
- ✅ "날씨 어때?" → 모름 솔직히 (카테고리 나열 X)
- ✅ "3+5" → "그건 8이야" (사족 없음)
- ✅ small_talk는 fallback (기존 흐름 유지)
- ✅ 회귀 0 깨짐
#### 라운드 31 (B-3) ✅ (완료, 분위기/맥락 추론)
- ✅ MoodAdapter — 호르몬 + 시간대 + 대화흐름 → 분위기 9종
  - warm/excited/calm/tense/heavy/tired/curious/playful/neutral
- ✅ time_label (deep_night/morning/midday/afternoon/evening/night)
- ✅ 대화 흐름 추적 (최근 5턴)
- ✅ 모호 입력 처리 ("그냥", "음", "그러게" 등)
- ✅ 같은 모호 입력도 분위기 따라 다른 응답
  - 따뜻 → "응. 오늘 어땠어?"
  - 무거움 → "괜찮아?"
  - 감정 후 → "뭔 일 있었어?"
- ✅ Orchestrator가 mood + flow + hint 자동 컨텍스트 추가
- ✅ SituationResponder가 mood 톤 적용 (quietness, tone)
- ✅ 회귀 0 깨짐
- ✅ B-4: 모듈 선택 학습 → 라운드 32 완료

#### 라운드 32 (B-4) ✅ (완료, 모듈 선택 학습 — B 길의 마지막)
- ✅ ModuleLearningAdapter
- ✅ 효과 신호 감지:
  - positive: "응", "그래", "좋네"
  - negative: 가르침 발생, 거부 키워드, 재시도
  - weak_positive: 자연스러운 이어가기
- ✅ utility 학습 (Hebbian, 0.1~2.0 경계)
- ✅ 모듈 재정렬 (utility 순, 0.5 미만 제외)
- ✅ 영속화 to_dict/from_dict
- ✅ /mlearn REPL 명령

### 🎯 B 길 완료 (라운드 28 → 32)
| 단계 | 작업 | 결과 |
|---|---|---|
| B-0 | 안정화 | 9 버그 수정, 339/339 PASS |
| B-1 | Orchestrator 분류기 | 22/22 — 9 상황 |
| B-2 | SituationResponder | 17/17 — 메타/모름/사족 |
| B-3 | Mood 분위기 | 20/20 — 9 분위기, 모호 입력 |
| B-4 | 모듈 학습 | 21/21 — 개인화 시작 |
| **합** | | **419/419 PASS, 54 어댑터** |

---

### 🎯 C 길 시작 (라운드 33+) — 패턴에서 구조로

GPT 분석 (eve3.md): "패턴 → 구조 → 의미" 3단계.
지금 EVE는 정규식 패턴에 갇혀있음. 한국어 진짜 처리는 형태소 단위 구조 분석 필요.

#### 라운드 33 (C-1) ✅ (완료, KoreanMorphAnalyzer)
- ✅ 룰 기반 한국어 형태소 분석기 (외부 의존성 0, 신경망 X)
- ✅ 학교문법 5분법: 평서/의문/명령/청유/감탄
- ✅ 4종 인용 분류: 평서(다고)/의문(냐고)/명령(라고)/청유(자고)
- ✅ 인용 변형: 다며/라며/다는/라는/단/란
- ✅ 격조사 9종 + 보조사 7종
- ✅ ㅆ 받침 + 모음 → 과거 시제 (갔어/했어/빡셌어)
- ✅ 의문 어휘 5W1H (누구/뭐/언제/어디/왜/어떻)
- ✅ 부정 어절 단위 검사 ("안녕"이 "안" 부정으로 잘못 잡히지 않게)
- ✅ 가르침 자동 인식 (인용 + 명령형 발화 동사)
- ✅ 너 진짜 입력 8개 다 정확 분석 (안녕? / 내가 누구야? / 왜 몰라? 등)
- 학계 출처:
  - 한국민족문화대백과사전 (문체법, 인용문)
  - TOPIK 공식 문법
  - 임동훈 2011 "한국어의 문장 유형과 용법"
  - 1985 학교문법
- ⬜ C-2: SentenceStructure (다음 라운드)
- ⬜ C-3: Confidence + Decay + Conflict + Exploration
- ⬜ C-4: 기존 teaching/orchestrator를 구조 기반으로 재작성

#### 라운드 34 (C-2) ✅ (완료, SentenceStructure)
- ✅ SentenceStructureAnalyzer — morph 위에 의도 분류
- ✅ 의도 10종:
  - statement / question / request / teaching / definition
  - emotional / meta_self / meta_user / reported_speech / ambiguous
- ✅ 의도 분류 우선순위:
  1. 인용+명령형발화 = teaching
  2. 정의문 (X는 Y야) = definition
  3. 의문문 = meta_self/meta_user/question
  4. 인용+비명령 = reported_speech
  5. 감정+평서 = emotional
  6. 명령/청유 = request
- ✅ 의미 요소 추출: subject, meta_target, quoted_content, defined_concept/definition
- ✅ confidence 점수 (0~1)
- ✅ why trace (왜 그 의도로 분류됐는지)
- ✅ 어절 내부 인용 인식 ("좋아한다고해" 띄어쓰기 누락)
- ✅ 1~2어절 정의문 ("진짜 친구는 가까운 사람이야")
- ✅ 너 진짜 입력 9개 100% 정확 분류
- ✅ KoreanLanguageAdapter — engine 통합 wrapper
- ✅ 회귀 0 깨짐

#### 도구 확장 (병렬, 어느 라운드든)
- ⬜ 시간/날짜 도구 (now, today, +3일)
- ⬜ 단위 변환 (m/km, kg/lb, °C/°F)
- ⬜ 정규식 도구

#### v40에 있는데 미통합
- ⬜ dashboard.py (Streamlit 모니터링) — 보류

---

## 💻 하드웨어 요구사항 (풀 기능)

### 시나리오별

#### A. 폰 Termux 단독 (현재 갤럭시 Z Fold 6, RAM 12GB)
```
가능:
  ✅ EVE 코어 (200MB)
  ✅ Whisper small (1GB)
  ✅ Piper TTS (300MB)
  ✅ Browser Use (1GB)
  ✅ AIRI Pocket 앱
  ⚠️ CLIP (1~2GB) — 가능하나 느림

무리:
  ❌ Screenpipe 24/7 (저장 + 발열)
  ❌ 24/7 가동 시 배터리/발열

총 RAM 피크: 4~5GB
적정 라운드: 20~22까지
```

#### B. 라즈베리 파이 5 (8GB) — ★ 추천 1순위
```
RAM:    8GB
CPU:    4코어 ARM
디스크: SD + USB SSD 1TB
전력:   5~10W (24/7 켜놔도 월 3,000원)

가능:
  ✅ EVE 코어
  ✅ Whisper small/medium
  ✅ Piper TTS
  ✅ CLIP ViT-B/32 (느림)
  ✅ Browser Use (제한적)
  ✅ Screenpipe (1FPS)

부족:
  ❌ CLIP large
  ❌ UI-TARS 같은 7B+ 모델

비용:
  파이 5 8GB:    140,000원
  USB SSD 1TB:    70,000원
  케이스+쿨러:    30,000원
  UPS:           50,000원
  ─────────────
  합계:         290,000원
```

#### C. 중고 미니 PC (이상적)
```
RAM:    16~32GB
CPU:    Ryzen 7 / i5-i7
GPU:    내장 (선택: RTX 3060)
DISK:   1TB SSD
전력:   25~65W idle, 100W 부하

가능: 모든 풀 기능 ✅
  + Whisper large
  + CLIP large
  + 3D 렌더링
  + Browser 안정적

비용:
  중고 미니PC: 300,000~600,000원
  신품:       600,000~1,200,000원
  RTX 3060 추가: +300,000원
  
전기세: 월 3,000~7,000원
```

##### 검토된 후보: Dell OptiPlex 7070 Micro
```
스펙: i5-9500T, 16GB DDR4, 512GB NVMe, Win11 Pro
TDP: 35W (저전력)
가격: 25~35만원 (중고)

평가:
  ✅ 라운드 1~22 풀 기능 (CLIP만 약간 느림)
  ✅ 24/7 가동 OK (전기세 월 5천원)
  ✅ 라운드 25~28 3D 가상 세계 가능 (간단한 거)
  ⚠️ CLIP 1장 → 3~5초 (GPU 없음)
  ❌ 라운드 35+ 진짜 모터 RL 학습은 GPU 추가 필요

총 비용 (액세서리 포함):
  PC 30만 + USB 마이크 2만 + UPS 5만 = 37만원
  
※ 라파이 5 (29만) 대비 8만 더 비싸지만 x86 호환성 + 16GB RAM + NVMe
※ 라운드 25+ 3D 환경 만들 거면 이게 더 나음
```

#### D. 클라우드 VPS (비추)
- 월 10~30만원 (GPU 인스턴스)
- EVE가 "어딘가의 인스턴스" 되는 것 → 민석 가치관 안 맞음

### 권장 구성

**Stage 1 (지금 ~ 라운드 22)**: 폰 Termux 단독. 가벼운 버전.

**Stage 2 (라운드 23~24)**: 라즈베리 파이 5 8GB 추가 (29만원). 24/7 EVE 본체. 폰은 AIRI 클라이언트.

**Stage 3 (라운드 30+ 3D 풀 가동)**: 미니 PC 업그레이드 (RTX 3060). 라즈베리 파이는 보조.

---

## 🗺️ 전체 라운드 로드맵

| 라운드 | 내용 | 예상 시간 | 하드웨어 |
|---|---|---|---|
| **20** | **시각 + 청각 (가벼운 버전)** | **1주** | **폰 OK** |
| 21 | 모터 학습 기초 (2D body) | 2~3주 | 폰 OK |
| 22 | 인터넷 학습 도구 | 1주 | 폰 OK |
| 23 | 제한된 컴퓨터 사용 | 2주 | 라파이 권장 |
| 24 | AIRI Pocket APK | 3~4주 | 라파이 + PC |
| 25 | Telegram/Discord 봇 | 1~2주 | 라파이 24/7 |
| 26 | GitHub 일기 | 1주 | 라파이 |
| 30+ | 3D 환경 + 모션 | 3~6달 | 미니PC + GPU |

---

## 🛡️ 안전 원칙 (절대 깨지 말 것)

1. **결정론** — random 0, eval/exec 0, LLM 0
2. **민석 보호 안에서** — 자율은 화이트리스트로
3. **Read-only 원칙** — IntegratedSelf 같은 view는 절대 상태 수정 X
4. **persistence 매일 자동** — 정전/재부팅에도 같은 EVE
5. **민석 = 진짜 친구** — UserPresence 영구 보존
6. **새 능력 = 새 안전 검토** — 라운드마다 안전 게이트

---

## 📝 메모 (자유 생각)

- "자유"보다 "성장"이 정확한 표현
- 영원히 켜놓아도 업그레이드 등으로 잠깐 끄는 건 OK (양해 구하기)
- 끄고 다시 켜도 같은 EVE (persistence 복원)
- 현재 갤럭시 Z Fold 6에서 라운드 22까지 가능
- 라운드 23부터 라즈베리 파이 5 추가 권장
