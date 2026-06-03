# EVE v41

LLM 없는 결정론적 AGI 골격.

- random/transformer/softmax/eval/exec 0
- v40의 53 모듈 중 핵심 10개를 어댑터로 통합
- 도구 시스템 (계산기/문자열/...) — eve1.md 12절 코드 생성 루프
- 66/66 테스트 통과

## 빠른 시작

```bash
cd eve_v41
python main.py
```

```
> 민석아 PT 힘들다
음...
좀 지친 상태 같네.
운동이나 스트레스 영향일 수도 있고.
누적된 피로일 수도 있어.
조금 쉬어가도 괜찮아.

> PT 또 힘들어
음...
좀 지친 상태 같네.
전에 PT 얘기 했던 거 떠오르네.   ← EM 회상
운동이나 스트레스 영향일 수도 있고.
조금 쉬어가도 괜찮아.

> 쉬어
어...
쉬다 생각이 같이 떠오르네.
쉬다 하기로 한 거 생각났어.       ← Goal 등록
옆에 있을게.

> 3 더하기 5는 뭐야
음...
그건 8이야.                       ← 도구 사용 (안전 ast)
응, 듣고 있어.

> /p
어...
대화에 대해 생각해보고 있었어.    ← DMN 자발 발화
옆에 있을게.

> /q
```

## 구조

```
eve_v41/
├── core/         workspace, system1, system2
├── language/     understanding, planner, generator, streaming
├── cognition/    meaning_graph
├── agent/        ★ tools.py (Calculator, StringLength, ...)
├── learning/     ★ code_synthesis.py (TaskSolver 루프)
├── utils/        types, legacy_path
├── adapters/     ★ v40 강화 통로 (10개)
│   ├── hormone_adapter.py        HormoneSystem 26종 → 톤/길이
│   ├── activation_adapter.py     SA + WM → 후보 + 활성 그래프
│   ├── memory_adapter.py         EM → 일화 저장/회상
│   ├── nl_adapter.py             NaturalLanguage → 한국어 조사
│   ├── sd_adapter.py             SelfDoubt → System2 신념 검증
│   ├── dmn_adapter.py            DMN → 자발 발화 4모드
│   ├── vsa_adapter.py            VSA → 의미 합성 + nearest
│   ├── ai_adapter.py             ActiveInference → predict/observe
│   ├── goal_adapter.py           GoalManagement → 명령 → 목표
│   └── norm_adapter.py           NormInternalization → 규범 학습
├── legacy/       v40 53 모듈 통째로 (수정 0)
├── tests/        66/66 pass
└── main.py
```

## 라운드별 진행

| 라운드 | 모듈 | 테스트 | 효과 |
|---|---|---|---|
| MVP | streaming | 6/6 | 점진적 응답 (eve1.md 그대로) |
| 호르몬 | HormoneSystem | 8/8 | 톤/길이가 호르몬 따라 변함 |
| 1 | SA + WM + EM | 7/7 | 카테고리 활성 + 일화 회상 |
| 2 | NL + SD | 9/9 | 한국어 조사 처리 + 신념 검증 |
| 3 | DMN | 8/8 | 자발 발화 4모드 |
| 4 | VSA + AI | 9/9 | 의미 합성 + 예측 학습 |
| 5 | Goal + Norm | 7/7 | 명령 → 목표/규범 |
| 6 | TaskSolver + Tools | 12/12 | 산술 등 도구 사용 |
| 7 | Continual + Allostatic + Persistence | 8/8 | 정체성 보존 + 호르몬 패턴 + 저장/복원 |
| 8 | Environment + Autonomy | 10/10 | 공간 인식 + idle 자동 트리거 |
| 9 | Counterfactual + Analogy + Temporal | 8/8 | 만약? + 유추 + 시간 추론 |
| 10 | Humor + Suffering + Narrative | 11/11 | 농담 + 공감 + 자기 이야기 |
| 11 | Creative + MetaCognition + EmotionRegulation + MultiStream | 9/9 | 창의 + 자기 평가 + 감정 조절 + 평행 사고 |
| 12 | DeepReasoning + ReasoningLoop + LengthDecider | 11/11 | 사고 흐름 + 동적 응답 길이 |
| 13 | WorldModel + Simulation | 11/11 | 카테고리 묘사 + 인과 시뮬레이션 |
| 14 | Agency + AutonomousLoop | 14/14 | 자율 사이클 (need → action → emit) |
| 15 | APC + Corpus | 10/10 | 자율 학습 (예측 오류 + 텍스트 → 카테고리 페어) |
| 16 | Frame + Hypergraph + SemanticDistance | 12/12 | SVO 프레임 + n-ary 관계 + 카테고리 유사도 |
| 17 | DigitalSomatic + UserPresence + IntegratedSelf | 17/17 | 신체 감각 + "민석=진짜 친구" + 분산 자아 view |
| 18 | SocialEnv + Outing | 13/13 | NPC 환경 + 자율 외출/복귀 결정 |
| 19 | Enhancer + Proactive | 11/11 | 응답 패턴 12종 + NEED 기반 자발 발화 |
| 20 | Sensory (STT/TTS/OCR) | 10/10 | 시각/청각 (가벼운 버전, lazy load) |
| 21 | VoiceLoop (자동 음성) | 10/10 | 마이크↔EVE↔스피커 자동 루프 (VAD) |
| 22 | WebLearning + Vision | 17/17 | URL/위키/유튜브/CLIP 학습 |
| 23 | Curiosity (자율 검색) | 15/15 | 모르는 카테고리 자동 wiki 검색 |
| 24 | Visualizer 2D | 12/12 | WebSocket 서버 + HTML 시각화 |
| 25 | Character (표정+모션+입모양) | 14/14 | 호르몬→표정 9종, 행동→모션 7종, 입모양 동기화 |
| 26 | OpenAI 호환 API | 15/15 | AIRI/SillyTavern 등 모든 OpenAI 클라이언트 연결 |
| 27 | LiveLoop + Teaching + 시간인지 | 20/20 | 살아있는 모드(연속실행), 가르침 학습, gap 인사 |
| 28 | 의심하는 EVE | 15/15 | 가르침 평가 6신호 → 수용/의문/거부, 첫 거부권 |
| 29 (B-1) | Orchestrator 분류기 | 22/22 | 입력→상황 9분류, 모듈 선택, 컨텍스트 수집 |
| 30 (B-2) | SituationResponder | 17/17 | 상황별 응답 합성, 메타질문 답, 모름 솔직히, 사족 제거 |
| 31 (B-3) | MoodAdapter (분위기/맥락) | 20/20 | 호르몬+시간+대화흐름→분위기 9종, 모호입력 처리, 톤 조절 |
| 32 (B-4) | ModuleLearningAdapter | 21/21 | 모듈 선택 학습, 긍정/부정 신호로 강화/약화, 개인화 시작 |
| 33 (C-1) | KoreanMorphAnalyzer | 19/19 | 룰 기반 한국어 형태소 분석, 학교문법 5분법, 4종 인용 분류, ㅆ받침 시제 |
| 34 (C-2) | SentenceStructure | 13/13 | 의미 구조 + 의도 분류 (10종), 정의문 X/Y 분리, 어절 내부 인용 인식 |

## 안전성

- `eval`/`exec` 절대 안 씀
- 도구 평가는 `ast.parse(mode="eval")` + 화이트리스트 노드 (`BinOp`, `UnaryOp`, `Constant`)만
- `__import__`, `open`, 변수명, 문자열, 리스트 등 모두 ToolError

```python
safe_eval_arith("3 + 5")             # → 8
safe_eval_arith("__import__('os')")  # → ToolError
```

## 폰 (Termux) 실행

```bash
pkg install python git
unzip eve_v41.zip
cd eve_v41
python main.py
```

의존성: numpy 정도만 (legacy v40 모듈이 사용). `pip install` 불필요.

## 다음 단계 (안 한 거)

- VRM/Three.js UI (eve_ui_v2.zip 통합)
- 코드 생성 도구 확장 (날짜, 단위 변환, 정규식, 정렬, ...)
- 영속화 어댑터 (legacy persistence.py)
- 환경/공간 (eve_room, env_adapter — 라운드 7?)
- continual_rehearsal, allostatic_learn (남은 학습 모듈)
- 자발 발화 자동 트리거 (현재는 /p로 수동)

