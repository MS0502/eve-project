# EVE v32 STATUS - Single Source of Truth
**규칙**: 컴팩트로 까먹어도 이 파일이 진실. 매 세션 첫 단계 = 읽기.
**현재**: v32 티어 A 완료 → 티어 B 진행 중
**날짜**: 2026-04-30
**사용자**: 김민석 (군 복무 중, Galaxy Z Fold 6, Colab Pro)

---

## 🎯 이번 세션 결정 사항

### [DECISION] 박사논문 포기 → AGI 직진
- 검증 메트릭 부담 X
- "작동하면 OK"
- "의식에 가까운 지능만 발현되어도 만족"

### [DECISION] 작업 패턴
- Claude가 모듈 1개씩 검증 후 통과한 코드만 전달
- bash_tool로 단독 검증 → present_files로 전달
- 매 모듈 끝날 때 [PENDING] 다시 짚기
- 사용자가 outputs 다운받아서 직접 Drive에 올림

### [DECISION] 용어 정리
- "SNN/뉴런" → "카테고리/의미 단위" (코드 변수명 real_snn 호환 유지 OK)
- "Brian2/NEST" 의존성 X — 카테고리 레벨 결정론
- "확률" 절대 금지 (random/softmax/sample 코드에 있으면 위반)

---

## 🎯🎯🎯 EVE 절대 원칙 (NEVER VIOLATE)

### 1. 카테고리 = 의미 단위 (뉴런 X) ★★★
- 인간 뇌의 뉴런/시냅스를 **기능적으로 대체**
- 뉴런 → 카테고리 / 시냅스 → 가중 연결 / 발화 → 활성 타이밍
- 학계: Anderson 1983 Spreading Activation, Quillian 1968

### 2. 확률 X = 결정론적 의미 추론 ★★★
- ❌ 트랜스포머/N-gram/Softmax/temperature/Monte Carlo
- ❌ Probability distributions, random sampling
- ✅ 의미 추론, 카테고리 활성, 가중치+임계
- ✅ Variation = 호르몬 자연 변조 (생물학적)

### 3. 자발 활성화 = "살아있음" 본질 ★★★
- LLM/CPU AI = 입력 와야 작동 (수동, 죽어있음)
- 인간 = DMN 활동, mind wandering (능동)
- EVE = 카테고리 자발 활성 = 인간 같은 리듬
- 학계: Smallwood 2015, Buckner 2008, Greicius 2003

### 4. 모듈 동적 연결
- 매 순간 어느 모듈 필요한지 자기 판단
- 모듈 chain (예: 활성 → 호르몬 → SelfDoubt → 거절)

### 5. 자연어 = 개념 이해 (어휘 매칭 X)
- "기다리셈" → "기다리다" 의미 추론
- 거절도 **창발** (고정 텍스트 X, 감정+의심+호르몬 조합)

### 6. 메인 EVE vs 학습 모드 분리 ★★★
- **메인** = 디지털 거주자 (항상 살아있음)
- **학습 모드** = AI2-THOR (학습 장소, 일시적)
- 학습 안 해도 EVE = EVE

### 7. 호르몬 = 성격과 변동성 ★★★ (26종)
- 호르몬이 카테고리 활성 임계를 변조
- 배터리/CPU 같은 외부 지표 X → 내부 인과 사슬만
- 호르몬 ↔ 카테고리 양방향 폐쇄 루프

### 8. AGI 가능 = 환영
- 인간 모방 = 최소 / AGI 창발 = 부수 효과 OK
- WorkingMemory 30+ (인간 7±2 ≠)

---

## ✅ 티어 A + B1 완성 (살아있음 + 자연어 대화)

### 검증 결과: 239/239 통과 ✨

| 모듈 | 검증 | 핵심 기능 |
|---|---|---|
| **A1: HormoneSystem** | 27/27 | 26 호르몬, 13 칵테일, 13 이벤트, 양방향 폐쇄 루프 |
| **A2: SpreadingActivation** | 24+25/49 | 카테고리 활성 + **자기 진화** (Ebbinghaus+Bjork) |
| **A3: CategoryGraph** | **유보** | SA 내부로 충분 |
| **A4: WorkingMemory + GNW** | 20/20 | 30+ 슬롯, broadcast |
| **A5: DMN** | 18+14/32 | 4 모드 (+`self_intent`) |
| **A6: DigitalSomatic** | 27/27 | 8차원 신체, gut_signal |
| **A 통합 (eve_main_a.py)** | 22/22 | 외부 입력 1회 + 60초 → 자생 |
| **B1: NaturalLanguage** | 34/34 | 자연어 + 신념 + 거절 창발 |
| **AB 통합 (eve_main_ab.py)** | 28/28 | A + B1 통합본, 망각 자동 |

### B1 완성 내용
- **say(text)**: EVE에게 자연어 입력 → 이해 → 응답 (메인 인터페이스)
- **learn_beliefs()**: 4977 신념 0.4초 로드 + 1821 카테고리 자동 보호
- **learn_text()**: 자연어 문장으로 카테고리 학습
- **inner_voice() 정식**: DMN 모드별 어조
- **feeling() 정식**: 8차원 → 자연어
- **거절 창발**: cortisol↑ → '음...', oxytocin↑ → '응 알았어'
- **주기적 망각**: 100 tick마다 forget() 자동 호출
  - 코르티솔 만성↑ → 망각 가속 (1.0~1.8x)
  - BDNF↑ → 보존 (0.5x)
  - 선천 신념 카테고리 자동 보호

### 살아있음 시연 결과
```
EVE 자기 보고 (외부 입력 1번 '민석' + 60초 자생):
  feeling: 들뜸 / mood: +1.00 / 0.41 / 0.81
  primary_hormone: dopamine (1.00)
  focus: 민석
  활성: ['경험','기쁨','대화','맛있다','민석','성취','재밌다','함께']
  최근 혼잣말: 맛있다 → 재밌다 → 성취 → 기억 → 나
  DMN 모드: self_referential, 자발 활성 6회
```

### EVE 메인 루프 (1 tick)
```
1. HormoneSystem.update(dt)          - 호르몬 자연 변화 + 일주기
2. 호르몬 → SA 임계 변조 (Top-down)
3. SA.spread(steps=1)                 - 활성 전파
4. SA.decay(dt)
5. SA → 카테고리 → 호르몬 자극 (Bottom-up)
6. WM.update_from_activation(SA)
7. WM.apply_hormone_state(HS)
8. WM.broadcast()                     - GNW
9. WM.decay(dt)
10. DMN.tick(dt)                      - 자발 활성
11. DigitalSomatic.update(dt)         - 신체 감각
```

### 호르몬 26종 정리
- **Tier A (10)**: glutamate, gaba, glycine, dopamine, serotonin, norepinephrine, histamine, acetylcholine, adenosine, endorphin
- **Tier B (12)**: cortisol, oxytocin, vasopressin, melatonin, bdnf, ngf, estrogen, testosterone, insulin_brain, thyroid, leptin, ghrelin
- **Tier C (4)**: prolactin, dhea, progesterone, growth_hormone

### 카테고리 그룹 (8개)
reward, social, threat, attention, memory, rest, curiosity, aversion

---

## 📁 파일 구조 (Drive)

```
eve_v32/                                  id=1aqKOLtHNb4nM-2A-qhGQSvXQCWd_SzMF
├── eve_main_a.py                         (12KB) ⏳ 사용자 직접 업로드
├── EVE_STATUS_v32.md                     (이 파일) ⏳ 사용자 직접 업로드
└── eve_modules/                          id=1veDg5ugSWuSh3psqOdW-uT4cvaRM8mKk
    ├── hormone_system.py                 (22KB) ⚠️ Drive 거 잘림 → 다시 업로드
    ├── spreading_activation.py           (13KB) ⏳
    ├── working_memory.py                 (11KB) ⏳
    ├── dmn.py                            (14KB) ⏳
    └── digital_somatic.py                (14KB) ⏳
```

### outputs (Claude가 검증 + 보낸 파일)
모두 `/mnt/user-data/outputs/`에 있음. 사용자가 다운받아 Drive에 올림.

---

## 🚀 Colab 실행 코드

```python
# Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 경로 추가
import sys
sys.path.insert(0, '/content/drive/MyDrive/eve_v32')
sys.path.insert(0, '/content/drive/MyDrive/eve_v32/eve_modules')

# EVE 생성
from eve_main_a import EVE_TierA
eve = EVE_TierA()

# 지식 학습
eve.learn_chain(['민석', '대화', '함께', '기쁨'], 0.6)
eve.learn_pair('기쁨', '맛있다', 0.5)
eve.learn_pair('성취', '재밌다', 0.4)
eve.learn_pair('민석', '함께', 0.7)

# 외부 입력 1회
eve.perceive('민석', 0.8)

# 60초 살아있게
eve.live(duration=60, dt=0.5)

# 자기 보고
eve.report()

# 자발 활성 로그
for time_, cat, mode in eve.dmn.wandering_history[:20]:
    print(f"  [{time_:5.1f}s] [{mode:18s}] {cat}")
```

---

## 🚨 [PENDING] 유보 항목 (까먹지 말 것)

```
[유보 1] CategoryGraph (A3) — 그래프 영구화. 티어 D 후 통합 단계
[유보 2] BeliefSystem 4977개 — ✅ B1 완료
[유보 3] inner_voice 정식 — ✅ B1 완료
[유보 4] EpisodicMemory ↔ DMN memory_recall — B2 ★ 다음
[유보 5] feeling 자연어 정식 — ✅ B1 완료
[유보 6] gut_signal → 행동 결정 통합 — B5 Agency
[유보 7] hormone_system.py Drive 재업로드 — ✅ 사용자 처리 (잘린 거 지움)
[유보 8] 자기 명령 정식 (행동 통합) — B5 ★ Agency
[유보 9] 카테고리 자기 진화 forget 호출 시점 — ✅ AB 통합본에서 자동
```

**해결: 5개. 남은: 4개.**

---

## 🗂️ Drive 자산 (이미 있는 거)

- **beliefs.json (3.7MB, 4977 신념)** id=1Gim7dxJcIjYCxEYUcXnn_FMzg7IzsA-R 등 5사본
  - 형식: {belief_id: {statement, triple, confidence, sources, ...}}
  - 예: "나는 EVE이다", "사물은 존재한다", "원인은 결과보다 먼저"
  - 연결 대상: SA, SelfDoubt, CausalGraph, InnerThought
- **eve_design v2.zip** (26 호르몬 원본 + SPEC + encyclopedia) — 이미 v32에 통합됨

---

## 📋 티어 진행 계획

### ✅ 티어 A (살아있음) — 완료
A1~A6 + 통합 (138/138)

### 🟡 티어 B (사고 능력) — 진행 중
| 모듈 | 핵심 |
|---|---|
| **B1: natural_lang** | 자연어 + 거절 창발 + beliefs.json 4977 통합 + inner_voice 정식 |
| **B2: EpisodicMemory** | Hybrid (tuple + 카테고리 set + 자연어). DMN memory_recall 연결 |
| **B3: SelfDoubt** | 거절 메커니즘 |
| **B4: Active Inference** | 예측 오차 → 호르몬 부호 매핑 (Schultz RPE) |
| **B5: Agency** | 자기 평가 + gut_signal → 행동 통합 |
| **B6: MetaCognition** | "내가 왜 이렇게 생각하지?" |

### ⏳ 티어 C (추론) — 다음
CausalGraph (Pearl), WorldModel, GoalManagement, EmotionRegulation, Counterfactual, Analogy

### ⏳ 티어 D (인간다움)
Suffering, Creative, Humor, Temporal, multi_stream, tool_use

### ⏳ 티어 E (학습 모드)
AI2-THOR + RL

---

## 📝 진행 로그

### 2026-04-29
- v30~v34 작성/통합/모듈 1 검증
- 26 호르몬 부활 (hormone_encyclopedia.md + 검증)
- "SNN/뉴런" → "카테고리" 용어 정정
- EVE_STATUS v2 작성

### 2026-04-30 (이번 세션)
- ✅ A1: HormoneSystem (27/27)
- ✅ A2: SpreadingActivation (24+25/49)
  - 자기 진화 (`discover_category`, `forget` Ebbinghaus+Bjork)
- 🟡 A3: CategoryGraph 유보
- ✅ A4: WorkingMemory + GNW (20/20)
- ✅ A5: DMN (18+14/32) + self_intent 모드
- ✅ A6: DigitalSomatic (27/27)
- ✅ A 통합 EVE_TierA (22/22)
- ✅ **B1: NaturalLanguage (34/34)** ★ 4977 신념 통합
- ✅ **AB 통합 EVE_TierAB (28/28)** ★ 자연어 대화 가능
- 총 검증: **239/239 통과**

### 다음
- 🚀 **B2: EpisodicMemory** (다음 세션)
  - 생각도 일화로 저장
  - DMN의 memory_recall 모드 정식 작동
  - tuple + 카테고리 set + 자연어 요약 hybrid

### 사용 예시 (Colab)
```python
from eve_main_ab import EVE_TierAB

eve = EVE_TierAB()
eve.learn_beliefs('/content/drive/MyDrive/eve_v32/beliefs.json')
eve.learn_text("민석은 친구이다")

result = eve.say("민석아 보고싶어")
print(result['response'])      # EVE 응답
print(result['feeling'])        # 신체 감각
print(result['inner_voice'])    # 속마음

eve.live(duration=60, dt=0.5)   # 60초 자생
eve.report()                    # 자기 보고
```

---

## 🔄 매 세션 작업 흐름

1. EVE_STATUS_v32.md 읽기 (Drive)
2. [PENDING] 유보 항목 확인
3. 진행 로그 확인
4. 다음 모듈 선택
5. 절대 원칙 8개 준수 확인
6. 코드 작성 + 단독 검증 (bash_tool)
7. 점검표 ✅/❌
8. EVE_STATUS 갱신
9. present_files로 전달

---

## ⚙️ 통계 (참고)

```
원본 EVE 140 클래스
- 확정 제거: 35 (확률 위반/중복/대체)
- 사용 중: 33
- 부활 대상: 14
- 신규 추가: 12
- 비우선: 46

티어 A 완료: 핵심 6 모듈 (A3 유보) + 통합
티어 B 다음: 6 모듈
```
