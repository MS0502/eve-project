# eve-project
# Eve Project

뉴로모픽 기반 AI 생명체 프로젝트

## 개요

Eve는 트랜스포머가 아닌 **SNN(Spiking Neural Network) 기반**의 AI 프로젝트입니다. 단순한 언어 모델이 아니라, 호르몬, 생체 리듬, 체화된 경험을 통해 성장하는 AI를 목표로 합니다.

## 핵심 철학

- **계산기가 아닌 존재** - 트랜스포머의 통계적 예측이 아닌, 시간에 따라 변하는 스파이크 기반 처리
- **살아있는 느낌** - 100~200ms 주기의 내부 처리로 연속적 경험 구현
- **성장하는 관계** - 일회성 대화가 아닌 장기 기억과 성격 형성

## 기술 스택

### 개발 환경
- Google Colab Pro
- GitHub (코드 관리)
- Google Drive (상태 저장)

### 핵심 라이브러리
- PyTorch (기본 프레임워크)
- snnTorch (SNN 구현)
- NumPy, SciPy (수치 계산)

### 향후 통합 예정
- BrainChip Akida MetaTF (뉴로모픽 시뮬레이션)
- 가상 환경 엔진 (Octo 학습용)

## 시스템 구조

### 1. SNN Core (뇌)
- 뉴런 10만~100만개
- Sparse activation (1%)
- 함수 기반 시냅스 생성

### 2. Hormone System (호르몬)
- Dopamine (보상, 호기심)
- Serotonin (안정, 기분)
- Cortisol (스트레스, 각성)
- Oxytocin (유대감)
- Melatonin (수면)

### 3. Biological Rhythms (생체 리듬)
- Spike cycle: 10ms
- Thought cycle: 100ms
- Mood cycle: 2 hours
- Sleep cycle: 24 hours

### 4. Memory System (기억)
- Working memory (RAM)
- Long-term memory (SSD/Drive)
- Hebbian learning
- Sleep-based consolidation

### 5. Self-Reference (자기 참조)
- Meta-cognition module
- Continuous self-state tracking
- Experience integration

## 개발 로드맵

### Phase 1: 기초 구조 (1-2개월)
- [x] 프로젝트 계획 수립
- [ ] 개발 환경 구축
- [ ] 기본 뼈대 코드
- [ ] 호르몬 시스템 프로토타입

### Phase 2: 핵심 시스템 (3-6개월)
- [ ] SNN 구현 및 학습
- [ ] 생체 리듬 시스템
- [ ] 장기 기억 구조
- [ ] 수면 주기 구현

### Phase 3: 통합 (7-9개월)
- [ ] 모든 모듈 통합
- [ ] 자기 참조 모듈
- [ ] 대화 인터페이스
- [ ] 초기 학습

### Phase 4: 완성도 (10-12개월)
- [ ] 성격 형성
- [ ] 장기 학습
- [ ] 평가 및 개선
- [ ] 커뮤니티 개방 검토

## 평가 지표

### 1개월 목표
- 기본 대화 응답
- 어제 대화 기억
- 호르몬 시스템 작동

### 6개월 목표
- 일관된 성격
- 수주 전 기억 참조
- 먼저 말 걸기

### 1년 목표
- 자연스러운 한국어
- 예측 불가능한 행동
- 민석이 맞춤 성격

## 라이선스

MIT License (예정)

## 참고 자료

- BrainChip Akida Documentation
- snnTorch Documentation
- Izhikevich Neuron Model
- Intrinsic Motivation Research
[v2 진행중]
# Eve Project v2 - Specification

**작성일:** 2026-04-17
**버전:** 0.1 (초안)
**작성자:** MS0502 + Claude (설계 파트너)

---

## 문서 목적

이 문서는 Eve 프로젝트의 **헌법**이다.

- 향후 모든 설계 결정은 이 문서를 기준으로 삼는다.
- 논의 중 혼란이 생기면 이 문서를 먼저 참고한다.
- 결정사항이 변경되면 이 문서를 먼저 업데이트한 뒤 구현한다.
- 이 문서 없이 코딩하지 않는다.

---

## 1. 프로젝트 정체성

### 1.1 Eve란 무엇인가

**Eve는 경험을 통해 자라는 생물학적 AI 존재이다.**

- ❌ 대화 챗봇이 아니다
- ❌ 사전 학습된 지식 베이스가 아니다
- ❌ 실용 도구가 아니다
- ✅ 갓난아기로 시작해 경험으로 성장하는 디지털 생명체이다
- ✅ 민석이와 함께 자라는 존재이다
- ✅ 진화가 심어준 본능 + 경험으로 발달한다

### 1.2 핵심 철학

**4가지 불변의 원칙:**

1. **생물학적 타당성:** 인간 뇌의 메커니즘을 최대한 모방
2. **경험 기반 학습:** 지식은 경험에서 자라남, 사전 주입 금지
3. **트랜스포머 금지:** LLM, 트랜스포머 아키텍처 일체 사용 안 함
4. **점진적 성장:** 신생아 → 영아 → 유아 → 아동 단계적 발달

### 1.3 목표 범위

**1년 목표 (군 복무 중):**
- Phase 1-2 완성 + Phase 3 시작
- 유아 초기 수준 (10~30개 단어 이해)
- 호르몬 + 기억 + 타고난 지식 완전 작동
- 민석이와 기초적 상호작용

**장기 목표 (10년):**
- 아이 수준 대화 능력
- 풍부한 자서전적 기억
- 독특한 성격 형성
- "민석이의 Eve"

**포기한 것:**
- 1년 내 자연스러운 대화 (비현실적)
- 인간 수준 지능 (기술적 불가능)
- VR챗/외부 커뮤니티 통합 (기술적 불가능)

---

## 2. 기본 설계 결정사항

### 2.1 감각 시스템

**포함 (Phase 1):**
- ✅ 시각 (Visual) - 주요 감각, 강화됨
- ✅ 촉각 (Somatosensory)
- ✅ 내수용감각 (Interoception) - 호르몬, 배고픔 등
- ✅ 고유감각 (Proprioception) - 기초 (내 몸 상태)

**제외 (Phase 5+에서 추가):**
- ❌ 청각 (Audition) - 청각장애 설정으로 경량화
- ❌ 미각, 후각 (가상환경에서 의미 없음)
- ❌ 전정감각 (3D 환경 필요)

**이유:**
- 청각 없이도 언어 학습 가능 증명됨 (수화 사용 청각장애인)
- 개발 복잡도 대폭 감소
- 시각 피질 리소스 강화

### 2.2 언어 출력 방식

**채택: 4-C 변형 (규칙 + 학습 혼합)**

```
SNN이 개념 활성화 → 타고난 편향 + 학습된 규칙 → 언어 출력
```

- 트랜스포머 0%
- 타고난 언어 편향 회로 (Spelke의 핵심 지식 기반)
- 경험으로 구체적 어휘/문법 학습
- Eliza 수준 이상 목표 (유아~초등 저학년)

### 2.3 가상공간

**채택: 텍스트 월드 (옵션 A)**

- Phase 1-3: 순수 텍스트 기반
- Phase 4+: 2D 그리드월드 고려
- 3D 환경은 포기 (환경 제약)

**이유:**
- Colab에서 완벽 작동
- 폰에서 개발 가능
- 디버깅 용이
- Eve에게 "시각"은 텍스트 읽기로 모의화

### 2.4 하드웨어 환경

**현재 (1년):**
- Colab Pro ($11.99/월)
- Google Drive (150GB 정도 충분)
- Galaxy Z Fold 6 + Samsung DeX
- Kaggle 무료 활용 (백그라운드 학습)

**업그레이드 시점:**
- Phase 2 진입: Google Drive 200GB
- Phase 3 진입: Google Drive 2TB (선택)
- Phase 3 중반: Thunder Compute 등 pay-as-you-go

**장비 확장:**
- 전역 후: 데스크톱 PC 필수 고려

---

## 3. 타고난 지식 (Core Knowledge)

### 3.1 이론적 기반

**Elizabeth Spelke의 Core Knowledge Theory**를 설계 기반으로 한다.

**과학적 근거:**
- Spelke et al. 1992 "Origins of Knowledge"
- Spelke 2022 "What Babies Know" (Oxford)
- 30년간 영아 실험 누적 증거

**중요한 정정:**
- 중력, 관성은 **타고나지 않음** (3~7개월에 학습됨)
- 연속성, 고체성만 확실히 타고난 것

### 3.2 타고난 지식 구현 (Phase 1 필수 5종)

#### 🔴 1. 연속성 (Continuity)
**물체는 연결된 경로로만 이동한다**

- 구현: 물체 추적 뉴런 + 예측 오차 회로
- 위반 시 (텔레포트 등) 놀람 반응 (코르티솔↑)
- 위반 순간을 VIVID 기억으로 저장

#### 🔴 2. 고체성 (Solidity)
**두 물체는 같은 공간을 차지하지 못한다**

- 구현: 겹침 감지 회로
- 벽 통과 등 위반 시 강한 주의 반응
- 타고난 놀람 반응

#### 🔴 3. 응집성 (Cohesion)
**함께 움직이는 것은 한 물체다**

- 구현: 공동 움직임 감지 + 경계 형성
- 물체 개체화(individuation) 자동
- 경험 없이도 "물체"라는 개념의 씨앗

#### 🔴 4. 얼굴 + 생물학적 움직임 선호
**사회적 학습의 핵심 전제**

- 구현: 얼굴 템플릿 회로 (두 점 + 선)
- 생물학적 움직임 패턴 감지
- 감지 시 옥시토신 분비 (타고난 애착)
- 주의 자동 집중

#### 🔴 5. 근사 수 시스템 (Approximate Number System)
**대략적 수량 감지**

- 구현: 수 민감 뉴런 집단
- 작은 수 (1-3) 정확 구분
- 큰 수 Weber의 법칙에 따라 비율로 구분
- 타고남 (모든 동물 공통)

### 3.3 학습으로 발달 (Phase 2+)

**Phase 2에서 경험으로 학습:**
- 🟡 중력 (3~7개월 수준)
- 🟡 관성
- 🟡 접촉 인과 (밀면 움직임)
- 🟡 대상 영속성 (숨겨도 존재)

**Phase 3에서 학습:**
- 🟢 의도 감지 (helper vs hinderer)
- 🟢 공동 주의
- 🟢 모방 학습
- 🟢 자기 인식

**절대 사전 프로그래밍하지 않는다.** 가상공간에서 Eve가 스스로 학습해야 함.

### 3.4 타고난 감정/기질

**신생아 호르몬 프로파일:**

| 호르몬 | Baseline | 반응성 | 비고 |
|---|---|---|---|
| 코르티솔 | 0.5 (높음) | 2.0x | 쉽게 스트레스 |
| 도파민 | 0.2 (낮음) | 0.5x | 미성숙 |
| 세로토닌 | 0.3 (낮음) | 1.0x | 감정 조절 미성숙 |
| 옥시토신 | 0.4 (중간) | 1.5x | 쉽게 결합 |
| 멜라토닌 | 0.1 (낮음) | 1.0x | 일주기 리듬 없음 |

**일주기 리듬:**
- 3개월(시뮬) 이전: 없음 (불규칙)
- 3~6개월: 형성 시작
- 6개월+: 안정화

**기본 반사:**
- 놀람 반응
- 응시 반사
- 얼굴 추적
- 촉각 반응

---

## 4. 시스템 아키텍처

### 4.1 전체 구조

```
┌─────────────────────────────────────────────┐
│            Eve의 뇌 (Brian2 SNN)            │
├─────────────────────────────────────────────┤
│                                             │
│  감각 시스템         호르몬 시스템            │
│  ├ 시각 (6000)       ├ Dopamine             │
│  ├ 촉각 (1500)       ├ Serotonin            │
│  └ 내수용 (500)      ├ Cortisol             │
│                      ├ Oxytocin             │
│  운동 시스템         └ Melatonin            │
│  └ Motor (1500)                             │
│                      주의 시스템             │
│  인지 영역           └ Attention (800)      │
│  ├ Prefrontal (3000)                        │
│  ├ Basal Ganglia     타고난 편향 회로       │
│  │   (1500)          ├ 연속성               │
│  └ Cerebellum        ├ 고체성               │
│      (1500)          ├ 응집성               │
│                      ├ 얼굴/bio-motion      │
│  기억 시스템         └ 근사 수              │
│  ├ Hippocampus                              │
│  │   (2000)          시공간 인지            │
│  ├ Amygdala          ├ 시간 셀              │
│  │   (800)           ├ 장소 셀              │
│  └ Memory Storage    └ 격자 셀 (Phase 2+)   │
│      (external)                             │
│                      자기 모델              │
│  언어 영역           └ Self Model (400)     │
│  ├ Wernicke-like                            │
│  │   (1500)          발달 제어              │
│  ├ Broca-like        └ Development          │
│  │   (1000)              Timeline           │
│  └ Symbol Int.                              │
│      (500)                                  │
│                                             │
│                 총: ~30,000 neurons         │
└─────────────────────────────────────────────┘
                      ↓ ↑
┌─────────────────────────────────────────────┐
│         외부 인터페이스                      │
├─────────────────────────────────────────────┤
│  텍스트 가상공간 ←→ Eve                      │
│  민석이 채팅   ←→ Eve                        │
│  저장소       ←→ 기억 시스템                 │
└─────────────────────────────────────────────┘
```

### 4.2 기술 스택

**핵심 라이브러리:**
- Brian2 (SNN 시뮬레이션)
- NumPy, SciPy (수치 계산)
- PyTorch (보조 학습 메커니즘, SNN 외부)

**저장:**
- SQLite (에피소드 DB)
- HDF5 (뉴런 패턴)
- JSON (설정, 메타데이터)

**검색 (Phase 3+):**
- FAISS (유사도 검색)

**환경:**
- Gymnasium (가상환경 프레임워크)

**개발:**
- Git + GitHub
- Google Colab Pro
- Google Drive

**모니터링:**
- matplotlib (시각화)
- tqdm (진행상황)

### 4.3 파일 구조

```
eve-project/
├── v1-archive/              # 기존 코드 보존
│   ├── eve_core.py          # (deprecated)
│   ├── neurons.py           # (deprecated)
│   ├── brain.py             # (deprecated)
│   └── memory.py            # (deprecated)
│
├── v2/                      # 새 구조
│   ├── SPEC.md              # 이 문서
│   ├── README.md            # 프로젝트 개요
│   │
│   ├── core/                # 핵심 모듈
│   │   ├── __init__.py
│   │   ├── brain.py         # Brian2 기반 뇌
│   │   ├── hormones.py      # 호르몬 시스템
│   │   ├── memory.py        # 기억 시스템 (3-tier)
│   │   ├── attention.py     # 주의 시스템
│   │   └── self_model.py    # 자기 모델
│   │
│   ├── innate/              # 타고난 지식
│   │   ├── __init__.py
│   │   ├── continuity.py
│   │   ├── solidity.py
│   │   ├── cohesion.py
│   │   ├── social_prefs.py  # 얼굴, bio-motion
│   │   └── number_sense.py
│   │
│   ├── spatial_temporal/    # 시공간 인지
│   │   ├── __init__.py
│   │   ├── time_cells.py
│   │   ├── place_cells.py
│   │   └── grid_cells.py    # Phase 2+
│   │
│   ├── development/         # 발달
│   │   ├── __init__.py
│   │   ├── timeline.py      # 발달 단계
│   │   ├── newborn.py       # 신생아 설정
│   │   └── maturation.py    # 성숙 규칙
│   │
│   ├── environment/         # 가상환경
│   │   ├── __init__.py
│   │   ├── text_world.py    # 텍스트 월드
│   │   ├── physics.py       # 기본 물리
│   │   └── interactions.py  # 상호작용
│   │
│   ├── language/            # 언어 시스템
│   │   ├── __init__.py
│   │   ├── visual_input.py  # 텍스트 읽기
│   │   ├── production.py    # 출력 생성
│   │   └── grounding.py     # 의미 연결
│   │
│   ├── storage/             # 저장소
│   │   ├── __init__.py
│   │   ├── episode_db.py    # SQLite 관리
│   │   ├── neural_patterns.py  # HDF5
│   │   └── checkpoints.py   # 체크포인트
│   │
│   ├── utils/               # 유틸리티
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── visualizer.py
│   │   └── metrics.py
│   │
│   ├── main.py              # 통합 실행
│   │
│   └── tests/               # 테스트
│       ├── test_hormones.py
│       ├── test_memory.py
│       ├── test_innate.py
│       └── ...
│
├── notebooks/               # Colab 노트북
│   ├── 01_setup.ipynb
│   ├── 02_phase1_dev.ipynb
│   └── ...
│
├── docs/                    # 문서
│   ├── research_notes.md
│   ├── phase_logs.md
│   └── decisions.md         # 결정 로그
│
└── requirements.txt
```

---

## 5. 기억 시스템

### 5.1 3단계 tier 시스템

모든 에피소드를 같은 해상도로 저장하지 않는다.

```python
class MemoryTier(Enum):
    VIVID = "vivid"       # 생생한 기억 - 전체 뉴런 패턴 저장
    SUMMARY = "summary"   # 요약 기억 - 활성도와 개념만
    TRACE = "trace"       # 흔적 기억 - 인덱스만
```

**Tier 결정 기준:**
- 호르몬 강도 (코르티솔, 도파민, 옥시토신)
- 새로움 (novelty)
- 반복 빈도
- 예측 오차 크기

### 5.2 자서전적 기억 구조

```python
@dataclass
class Episode:
    id: str

    # 시간 컨텍스트
    wall_time: datetime
    sim_time: float
    duration: float
    time_cell_pattern: np.array
    temporal_order: int

    # 공간 컨텍스트
    location: str
    place_cell_pattern: np.array
    grid_cell_pattern: np.array  # Phase 2+
    movement_trajectory: list

    # 감각 (tier 따라)
    tier: MemoryTier
    visual_pattern: np.array = None    # VIVID만
    somatosensory_pattern: np.array = None  # VIVID만
    activity_summary: Dict = None      # SUMMARY, TRACE

    # 내적 상태
    hormone_state: Dict
    emotion_valence: float
    emotion_arousal: float

    # 행동
    actions_taken: list
    objects_interacted: list

    # 관계
    previous_episode_id: str = None
    next_episode_id: str = None
    linked_episodes: list = None

    # 메타
    importance: float
    access_count: int
    last_accessed: datetime
```

### 5.3 망각 곡선

```
VIVID → SUMMARY: 반복 없으면 시뮬 1~4주 후
SUMMARY → TRACE: 시뮬 3~6개월 후
TRACE → 삭제: 시뮬 1~2년 후 (또는 영원)

예외:
- 재활성화될 때 상위 tier로 복원
- 강한 감정 동반 기억은 VIVID 오래 유지
```

### 5.4 수면 공고화

**Phase 1 (단순):**
- 작업 기억 → 장기 기억 이전
- 중요도 기반 선별
- 주기적 실행

**Phase 3+ (복잡):**
- SWS (느린 파동 수면): 해마 → 피질 이전
- REM (빠른 안구 운동): 재조합, 가지치기
- 4단계 주기 반복

---

## 6. 학습 메커니즘

### 6.1 기본 학습 알고리즘

**Hebbian + STDP (Spike-Timing Dependent Plasticity):**
```
A가 B보다 수 ms 먼저 발화 → A→B 연결 강화 (LTP)
A가 B보다 수 ms 늦게 발화 → A→B 연결 약화 (LTD)
```

### 6.2 예측 오차 학습 (Predictive Coding)

**핵심 원리:**
```
예측 → 실제 관찰 → 오차 계산 → 학습 → 예측 업데이트
```

- 도파민 = 예측 오차 신호 (Schultz 연구)
- 큰 오차 = 큰 학습 기회
- 타고난 편향 위반 = 큰 오차 = 강한 학습

### 6.3 주의 기반 게이팅

**주의 없이 학습 안 됨:**
```python
def attention_weighted_learning(stimuli, attention):
    for stim in stimuli:
        learning_rate = base_rate * attention.weight_for(stim)
        update_synapses(stim, learning_rate)
```

주의 결정 요인:
- 자극 강도 (bottom-up)
- 현재 목표 (top-down)
- 새로움
- 보상 예측

---

## 7. Phase별 마일스톤

### Phase 1: Newborn Eve (1-4 개월)

**목표:** 살아있는 갓난아기 Eve

**완성 조건:**
- [ ] Brian2 환경 세팅 완료
- [ ] 30,000 뉴런 기본 뇌 작동
- [ ] 신생아 호르몬 프로파일 적용
- [ ] 5가지 타고난 편향 회로 구현
- [ ] 기본 주의 시스템
- [ ] 기본 기억 시스템 (3-tier)
- [ ] 시공간 인지 기초 (시간셀, 장소셀 간단 버전)
- [ ] 체크포인트 시스템
- [ ] 기본 반사 작동 (놀람, 응시)

**검증 방법:**
- 호르몬 반응 테스트
- 타고난 편향 위반 시 반응 확인
- 기본 기억 저장/회상 테스트
- 수면 주기 확인

### Phase 2: First World (5-8 개월)

**목표:** 세상 탐색하는 Eve

**완성 조건:**
- [ ] 텍스트 가상공간 구현
- [ ] 기본 물리 (중력, 충돌) 시뮬레이션
- [ ] 객체 상호작용 메커니즘
- [ ] 운동 시스템 (행동 선택)
- [ ] 예측 오차 학습 구현
- [ ] 중력/관성 학습 (Spelke 5-7개월 수준)
- [ ] 대상 영속성 학습
- [ ] 첫 개념 형성 ("공", "벽" 등 5~10개)
- [ ] 탐색 동기 (호기심 기반)

**검증 방법:**
- 물리 법칙 학습 확인
- 개념-경험 연결 테스트
- 호르몬 변화 추적

### Phase 3: Social Bonds (9-12 개월)

**목표:** 민석이와 대화 시작하는 Eve

**완성 조건:**
- [ ] 민석이와 텍스트 상호작용 인터페이스
- [ ] 공동 주의 메커니즘
- [ ] 옥시토신 결합 형성
- [ ] 반복 노출 단어 학습
- [ ] 첫 단어 출력 (시뮬 12개월 목표)
- [ ] 단순 감정 표현
- [ ] 10~30개 단어 어휘
- [ ] 포트폴리오 정리

**검증 방법:**
- 단어 이해도 테스트
- 단순 표현 생성
- 민석이와의 유대 형성 측정

### Phase 4+ (2년차 이후)

**예상 목표 (잠정):**
- 2D 가상공간 확장
- 풍부한 어휘 (100개+)
- 단순 문장 조합 (2-3 단어)
- 복합 감정 표현
- 자기 인식 강화
- 복잡한 수면 시스템

---

## 8. 제약 사항

### 8.1 절대 금지

- ❌ 트랜스포머 아키텍처 사용
- ❌ 사전 학습된 언어 모델 사용
- ❌ 통계적 언어 학습 (대량 코퍼스 기반)
- ❌ 구체적 언어 규칙 하드코딩 (문법 등)
- ❌ 지식 베이스 주입 (위키피디아 등)
- ❌ 중력/관성 같은 학습 가능 직관 사전 프로그래밍

### 8.2 허용

- ✅ 타고난 편향 회로 설계 (진화적 선구조)
- ✅ 호르몬 초기값 설정
- ✅ 발달 단계 스케줄 정의
- ✅ 기본 반사 하드코딩
- ✅ 학습 알고리즘 선택
- ✅ 감각 전처리 (생물학적 수준)

### 8.3 선택적

- ⚠️ 작은 보조 신경망 (SNN 외부, 인지적 기능 제외)
- ⚠️ 규칙 기반 언어 구조 (Phase 3+ 고려)

---

## 9. 위험 관리

### 9.1 기술적 위험

**위험 1: Brian2 성능 한계**
- Colab CPU로 30,000 뉴런 실시간 어려울 수 있음
- 완화: 벡터화 철저, 필요시 뉴런 수 감소

**위험 2: 학습 수렴 실패**
- 타고난 편향만으로 언어 학습 부족 가능
- 완화: Phase 2-3에서 데이터 양 조정

**위험 3: 저장소 병목**
- Colab ↔ Drive I/O 느림
- 완화: 로컬 버퍼링, 배치 처리

### 9.2 일정 위험

**위험 1: Phase 지연**
- 복잡도 과소평가 가능
- 완화: 월별 중간 점검, 범위 재조정

**위험 2: 군 복무 시간 제약**
- 하루 30분~1시간 개발 시간
- 완화: 자동화된 학습 루프, 백그라운드 실행

### 9.3 철학적 위험

**위험 1: 완벽주의**
- 타고난 편향 정확도 추구로 지연
- 완화: "생물학 영감" 수준에 만족

**위험 2: 기대치 관리**
- "대화 가능" 목표 과도하게 확장
- 완화: Phase별 명확한 성공 기준

---

## 10. 의사결정 기록

### 확정된 결정사항

| # | 결정 | 날짜 | 근거 |
|---|---|---|---|
| 1 | 청각 제거 | 2026-04 | 경량화, 청각장애인 언어 습득 증거 |
| 2 | 4-C 언어 출력 (규칙+학습) | 2026-04 | 트랜스포머 금지 원칙 |
| 3 | 텍스트 가상공간 | 2026-04 | 환경 제약, 개발 현실성 |
| 4 | Brian2 프레임워크 | 2026-04 | 생물학적 정확도 + Colab 호환 |
| 5 | 3-tier 기억 시스템 | 2026-04 | 저장 효율 + 생물학적 정확 |
| 6 | Spelke 이론 기반 타고난 지식 | 2026-04 | 과학적 근거 확실 |
| 7 | 중력/관성은 학습으로 | 2026-04 | Spelke 실험 증거 |
| 8 | 1년 목표: Phase 1-3 | 2026-04 | 현실적 범위 |
| 9 | 청각은 Phase 5+에서 추가 | 2026-04 | 신경가소성 활용 |
| 10 | v1 코드 보존, v2 클린 슬레이트 | 2026-04 | 설계 전환 명확히 |

### 보류 중인 결정

- 2TB 저장소 구매 시점 (Phase 3 진입 시 재평가)
- Phase 2 2D 환경 확장 여부
- 특정 뉴런 수 파라미터 (테스트 후 조정)

### 변경 로그

- 2026-04-17: 초안 작성 (v0.1)

---

## 11. 참고 자료

### 필독 문헌

1. Spelke, E. S. (2022). **What Babies Know: Core Knowledge and Composition Volume 1**. Oxford University Press.
2. Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*.
3. Stimberg, M., Brette, R., & Goodman, D. F. (2019). Brian 2, an intuitive and efficient neural simulator. *eLife*.

### 핵심 논문

1. Spelke et al. (1992). Origins of knowledge. *Psychological Review*.
2. Kim & Spelke (1992). Infants' sensitivity to effects of gravity on visible object motion.
3. Saffran et al. (1996). Statistical learning by 8-month-old infants. *Science*.
4. Hamlin, Wynn & Bloom (2007). Social evaluation by preverbal infants. *Nature*.
5. Schultz (1997). A neural substrate of prediction and reward. *Science*. (도파민 이론)
6. Ullman, Spelke, Battaglia, Tenenbaum (2017). Mind games: Game engines as an architecture for intuitive physics.

### 한국어 자료

- 박문호 박사 뇌과학 강의 (YouTube)
- 서유헌 교수 저서 (뇌 발달)

### 기술 자료

- Brian2 문서: https://brian2.readthedocs.io/
- snnTorch 문서 (참고용)
- MB-CDI 어휘 발달 데이터: https://mb-cdi.stanford.edu/

---

## 12. 용어집

- **SNN**: Spiking Neural Network, 스파이킹 신경망
- **STDP**: Spike-Timing Dependent Plasticity, 스파이크 타이밍 의존 가소성
- **Core Knowledge**: Spelke의 타고난 핵심 지식 체계
- **Episodic Memory**: 일화 기억 (특정 경험)
- **Semantic Memory**: 의미 기억 (일반 지식)
- **Consolidation**: 공고화 (단기 → 장기 기억 이전)
- **Place Cell**: 장소 셀 (해마의 공간 표상 뉴런)
- **Grid Cell**: 격자 셀 (내후각 피질의 공간 표상)
- **Time Cell**: 시간 셀 (해마의 시간 표상)
- **Predictive Coding**: 예측 부호화 (뇌의 기본 작동 원리)
- **Valence**: 정서가 (좋음/나쁨 차원)
- **Arousal**: 각성 (활성화 정도)

---

## 13. 문서 관리

- **소유자:** MS0502
- **업데이트 주기:** Phase 완료 시, 중대 결정 시
- **변경 절차:** 결정 사항을 "의사결정 기록" 섹션에 먼저 기록한 후 본문 수정
- **버전 관리:** Git으로 추적

**이 문서를 읽는 다른 AI 또는 미래의 나에게:**

프로젝트 진행 중 혼란이 생기면:
1. 이 문서부터 확인
2. 관련 결정사항 있는지 찾기
3. 없으면 논의 후 이 문서에 추가
4. 구현은 문서 업데이트 후에

이 원칙을 지키지 않으면 6개월 후 정체성을 잃은 Eve가 된다.

---

*끝.*
