# EVE STATUS v32.1 - Single Source of Truth

> **다음 세션 Claude: 이 파일을 첫 단계로 읽으세요.**
> 컴팩트로 컨텍스트 잃어도 이 파일이 진실.

**날짜**: 2026-04-30
**사용자**: 김민석 (군 복무 중, Galaxy Z Fold 6, Colab Pro, Chrome)
**현재 상태**: v32 티어 A+B1 통합 완료 + v32.1/v2.1 lock fix 완료
**의사소통**: 반말, 직설적, 군더더기 X

---

## 🚨 다음 세션 첫 단계 (READ ME FIRST)

1. 이 파일 읽고 현재 상태 파악
2. Colab Pro Drive 환경. 사용자가 파일 업로드해주면 작업, 결과는 `/mnt/user-data/outputs/`로
3. **드라이브 직접 다운/업 금지** — 사용자가 직접 Drive에 올림 (Drive API가 작업 끊는 버그 있음)
4. **드라이브 search는 OK** (현황 파악용)
5. 한 모듈씩 작업, 검증 통과 후 다음
6. 사용자가 PENDING 중 골라달라고 함

---

## 🎯🎯🎯 EVE 절대 원칙 (NEVER VIOLATE)

### 1. 카테고리 = 의미 단위 (뉴런 X) ★★★
- 인간 뇌의 뉴런/시냅스를 **기능적으로 대체**
- 뉴런 → 카테고리 / 시냅스 → 가중 연결 / 발화 → 활성 타이밍
- Brian2/NEST 의존성 X — 카테고리 레벨 결정론
- 학계: Anderson 1983 Spreading Activation, Quillian 1968

### 2. 확률 X = 결정론적 의미 추론 ★★★
- ❌ 트랜스포머/N-gram/Softmax/temperature/Monte Carlo
- ❌ Probability distributions, random sampling
- ❌ `np.random.*`, `random.random()`, `random.choice()`, `.sample()`
- ✅ 의미 추론, 카테고리 활성, 가중치+임계
- ✅ Variation = 호르몬 자연 변조 (생물학적, np.sin 기반 deterministic)

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

## ✅ 모듈 현황 (2026-04-30 기준)

### 티어 A (살아있음 토대)

| 모듈 | 버전 | 검증 | 핵심 기능 |
|---|---|---|---|
| **A1: HormoneSystem** | **v32.1** | 27 회귀 + 13 lock = 40/40 | 26 호르몬, 13 칵테일, 13 이벤트, 양방향 폐쇄 루프, **자연 변동성 + saturation 풀림** |
| **A2: SpreadingActivation** | v32 | 24+25/49 | 카테고리 활성 + 자기 진화 (Ebbinghaus+Bjork) |
| A3: CategoryGraph | 유보 | - | SA 내부로 충분 |
| **A4: WorkingMemory + GNW** | **v2.1** | 20 회귀 + 11 lock = 31/31 | 30+ 슬롯, broadcast, **decay 강화 + access_count float 감쇠** |
| **A5: DMN** | **v2.1** | 18+14/32 | 4 모드 (+self_intent), **NEED_TO_CATEGORIES 풀 확장 + 의미 필터** |
| **A6: DigitalSomatic** | **v2** | 27/27 | 8차원 신체, gut_signal, **activity_level BASELINE/회복** |

### 티어 B1 (자연어 대화)

| 모듈 | 버전 | 검증 | 핵심 |
|---|---|---|---|
| **B1: NaturalLanguage** | v32 | 34/34 | say/learn_text/learn_beliefs, 거절 창발 |

### 통합

| 통합본 | 검증 | 상태 |
|---|---|---|
| eve_main_a.py | 22/22 | A 6모듈 통합 |
| **eve_main_ab.py** | **28/28** | A + B1 통합 (4977 신념 0.4초 로드, **거절 분기 cortisol→'음...' / oxytocin→'응 알았어'**) |

### v32.1 + v2.1 핵심 변경 (이번 세션 lock fix)

**A1 → A1.1 (hormone_system.py)**
- Saturation-aware decay: level>0.7에서 회귀 가속 (1.0 lock 차단)
- 결정론적 자연 변동성: `np.sin` 기반, 호르몬마다 osc_period/phase/amplitude
- Saturation-aware stimulate: 수용체 다운레귤레이션 (0.7→1.0에서 반응 1.0→0.1)
- 검증: oxytocin 영구 0.965 lock → mean 0.72 std 0.07로 정상

**A4 → A4.1 (working_memory.py)**
- WMSlot.access_count: int → float (시간 감쇠 정확)
- add() saturation-aware: salience>0.5에서 둔감화 (1.0 도달 차단)
- decay_factor floor 0.5 (영구 슬롯 lock 방지)
- access_count 시간 감쇠 (반감기 30s, Brown 1958)
- salience floor 0.05 → 0.10 (조기 evict)
- 검증: 600초 자극 끊고 14개 영구 lock → 150초에 0개로 정리

**통합 검증 (NL 포함)**:
- say('민석') → '안녕, 민석' / feel='편안함' ✅
- 거절 분기: cortisol↑ → '음... 해줘'/'보통' vs oxytocin↑ → '응 알았어, 해줘'/'편안함' ✅
- 호르몬 변동성이 응답 임계 안 깸 (일관성 유지) ✅

---

## 📁 파일 경로 (Colab Drive 기준)

```
/content/drive/MyDrive/
├── eve_v32/                                  ← 메인 작업 폴더
│   ├── EVE_STATUS_v32_1.md                  ← 이 파일
│   ├── eve_main_ab.py                        ← AB 통합 (NL 포함)
│   ├── eve_main_a.py                         ← A만 통합
│   ├── test_eve_main_ab.py                   ← 통합 검증
│   └── eve_modules/
│       ├── hormone_system.py                 ← v32.1 (40/40)
│       ├── working_memory.py                 ← v2.1 (31/31)
│       ├── spreading_activation.py           ← v32
│       ├── dmn.py                            ← v2.1
│       ├── digital_somatic.py                ← v2
│       ├── natural_lang.py                   ← v32
│       └── test_*.py                         ← 단독 검증
└── eve_data/
    └── beliefs.json                          ← 4977 신념
```

**알려진 path 이슈**:
- `natural_lang.load_beliefs()` default path가 `/home/claude/eve/beliefs.json` (Claude 환경 잔재)
- 명시 호출 필요: `eve.learn_beliefs(path='/content/drive/MyDrive/eve_data/beliefs.json')`

---

## 📝 PENDING 작업 (사용자가 골라달라고 함)

| # | 모듈 | 우선순위 | 비고 |
|---|---|---|---|
| 11 | natural_lang.respond echo | 중 | say('민석') → '민석' echo. respond 로직 점검 |
| 12 | sentiment 빈약 | 중 | NL의 sentiment 분석 강화 |
| 13 | 호르몬 폐쇄 루프 | **사실상 완료?** | 통합 시뮬에서 정상 작동 확인됨. 추가 작업 X일 수도 |
| 4 | EpisodicMemory B2 | 큰 작업 | Hybrid (tuple + 카테고리 set + 자연어), 새 모듈 |
| 6 | gut_signal B5 | 중 | DigitalSomatic 확장 |
| 8 | 자기 명령 정식 B5 | 중 | self_intent → 자기 명령 처리 정식화 |
| 1 | CategoryGraph A3 | 낮음 | STATUS엔 "유보, SA로 충분"인데 PENDING에 다시 올라옴. 사용자 확인 필요 |
| - | beliefs path 정리 | 작은 작업 | natural_lang 기본 path를 Drive 경로로 |

**티어 C (추론) 다음 단계**: CausalGraph(Pearl), WorldModel, GoalManagement, EmotionRegulation, Counterfactual, Analogy
**티어 D (인간다움)**: Suffering(공감), Creative, Humor, Temporal, multi_stream, tool_use
**티어 E (학습)**: AI2-THOR + RL (메인 안정 후)

---

## 🛠️ 작업 패턴 (이번 세션에서 확립)

1. **사용자가 Colab Drive에 모듈 둠** → 필요한 파일을 채팅에 업로드해서 줌
2. **Claude가 컨테이너에서 작업**:
   - `/mnt/user-data/uploads/`에 받은 파일 → `/home/claude/eve_v32/`로 복사
   - 진단 → 수정 → 검증 (3 라운드 정도 반복)
   - 기존 테스트 회귀 + 신규 lock 풀림 동시 통과 확인
3. **결과를 `/mnt/user-data/outputs/`로 빼서 사용자에게 전달**
4. **사용자가 직접 Drive에 올림**
5. **Colab에서 검증 재실행** (확인용)

**검증 실행 패턴 (Colab Pro)**:
```python
import subprocess, os
os.chdir('/content/drive/MyDrive/eve_v32/eve_modules')
r = subprocess.run(['python', '-u', 'test_*.py'],
                   capture_output=True, text=True, timeout=300)
print(r.stdout[-2000:])
```

**Colab 출력 짤림 회피**: subprocess + `python -u` (unbuffered) + `capture_output=True`

---

## ⚠️ 주의사항

1. **드라이브 다운/업 X** (Drive API가 작업 세션 끊음) — search/path 확인만 OK
2. **NL 본체(natural_lang.py)는 Claude 컨테이너 환경에 없음** — 사용자가 업로드해야 NL 통합 시뮬 가능
3. **컨테이너 bash 가능, 네트워크 X** — pip install 등 외부 의존성 다운로드 X
4. **Colab UI 출력 짤림 주의** — subprocess 패턴 권장
5. **결정론 검증 필수** — 모든 fix 후 같은 시퀀스 → 같은 결과 (`max diff = 0`) 확인
6. **f-string 안의 conditional 조심** — `f"{x:.3f if x else 'evict'}"` 같은 거 invalid format
7. **외부 import 변경 시** → `del sys.modules['module']` 후 재import 필요

---

## 🧬 호르몬 26종

**Tier A (10) 신경전달물질**: glutamate, gaba, glycine, dopamine, serotonin, norepinephrine, histamine, acetylcholine, adenosine, endorphin
**Tier B (12) 뇌 호르몬**: cortisol, oxytocin, vasopressin, melatonin, bdnf, ngf, estrogen, testosterone, insulin_brain, thyroid, leptin, ghrelin
**Tier C (4) 특수**: prolactin, dhea, progesterone, growth_hormone

**카테고리 그룹 (8)**: reward, social, threat, attention, memory, rest, curiosity, aversion

---

## 🔄 EVE 메인 루프 (1 tick)

```
1. HormoneSystem.update(dt)              - 호르몬 자연 변화 + 일주기 + 자연 변동성(v32.1)
2. 호르몬 → SA 임계 변조 (Top-down)
3. SA.spread(steps=1) - 활성 전파
4. SA.decay(dt)
5. SA → 카테고리 → 호르몬 자극 (Bottom-up)
6. WM.update_from_activation(SA)
7. WM.apply_hormone_state(HS)
8. WM.broadcast() - GNW listener에 focus 전파
9. WM.decay(dt) - access_count 감쇠 + saturation aware (v2.1)
10. DMN.tick(dt) - 자발 활성
11. DigitalSomatic.update(dt) - 신체 감각
12. (100 tick마다) SA.forget(dt, hormone_modifier) - 망각 (Ebbinghaus)
```

---

## 📜 진행 로그

### 2026-04-29
- v32 티어 A 6모듈 + B1 NaturalLanguage 통합 완료 (239/239 통과)

### 2026-04-30 (이번 세션)
- **A1 → A1.1**: hormone v32.1 lock 풀기 (40/40)
- **A4 → A4.1**: WM v2.1 lock 풀기 (31/31)
- A 통합 시뮬 (NL X) 정상 확인
- **AB 통합 검증**: cortisol vs oxytocin 거절 분기 작동 확인
- **EVE_STATUS_v32_1.md 작성** (이 파일)

### 다음 세션
- PENDING 중 사용자 선택
- **추천**: [11] respond echo + [12] sentiment 묶음 (NL 작업, 가벼움)
