"""
B4 ActiveInference 검증
=========================
Mock HS/SA/WM/SD/DS — 핵심 로직만.

35+ 검증 케이스:
- 인스턴스화
- predict (학습 전/후)
- observe (error 계산)
- 학습 규칙 (Hebbian + winner-take-soft)
- weight clamp [-1, 1]
- update_from_outcome (SD 짝꿍, inline, deque pairing)
- get_hormone_signature
- 결정론 (random 미사용, 같은 입력 → 같은 결과)
- lock 방지 (deque maxlen, weight bounded)
- 다단계 학습 정확도 향상
"""
import sys
sys.path.insert(0, '/home/claude/eve_modules')

import numpy as np
from active_inference import ActiveInference


# ============= Mocks =============

class MockHormone:
    def __init__(self, level=0.5):
        self.level = float(level)


class MockHS:
    def __init__(self, **levels):
        defaults = {
            'cortisol': 0.5, 'norepinephrine': 0.5, 'oxytocin': 0.5,
            'serotonin': 0.5, 'dopamine': 0.5, 'bdnf': 0.3,
            'endorphin': 0.2, 'adenosine': 0.3,
        }
        defaults.update(levels)
        self.hormones = {n: MockHormone(v) for n, v in defaults.items()}
        self.active_hormones = set(defaults.keys())
    
    def set(self, name, level):
        self.hormones[name].level = float(level)
    
    def compute_mood(self):
        cort = self.hormones['cortisol'].level
        ot = self.hormones['oxytocin'].level
        da = self.hormones['dopamine'].level
        ne = self.hormones['norepinephrine'].level
        valence = (ot + da - cort) / 1.5 - 0.3
        arousal = (ne + cort) / 2.0
        return {'valence': float(np.clip(valence, -1, 1)),
                'arousal': float(np.clip(arousal, 0, 1)),
                'dominance': 0.5}
    
    def primary_hormone(self):
        m = max(self.hormones.items(), key=lambda x: (x[1].level, x[0]))
        return (m[0], m[1].level)


class MockSA:
    def __init__(self):
        self.activations = {}
    def activate(self, c, s):
        self.activations[c] = max(self.activations.get(c, 0), s)


class MockWM:
    def __init__(self):
        self.slots = {}
        self._focus = None
    def get_focus(self):
        return self._focus


def passed(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  | {detail}" if detail else ""))
    return ok


print("=" * 70)
print("B4 ActiveInference (Friston FEP) 검증")
print("=" * 70)
results = []


# ============= [1] 인스턴스화 =============
print("\n[1] 인스턴스화")
hs = MockHS()
sa = MockSA()
wm = MockWM()
ai = ActiveInference(hs, sa, working_memory=wm)
results.append(passed("AI 생성", ai is not None))
results.append(passed("초기 가중치 모두 0",
                     all(w == 0.0
                         for h in ai.hormone_outcome_weights.values()
                         for w in h.values())))
results.append(passed("OUTCOME_LABELS 3개",
                     len(ai.OUTCOME_LABELS) == 3
                     and 'good' in ai.OUTCOME_LABELS
                     and 'bad' in ai.OUTCOME_LABELS
                     and 'neutral' in ai.OUTCOME_LABELS))
results.append(passed("8 호르몬 추적",
                     len(ai.HORMONE_NAMES_TRACKED) == 8))
results.append(passed("predictions deque 비어있음",
                     len(ai.predictions) == 0))


# ============= [2] predict (학습 전) =============
print("\n[2] predict (학습 전 — confidence 0)")
pred = ai.predict(horizon=1.0)
results.append(passed("prediction id 생성",
                     pred['id'] == 'pred_0000'))
results.append(passed("학습 0이면 confidence 0",
                     pred['confidence'] == 0.0,
                     f"conf={pred['confidence']}"))
results.append(passed("학습 0이면 outcome=neutral",
                     pred['expected_outcome'] == 'neutral'))
results.append(passed("expected_state 캡처",
                     'hormones' in pred['expected_state']
                     and len(pred['expected_state']['hormones']) >= 7))
results.append(passed("predictions deque 추가",
                     len(ai.predictions) == 1))


# ============= [3] observe (학습 없이) =============
print("\n[3] observe (실제 outcome 없이) — error만")
err = ai.observe()  # 가장 최근 미관측
results.append(passed("error_record 반환", 'mood_error' in err))
results.append(passed("hormone_errors 8 호르몬",
                     len(err['hormone_errors']) >= 7))
results.append(passed("outcome_match=None (actual 없음)",
                     err['outcome_match'] is None))
results.append(passed("prediction이 observed=True 표시",
                     ai.predictions[0]['observed'] == True))


# ============= [4] 학습 — bad outcome with high cortisol =============
print("\n[4] ★ 학습: 고-cortisol 상황 → 'bad' outcome")
ai = ActiveInference(hs, sa)  # reset
hs.set('cortisol', 0.9)  # 강한 cortisol
hs.set('norepinephrine', 0.8)
hs.set('oxytocin', 0.2)
pred = ai.predict()
err = ai.observe(actual_outcome='bad')

cort_bad_w = ai.hormone_outcome_weights['cortisol']['bad']
cort_good_w = ai.hormone_outcome_weights['cortisol']['good']
ot_bad_w = ai.hormone_outcome_weights['oxytocin']['bad']

print(f"     cortisol bad weight = {cort_bad_w:+.4f}")
print(f"     cortisol good weight = {cort_good_w:+.4f}")
print(f"     oxytocin bad weight = {ot_bad_w:+.4f}  (deviation 음수라 음수)")

results.append(passed("cortisol-bad 가중치 양수 (강화)",
                     cort_bad_w > 0,
                     f"{cort_bad_w:+.4f}"))
results.append(passed("cortisol-good 가중치 음수 (약화, winner-take-soft)",
                     cort_good_w < 0,
                     f"{cort_good_w:+.4f}"))
results.append(passed("oxytocin-bad 가중치 음수 (낮은 ot이 bad 신호 X)",
                     ot_bad_w < 0,
                     f"{ot_bad_w:+.4f}"))
results.append(passed("update_count 증가",
                     ai.update_count == 1))


# ============= [5] 반복 학습 → confidence 향상 =============
print("\n[5] 반복 학습: 같은 호르몬-outcome 패턴 5회")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 0.9)
hs.set('oxytocin', 0.2)
hs.set('norepinephrine', 0.8)

for i in range(5):
    p = ai.predict()
    ai.observe(actual_outcome='bad')

# 이제 같은 호르몬 상태에서 predict → 'bad' 예측 + confidence 높음
final_pred = ai.predict()
print(f"     5회 학습 후: outcome={final_pred['expected_outcome']}, "
      f"conf={final_pred['confidence']:.3f}")
print(f"     scores={final_pred['outcome_scores']}")
results.append(passed("학습 후 outcome='bad' 예측",
                     final_pred['expected_outcome'] == 'bad',
                     f"got {final_pred['expected_outcome']}"))
results.append(passed("학습 후 confidence > 0.4",
                     final_pred['confidence'] > 0.4,
                     f"conf={final_pred['confidence']:.3f}"))


# ============= [6] 정반대 시나리오 — 'good' =============
print("\n[6] ★ 정반대: 고-oxytocin/serotonin → 'good'")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 0.2)
hs.set('oxytocin', 0.9)
hs.set('serotonin', 0.85)
hs.set('dopamine', 0.7)
hs.set('norepinephrine', 0.3)

for _ in range(5):
    ai.predict()
    ai.observe(actual_outcome='good')

# 같은 상태로 prediction
p = ai.predict()
print(f"     예측: {p['expected_outcome']}, conf={p['confidence']:.3f}")
results.append(passed("학습 후 outcome='good' 예측",
                     p['expected_outcome'] == 'good'))

ot_good = ai.hormone_outcome_weights['oxytocin']['good']
cort_good = ai.hormone_outcome_weights['cortisol']['good']
results.append(passed("oxytocin-good 강화 (양수)",
                     ot_good > 0, f"{ot_good:+.4f}"))
results.append(passed("cortisol-good 약화 (음수, dev<0이라)",
                     cort_good < 0, f"{cort_good:+.4f}"))


# ============= [7] weight clamp [-1, 1] =============
print("\n[7] weight clamp — 100회 반복해도 weight ≤ 1")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 1.0)  # 최대
for _ in range(100):
    ai.predict()
    ai.observe(actual_outcome='bad')

max_w = max(ai.hormone_outcome_weights['cortisol'].values())
min_w = min(ai.hormone_outcome_weights['cortisol'].values())
print(f"     cortisol weights after 100x: max={max_w:+.3f}, min={min_w:+.3f}")
results.append(passed("max weight ≤ 1.0", max_w <= 1.0))
results.append(passed("min weight ≥ -1.0", min_w >= -1.0))
results.append(passed("실제로 clamp에 도달 (강한 학습)",
                     max_w > 0.9 or min_w < -0.9,
                     f"max={max_w:+.3f}"))


# ============= [8] update_from_outcome (SD 짝꿍) — pairing =============
print("\n[8] update_from_outcome — predict 있을 때 pair")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 0.8)
hs.set('oxytocin', 0.3)

p = ai.predict()
result = ai.update_from_outcome(predicted_doubt=0.7, actual_outcome='bad')

results.append(passed("source='sd_outcome' (pair 성공)",
                     result.get('source') == 'sd_outcome'))
results.append(passed("predicted_doubt 기록",
                     result.get('predicted_doubt') == 0.7))
results.append(passed("prediction이 observed 표시",
                     p['observed'] == True))
results.append(passed("학습 발생 (가중치 변동)",
                     ai.hormone_outcome_weights['cortisol']['bad'] > 0))


# ============= [9] update_from_outcome inline (predict 없이) =============
print("\n[9] update_from_outcome — predict 없을 때 inline")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 0.8)
result = ai.update_from_outcome(predicted_doubt=0.5, actual_outcome='bad')

results.append(passed("source='sd_outcome_inline'",
                     result.get('source') == 'sd_outcome_inline'))
results.append(passed("inline_learn=True",
                     result.get('inline_learn') == True))
results.append(passed("inline 학습도 가중치 변동",
                     ai.hormone_outcome_weights['cortisol']['bad'] > 0))


# ============= [10] get_hormone_signature =============
print("\n[10] get_hormone_signature 조회")
ai = ActiveInference(hs, sa)
hs.set('cortisol', 0.9)
hs.set('oxytocin', 0.2)
for _ in range(3):
    ai.predict()
    ai.observe(actual_outcome='bad')

sig_bad = ai.get_hormone_signature('bad')
sig_good = ai.get_hormone_signature('good')

results.append(passed("get_hormone_signature 반환 dict",
                     isinstance(sig_bad, dict)))
results.append(passed("8 호르몬 모두 포함",
                     len(sig_bad) == 8))
results.append(passed("cortisol → bad 양수 (학습됨)",
                     sig_bad['cortisol'] > 0,
                     f"{sig_bad['cortisol']:+.3f}"))
results.append(passed("get_dominant_hormones — 정렬",
                     len(ai.get_dominant_hormones('bad', top_n=3)) == 3))


# ============= [11] 결정론 =============
print("\n[11] 결정론")
import inspect
src = inspect.getsource(ActiveInference)
results.append(passed("코드: random 미사용",
                     'np.random' not in src
                     and 'random.choice' not in src
                     and '.sample(' not in src
                     and 'softmax' not in src.lower()))

# 같은 입력 → 같은 출력
hs1 = MockHS(cortisol=0.7, oxytocin=0.4)
hs2 = MockHS(cortisol=0.7, oxytocin=0.4)
sa1 = MockSA()
sa2 = MockSA()
ai1 = ActiveInference(hs1, sa1)
ai2 = ActiveInference(hs2, sa2)

for _ in range(3):
    ai1.predict()
    ai1.observe(actual_outcome='bad')
    ai2.predict()
    ai2.observe(actual_outcome='bad')

w1 = ai1.hormone_outcome_weights
w2 = ai2.hormone_outcome_weights
results.append(passed("동일 입력 시퀀스 → 동일 weight",
                     w1 == w2))


# ============= [12] deque maxlen lock 방지 =============
print("\n[12] deque maxlen — 영구 누적 X")
ai = ActiveInference(hs, sa, prediction_window=5, error_window=10)
for _ in range(20):
    ai.predict()
    ai.observe(actual_outcome='neutral')

results.append(passed("predictions deque ≤ 5",
                     len(ai.predictions) == 5))
results.append(passed("errors deque ≤ 10",
                     len(ai.errors) == 10))
results.append(passed("predict_count 누적은 정확 (overflow 영향 X)",
                     ai.predict_count == 20))


# ============= [13] tick 시간 누적 =============
print("\n[13] tick(dt)")
ai = ActiveInference(hs, sa)
ai.tick(dt=1.5)
ai.tick(dt=2.0)
results.append(passed("time 누적", abs(ai.time - 3.5) < 1e-6))
ai.tick(dt=0.0)
results.append(passed("dt=0이어도 안전", abs(ai.time - 3.5) < 1e-6))
ai.tick(dt=-1.0)  # 음수 무시
results.append(passed("음수 dt 무시", abs(ai.time - 3.5) < 1e-6))


# ============= [14] 시나리오: 다단계 학습 후 정확도 =============
print("\n[14] ★ 다단계: 호르몬 상태 별 outcome 학습 → 정확도")
ai = ActiveInference(MockHS(), MockSA())  # 새 mock

# 학습: 고cort = bad, 고ot = good 패턴 각 5회
for _ in range(5):
    ai.hs.set('cortisol', 0.9); ai.hs.set('oxytocin', 0.2)
    ai.predict(); ai.observe(actual_outcome='bad')
    ai.hs.set('cortisol', 0.2); ai.hs.set('oxytocin', 0.9)
    ai.predict(); ai.observe(actual_outcome='good')

# 테스트: 새 호르몬 상태 → 예측
ai.hs.set('cortisol', 0.85); ai.hs.set('oxytocin', 0.25)
p_bad = ai.predict()
results.append(passed("고cort/저ot → 'bad' 예측 (맞춤)",
                     p_bad['expected_outcome'] == 'bad',
                     f"got {p_bad['expected_outcome']}"))

ai.hs.set('cortisol', 0.25); ai.hs.set('oxytocin', 0.85)
p_good = ai.predict()
results.append(passed("저cort/고ot → 'good' 예측 (맞춤)",
                     p_good['expected_outcome'] == 'good'))

# 가짜 outcome 입력 → accuracy 계산
ai.observe(actual_outcome='good')  # p_good 매칭
state = ai.get_state()
print(f"     final state: {state}")
results.append(passed("get_state 정상 반환",
                     'outcome_accuracy' in state
                     and 'weight_summary' in state))


# ============= [15] outcome 라벨 검증 =============
print("\n[15] 잘못된 outcome 라벨 안전 처리")
ai = ActiveInference(MockHS(), MockSA())
ai.predict()
result = ai.update_from_outcome(0.5, 'invalid_label')
results.append(passed("잘못된 라벨 → 'neutral' fallback (예외 X)",
                     'error' not in result or result.get('actual_outcome') == 'neutral'))


# ============= 요약 =============
print("\n" + "=" * 70)
total = len(results)
passed_count = sum(results)
print(f"검증: {passed_count} / {total} 통과")
print("=" * 70)

if passed_count == total:
    print("\n🎉 B4 ActiveInference 모든 검증 통과!")
    print()
    print("핵심 기능:")
    print("  ✅ predict() — 호르몬 → outcome 예측 (학습 전 neutral)")
    print("  ✅ observe() — error 계산 + (outcome 있으면) 학습")
    print("  ✅ update_from_outcome() — SD 짝꿍 (pair / inline 양쪽)")
    print("  ✅ get_hormone_signature() — 학습된 매핑 조회")
    print("  ✅ get_dominant_hormones() — outcome 별 top 호르몬")
    print()
    print("학습 검증:")
    print("  ✅ Hebbian: deviation × LR로 강화")
    print("  ✅ winner-take-soft: 다른 라벨 0.3× 약화")
    print("  ✅ 5회 학습 → confidence > 0.4")
    print("  ✅ 다단계 패턴: cort↑→bad, ot↑→good 정확히 학습")
    print()
    print("EVE 절대 원칙:")
    print("  ✅ 결정론 (random/sample/softmax 미사용)")
    print("  ✅ 호르몬 변조 (deviation 기반)")
    print("  ✅ 동일 입력 → 동일 출력")
    print()
    print("Lock 방지:")
    print("  ✅ weight [-1, 1] clamp")
    print("  ✅ predictions/errors deque maxlen")
    print("  ✅ 100회 반복 후도 안정")
    print()
    print("→ SD.HORMONE_DOUBT_WEIGHTS 고정 상수를 동적으로 보강 가능")
    print("→ B4 신규 모듈 완성. eve_main_abc 통합은 다음 세션 작업.")
else:
    print(f"\n⚠️  {total - passed_count}개 실패")
    sys.exit(1)
