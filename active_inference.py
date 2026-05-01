"""
EVE v32 - B4: ActiveInference (Friston 2010 FEP)
==================================================

EVE의 능동적 추론 / 자유 에너지 원리.

학계:
- Friston 2010 - Free Energy Principle (FEP)
- Friston 2017 - Active Inference: process theory
- Clark 2013 - Predictive processing
- Helmholtz 1867 - Perception as inference

핵심 루프:
    predict(t) → act/wait → observe(t+dt) → prediction_error → 학습

EVE 특화 (호르몬 부호 매핑):
    호르몬 상태 H_t → outcome label (good/bad/neutral) 예측
    실제 outcome 받으면 H × outcome 매핑 학습 (Hebbian)
    
    SD.HORMONE_DOUBT_WEIGHTS는 고정 상수.
    AI는 그것을 동적으로 학습해서 보강 hint 제공.

EVE 절대 원칙:
- 확률 X (np.random/sample/softmax X) ✅
- 결정론 (Hebbian update only) ✅
- 호르몬 변조 (호르몬 deviation × outcome reinforcement) ✅
- 카테고리 단위 (outcome → 카테고리 broadcast 가능) ✅
- 모듈 chain (HS, SA, WM, SD, DS 통합) ✅

핵심 기능 6개:
1. predict(horizon)          - 다음 horizon 초 후 예측
2. observe(pred_id, actual)  - 예측 vs 실제 → error + 학습
3. update_from_outcome()     - SD와 짝꿍 (외부 트리거 학습)
4. get_hormone_signature()   - 학습된 호르몬-outcome 매핑 조회
5. tick(dt)                  - 시간 누적 + 정리
6. get_state()               - 진단

Lock 방지:
- weight [-1, 1] clamp
- predictions deque(maxlen=50) — 영구 누적 X
- errors deque(maxlen=100)

Author: 김민석 + Claude v32 (B4)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque


class ActiveInference:
    """
    EVE v32 Active Inference / FEP.
    
    내부 상태:
        hormone_outcome_weights: {hormone: {label: weight}} 학습 매핑
        predictions: 최근 N개 prediction (deque)
        errors: 최근 N개 prediction error (deque)
    
    연결 (모두 옵션 또는 기본):
        HormoneSystem (필수) - 현재 호르몬 상태
        SpreadingActivation (필수) - outcome 카테고리 broadcast
        WorkingMemory (옵션) - state 캡처에 focus 추가
        SelfDoubt (옵션) - update_from_outcome 동기화
        DigitalSomatic (옵션) - state 캡처에 feeling 추가
    """
    
    # outcome 라벨 (단순화: 3분류)
    OUTCOME_LABELS = ('good', 'bad', 'neutral')
    
    # 추적할 호르몬 (HS에 있는 것만)
    HORMONE_NAMES_TRACKED = (
        'cortisol', 'norepinephrine', 'oxytocin', 'serotonin',
        'dopamine', 'bdnf', 'endorphin', 'adenosine',
    )
    
    # 학습률 (느리게 → 안정성, lock 방지)
    LEARNING_RATE = 0.05
    
    # winner-take-soft 비율: 다른 라벨 약화 강도
    NEGATIVE_REINFORCE_RATIO = 0.3
    
    # 가중치 lock 방지
    WEIGHT_MIN = -1.0
    WEIGHT_MAX = +1.0
    
    # 신뢰도 임계값 (이 이하면 'neutral'로 fallback)
    CONFIDENCE_FLOOR = 0.05
    
    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 self_doubt=None,
                 digital_somatic=None,
                 prediction_window: int = 50,
                 error_window: int = 100):
        """
        Args:
            hormone_system: A1 (필수)
            spreading_activation: A2 (필수)
            working_memory: A4 (옵션 - state 캡처)
            self_doubt: B3 (옵션 - update 동기화)
            digital_somatic: A6 (옵션 - state 캡처)
            prediction_window: 보존 prediction 수
            error_window: 보존 error 수
        """
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.sd = self_doubt
        self.ds = digital_somatic
        
        # 학습된 호르몬-outcome 매핑 (초기 0)
        # 결정론적 dict — 키 정렬 순으로 iteration
        self.hormone_outcome_weights: Dict[str, Dict[str, float]] = {
            h: {label: 0.0 for label in self.OUTCOME_LABELS}
            for h in self.HORMONE_NAMES_TRACKED
        }
        
        # 최근 prediction/error
        self.predictions: deque = deque(maxlen=prediction_window)
        self.errors: deque = deque(maxlen=error_window)
        
        # 시간/통계
        self.time = 0.0
        self.predict_count = 0
        self.observe_count = 0
        self.update_count = 0
    
    # ============= 0. 내부: 상태 캡처 =============
    
    def _capture_state(self) -> Dict[str, Any]:
        """현재 EVE 상태 스냅샷 (예측 비교 기준)."""
        mood = self.hs.compute_mood()
        primary = self.hs.primary_hormone()
        
        # 호르몬 레벨 (추적 대상만)
        hormones = {}
        for h in self.HORMONE_NAMES_TRACKED:
            if h in self.hs.hormones:
                hormones[h] = float(self.hs.hormones[h].level)
        
        state = {
            'time': self.time,
            'mood_valence': float(mood['valence']),
            'mood_arousal': float(mood['arousal']),
            'primary_hormone': primary[0],
            'primary_level': float(primary[1]),
            'hormones': hormones,
        }
        
        # 옵션: WM focus
        if self.wm is not None and hasattr(self.wm, 'get_focus'):
            state['focus'] = self.wm.get_focus()
        
        # 옵션: DS feeling
        if self.ds is not None and hasattr(self.ds, 'somatic_state'):
            ss = self.ds.somatic_state
            state['fatigue'] = float(ss.get('fatigue', 0))
            state['tension'] = float(ss.get('tension', 0))
        
        return state
    
    def _score_outcomes(self, hormones: Dict[str, float]) -> Dict[str, float]:
        """
        호르몬 상태 → outcome 점수 (학습된 가중치 기반).
        
        deviation × weight 합산 (per outcome label).
        """
        scores = {label: 0.0 for label in self.OUTCOME_LABELS}
        for h, level in hormones.items():
            if h not in self.hormone_outcome_weights:
                continue
            deviation = level - 0.5  # baseline = 0.5
            for label, weight in self.hormone_outcome_weights[h].items():
                scores[label] += deviation * weight
        return scores
    
    # ============= 1. predict =============
    
    def predict(self, horizon: float = 1.0) -> Dict[str, Any]:
        """
        현재 상태 → horizon 초 후 예측.
        
        - 호르몬 상태는 단순 외삽 (현재값 유지)
        - outcome 라벨은 학습된 매핑으로 예측
        - confidence 매우 낮으면 'neutral' fallback
        
        Args:
            horizon: 미래 시간 (초)
        
        Returns: prediction dict (id 포함, predictions deque에 저장됨)
        """
        current = self._capture_state()
        
        # outcome 점수 계산
        outcome_scores = self._score_outcomes(current['hormones'])
        
        # 학습 거의 없으면 neutral fallback
        max_abs = max(abs(v) for v in outcome_scores.values())
        if max_abs < self.CONFIDENCE_FLOOR:
            predicted_outcome = 'neutral'
            confidence = 0.0
        else:
            # 결정론 정렬: score 내림차순, label 오름차순 (tie-break)
            sorted_outcomes = sorted(
                outcome_scores.items(),
                key=lambda x: (-x[1], x[0])
            )
            predicted_outcome = sorted_outcomes[0][0]
            top_score = sorted_outcomes[0][1]
            # confidence: top score / sum of abs (정규화)
            total_abs = sum(abs(v) for v in outcome_scores.values())
            if total_abs > 0:
                confidence = float(np.clip(abs(top_score) / total_abs, 0.0, 1.0))
            else:
                confidence = 0.0
        
        prediction = {
            'id': f'pred_{self.predict_count:04d}',
            'time': self.time,
            'horizon': horizon,
            'target_time': self.time + horizon,
            'expected_state': current,
            'expected_outcome': predicted_outcome,
            'confidence': confidence,
            'outcome_scores': dict(outcome_scores),
            'observed': False,
        }
        
        self.predictions.append(prediction)
        self.predict_count += 1
        return prediction
    
    # ============= 2. observe =============
    
    def observe(self,
                prediction_id: Optional[str] = None,
                actual_outcome: Optional[str] = None) -> Dict[str, Any]:
        """
        예측 vs 실제 비교 → error 계산 + (outcome 있으면) 학습.
        
        Args:
            prediction_id: 비교할 prediction ID (None이면 가장 최근 미관측)
            actual_outcome: 'good'/'bad'/'neutral' (옵션)
        
        Returns: error_record dict
        """
        # prediction 찾기
        pred = None
        if prediction_id is not None:
            for p in self.predictions:
                if p['id'] == prediction_id:
                    pred = p
                    break
        else:
            # 가장 최근 미관측
            for p in reversed(self.predictions):
                if not p.get('observed'):
                    pred = p
                    break
        
        if pred is None:
            return {
                'error': True,
                'reason': 'no_prediction',
                'time': self.time,
            }
        
        # 현재 실제 상태
        actual = self._capture_state()
        
        # 호르몬 차이 (per hormone)
        hormone_errors = {}
        for h, expected_level in pred['expected_state']['hormones'].items():
            actual_level = actual['hormones'].get(h, expected_level)
            hormone_errors[h] = float(actual_level - expected_level)
        
        # mood error (절댓값 합)
        mood_error = float(
            abs(actual['mood_valence'] - pred['expected_state']['mood_valence'])
            + abs(actual['mood_arousal'] - pred['expected_state']['mood_arousal'])
        )
        
        # outcome match
        outcome_match = None
        if actual_outcome:
            if actual_outcome not in self.OUTCOME_LABELS:
                actual_outcome = 'neutral'
            outcome_match = (actual_outcome == pred['expected_outcome'])
        
        error_record = {
            'prediction_id': pred['id'],
            'time': self.time,
            'mood_error': mood_error,
            'hormone_errors': hormone_errors,
            'predicted_outcome': pred['expected_outcome'],
            'actual_outcome': actual_outcome,
            'outcome_match': outcome_match,
            'pred_confidence': pred.get('confidence', 0.0),
        }
        
        pred['observed'] = True
        self.errors.append(error_record)
        self.observe_count += 1
        
        # outcome 있으면 학습
        if actual_outcome:
            self._learn_from_outcome(pred, actual_outcome)
        
        return error_record
    
    # ============= 내부: 학습 (Hebbian) =============
    
    def _learn_from_outcome(self, prediction: Dict, actual_outcome: str):
        """
        prediction 시점 호르몬 상태 → actual_outcome 매핑 학습.
        
        규칙:
        - actual_outcome 라벨에 deviation × LR 강화
        - 다른 라벨에 deviation × LR × NEGATIVE_REINFORCE_RATIO 약화
        - 가중치 [-1, 1] clamp
        """
        if actual_outcome not in self.OUTCOME_LABELS:
            return
        
        snapshot = prediction.get('expected_state', {})
        hormones = snapshot.get('hormones', {})
        
        for h, level in hormones.items():
            if h not in self.hormone_outcome_weights:
                continue
            deviation = level - 0.5
            # actual outcome 강화
            self.hormone_outcome_weights[h][actual_outcome] += (
                self.LEARNING_RATE * deviation
            )
            # 다른 outcome 약화 (winner-take-soft)
            for other in self.OUTCOME_LABELS:
                if other != actual_outcome:
                    self.hormone_outcome_weights[h][other] -= (
                        self.LEARNING_RATE * deviation
                        * self.NEGATIVE_REINFORCE_RATIO
                    )
            # clamp
            for label in self.OUTCOME_LABELS:
                self.hormone_outcome_weights[h][label] = float(np.clip(
                    self.hormone_outcome_weights[h][label],
                    self.WEIGHT_MIN, self.WEIGHT_MAX,
                ))
        
        self.update_count += 1
    
    # ============= 3. update_from_outcome (SD 짝꿍) =============
    
    def update_from_outcome(self,
                            predicted_doubt: float,
                            actual_outcome: str) -> Dict[str, Any]:
        """
        SD.update_from_outcome과 짝꿍.
        
        SD는 baseline 의심 조정. AI는 호르몬-outcome 매핑 학습.
        
        외부에서 호출 시:
        - 미관측 prediction 있으면 → observe()로 학습
        - 없으면 → 즉석으로 현재 상태 기준 학습 (inline)
        
        Args:
            predicted_doubt: SD가 예측한 doubt level (참고용)
            actual_outcome: 'good'/'bad'/'neutral'
        
        Returns: 처리 결과
        """
        # 정규화
        if actual_outcome not in self.OUTCOME_LABELS:
            actual_outcome = 'neutral'
        
        # 미관측 최신 prediction 있으면 observe로 처리
        last_unobs = None
        for p in reversed(self.predictions):
            if not p.get('observed'):
                last_unobs = p
                break
        
        if last_unobs is not None:
            error_rec = self.observe(prediction_id=last_unobs['id'],
                                     actual_outcome=actual_outcome)
            error_rec['source'] = 'sd_outcome'
            error_rec['predicted_doubt'] = float(predicted_doubt)
            return error_rec
        
        # 없으면 inline (현재 상태 → 즉석 학습)
        current = self._capture_state()
        inline_pred = {
            'id': f'pred_inline_{self.predict_count:04d}',
            'time': self.time,
            'expected_state': current,
            'expected_outcome': 'neutral',  # inline이라 예측 없음
            'confidence': 0.0,
            'observed': True,
            'inline': True,
        }
        self._learn_from_outcome(inline_pred, actual_outcome)
        return {
            'inline_learn': True,
            'time': self.time,
            'actual_outcome': actual_outcome,
            'predicted_doubt': float(predicted_doubt),
            'source': 'sd_outcome_inline',
        }
    
    # ============= 4. get_hormone_signature =============
    
    def get_hormone_signature(self, outcome_label: str) -> Dict[str, float]:
        """
        주어진 outcome에 학습된 호르몬 시그니처.
        
        weight > 0: 그 호르몬 ↑가 outcome 신호
        weight < 0: 그 호르몬 ↑가 outcome 반대 신호
        weight ≈ 0: 학습 안 됨 또는 무관
        
        Returns: {hormone_name: weight}
        """
        if outcome_label not in self.OUTCOME_LABELS:
            return {}
        return {
            h: float(self.hormone_outcome_weights[h][outcome_label])
            for h in self.HORMONE_NAMES_TRACKED
        }
    
    def get_dominant_hormones(self,
                              outcome_label: str,
                              top_n: int = 3) -> List[Tuple[str, float]]:
        """
        outcome에 가장 강하게 학습된 호르몬 top_n.
        |weight| 기준 정렬.
        """
        sig = self.get_hormone_signature(outcome_label)
        # 결정론 정렬: |weight| 내림차순, name 오름차순
        sorted_h = sorted(sig.items(), key=lambda x: (-abs(x[1]), x[0]))
        return sorted_h[:top_n]
    
    # ============= 5. tick =============
    
    def tick(self, dt: float):
        """
        매 tick 호출 (옵션). 시간 누적만.
        predict는 자동 안 함 — 외부에서 명시적 호출.
        """
        if dt > 0:
            self.time += dt
    
    # ============= 6. 진단 =============
    
    def get_state(self) -> Dict[str, Any]:
        pending = sum(1 for p in self.predictions if not p.get('observed'))
        recent_mood_errors = [e['mood_error'] for e in self.errors]
        recent_outcome_matches = [e['outcome_match'] for e in self.errors
                                  if e.get('outcome_match') is not None]
        
        return {
            'time': self.time,
            'predict_count': self.predict_count,
            'observe_count': self.observe_count,
            'update_count': self.update_count,
            'pending_predictions': pending,
            'mean_mood_error': (float(np.mean(recent_mood_errors))
                                if recent_mood_errors else 0.0),
            'outcome_accuracy': (float(np.mean(recent_outcome_matches))
                                 if recent_outcome_matches else None),
            'weight_summary': {
                label: {
                    'max_abs': max((abs(self.hormone_outcome_weights[h][label])
                                    for h in self.HORMONE_NAMES_TRACKED),
                                   default=0.0),
                    'sum_abs': sum(abs(self.hormone_outcome_weights[h][label])
                                   for h in self.HORMONE_NAMES_TRACKED),
                }
                for label in self.OUTCOME_LABELS
            },
        }
    
    def __repr__(self):
        s = self.get_state()
        return (f"ActiveInference(t={self.time:.1f}s, "
                f"predicts={s['predict_count']}, "
                f"learns={s['update_count']}, "
                f"acc={s['outcome_accuracy']})")
