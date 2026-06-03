"""
EVE v32 - C7: EmotionRegulation (감정 조절)
=============================================

EVE의 감정 자기 조절 — 부정 정서를 인식하고 능동적으로 다스림.

학계:
- Gross 1998 — Process Model of Emotion Regulation (5단계)
- Gross & Thompson 2007 — Emotion Regulation: Conceptual Foundations
- Ochsner & Gross 2005 — Cognitive control of emotion (reappraisal vs suppression)
- Lazarus 1991 — Cognitive Appraisal Theory
- Davidson 1998 — Affective Style
- McRae & Gross 2020 — Emotion regulation strategies meta-analysis

Gross 5단계 전략:
1. situation_selection (회피)        — 외부 행동 (EVE 본 모듈 X)
2. situation_modification (수정)     — 외부 행동 (EVE 본 모듈 X)
3. attentional_deployment (주의)     — distraction ★
4. cognitive_change (인지)           — reappraisal ★★ (가장 효과적)
5. response_modulation (반응)        — suppression ★

EVE 본 모듈 구현 3가지 (인지 수준):

A. distraction:
   부정 카테고리(focus)에서 다른 (긍정) 카테고리로 주의 전환.
   - WM focus 강제 변경
   - SA에서 긍정 카테고리 활성 부스트
   - 즉각 효과, 약한 (재발 가능)

B. reappraisal (Ochsner — '재평가가 가장 효과적'):
   부정 상황의 의미를 재해석.
   - Counterfactual로 "이게 더 나빴을 수도" (다행)
   - Temporal로 "곧 지나갈 거야"
   - 학습된 인과 다시 평가
   - 효과 큼, 호르몬 변조 + EM 일화 의미 재구성

C. suppression:
   호르몬 직접 억제.
   - cortisol 약하게 감소
   - GABA 증강 (있으면)
   - 빠르지만 비용: 누적되면 NE↑ (Gross — suppression 비효율)

자동 트리거:
- cortisol > 0.7 — 스트레스 너무 높음
- regret 직후 (CF에서)
- 부정 dominant_need (tension)

전략 학습:
- regulation_history에 (strategy, before_cort, after_cort) 기록
- 가장 효과적인 전략 우선 추천

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론
- 호르몬 변조 (조절 결과가 호르몬 자극)
- 카테고리 단위
- 모듈 chain (HS + WM + SA + CF + TM + EM)

핵심 함수 8:
1. detect_need() → bool (조절 필요?)
2. regulate(strategy=None, auto=False) → 자동 또는 수동
3. distraction(target_category=None)
4. reappraisal(situation=None)
5. suppression()
6. suggest_strategy() → 학습된 효과 기반
7. evaluate_effect() → 직전 조절 효과 측정
8. tick(dt)

Author: 김민석 + Claude v32 (C7)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import deque


class EmotionRegulation:
    """EVE v32 C7 EmotionRegulation — Gross 5 전략 中 인지 3가지"""

    # 자동 트리거 임계
    AUTO_TRIGGER_CORT = 0.55  # was 0.70, 베타 운영에서 안 발동되어 낮춤
    AUTO_TRIGGER_TENSION = 0.55  # was 0.65

    # 전략별 호르몬 영향
    DISTRACTION_CORT_DECREASE = 0.03    # 약한 즉각
    REAPPRAISAL_CORT_DECREASE = 0.08    # 가장 강함
    REAPPRAISAL_SEROTONIN_BOOST = 0.05  # 인지 안정화
    SUPPRESSION_CORT_DECREASE = 0.05    # 직접
    SUPPRESSION_NE_BOOST = 0.03         # 비용 (각성 ↑)
    SUPPRESSION_GABA_BOOST = 0.02       # 짧은 안정

    # 긍정 카테고리 그룹 (distraction 타겟 후보)
    POSITIVE_CATEGORIES_DEFAULT = {
        '기쁨', '성취', '재밌다', '맛있다',  # reward
        '대화', '함께', '친구',              # social
        '편안', '잠', '휴식',                # rest
        '집중', '관찰',                       # attention
    }

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 counterfactual=None,
                 temporal=None,
                 episodic_memory=None,
                 history_size: int = 50):
        """
        Args:
            hormone_system: 필수 (조절 대상)
            spreading_activation: 필수 (distraction 타겟)
            나머지: 옵션
        """
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.cf = counterfactual
        self.tm = temporal
        self.em = episodic_memory

        self.time = 0.0

        # 전략 효과 학습 (strategy → 효과 점수 누적)
        self.strategy_effectiveness: Dict[str, List[float]] = {
            'distraction': [],
            'reappraisal': [],
            'suppression': [],
        }
        # 직전 조절 (효과 측정용)
        self.last_regulation: Optional[Dict[str, Any]] = None
        # history
        self.history: deque = deque(maxlen=history_size)

        # 통계
        self.detect_count = 0
        self.regulate_count = 0
        self.auto_trigger_count = 0
        self.distraction_count = 0
        self.reappraisal_count = 0
        self.suppression_count = 0

    # ============= 1. detect_need =============
    def detect_need(self) -> Dict[str, Any]:
        """
        조절 필요한지 감지.

        Returns: {
            'need': bool,
            'reasons': [str],
            'cort_level': float,
            'severity': float,  # 0~1
        }
        """
        self.detect_count += 1
        reasons = []
        cort = self.hs.hormones.get('cortisol')
        cort_level = cort.level if cort else 0.0

        if cort_level > self.AUTO_TRIGGER_CORT:
            reasons.append(f'cortisol high ({cort_level:.2f})')

        # CF에서 직전 regret 발생?
        if self.cf is not None:
            recent_regret = (self.cf.regret_count > 0
                            and len(self.cf.simulations) > 0)
            if recent_regret:
                last_sim = list(self.cf.simulations)[-1]
                if last_sim.get('signal') == 'regret':
                    age = self.time - last_sim.get('time', 0)
                    if age < 30.0:  # 30초 이내
                        reasons.append('recent regret')

        # severity = cort + regret 가산
        severity = float(np.clip(
            (cort_level - 0.3) * 1.5 + (0.3 if 'recent regret' in reasons else 0),
            0.0, 1.0
        ))

        return {
            'need': len(reasons) > 0,
            'reasons': reasons,
            'cort_level': cort_level,
            'severity': severity,
        }

    # ============= 2. distraction =============
    def distraction(self,
                   target_category: Optional[str] = None
                   ) -> Dict[str, Any]:
        """
        주의 전환. 긍정 카테고리 활성 + WM focus 변경.

        Args:
            target_category: None이면 자동 선택 (긍정 카테고리 中 SA 가장 강한 거)
        """
        self.regulate_count += 1
        self.distraction_count += 1

        # 타겟 결정
        if target_category is None:
            # 긍정 카테고리 中 SA에 있는 거 (그 카테고리가 그래프에 있는 거 우선)
            candidates = [
                cat for cat in self.POSITIVE_CATEGORIES_DEFAULT
                if cat in self.sa.neighbors or cat in self.sa.activations
            ]
            if not candidates:
                # 전부 없으면 default 첫 거
                candidates = list(self.POSITIVE_CATEGORIES_DEFAULT)
            # 결정론 (알파벳)
            candidates.sort()
            target_category = candidates[0]

        before_cort = self._get_level('cortisol')

        # SA 활성 (강하게 — 주의 끌어당김)
        self.sa.activate(target_category, 0.7)

        # WM focus로 (있으면)
        if self.wm is not None and hasattr(self.wm, 'add'):
            try:
                self.wm.add(target_category, salience=0.8, source='regulation')
            except Exception:
                pass

        # 호르몬 — 약한 cort 감소
        self._modify_hormone('cortisol', -self.DISTRACTION_CORT_DECREASE)

        after_cort = self._get_level('cortisol')

        result = {
            'strategy': 'distraction',
            'target': target_category,
            'before_cort': before_cort,
            'after_cort': after_cort,
            'cort_delta': after_cort - before_cort,
            'time': self.time,
        }
        self.last_regulation = result
        self.history.append({'event': 'regulate', **result})
        return result

    # ============= 3. reappraisal =============
    def reappraisal(self,
                   situation_category: Optional[str] = None
                   ) -> Dict[str, Any]:
        """
        인지 재평가 (Ochsner — 가장 효과적).

        - CF로 "더 나빴을 수도" 시뮬 (worse_alternative)
        - TM으로 "곧 지나갈 거야" 시간 perspective
        - 호르몬 — cort↓ 강하게, serotonin↑ (안정)
        """
        self.regulate_count += 1
        self.reappraisal_count += 1

        before_cort = self._get_level('cortisol')
        before_ser = self._get_level('serotonin')

        reframe_notes = []

        # 상황 카테고리 결정
        if situation_category is None and self.cf is not None and self.cf.simulations:
            # 최근 regret 일화에서
            last_sim = list(self.cf.simulations)[-1]
            situation_category = last_sim.get('actual_action')

        # 1) Counterfactual: "더 나빴을 수도"
        if situation_category and self.cf is not None:
            try:
                # 그 상황의 closest_world 中 더 나쁜 것 찾기
                alts = self.cf.closest_world(situation_category, top_n=2)
                for alt_cat, _ in alts:
                    cf_action = self.cf.counterfactual_action(
                        situation_category, alt_cat)
                    # alternative_score < actual_score → "다행"
                    if cf_action.get('outcome_score', 0.0) < -0.2:
                        reframe_notes.append(
                            f"could be worse ('{alt_cat}')")
                        break
            except Exception:
                pass

        # 2) Temporal: "곧 지나갈 거야"
        if situation_category and self.tm is not None:
            try:
                dist = self.tm.time_distance(situation_category)
                if dist is not None and dist > 60:
                    reframe_notes.append(f"already {dist:.0f}s ago")
            except Exception:
                pass

        # 3) 호르몬 — 가장 강한 효과
        self._modify_hormone('cortisol', -self.REAPPRAISAL_CORT_DECREASE)
        self._modify_hormone('serotonin', self.REAPPRAISAL_SEROTONIN_BOOST)

        after_cort = self._get_level('cortisol')
        after_ser = self._get_level('serotonin')

        result = {
            'strategy': 'reappraisal',
            'situation': situation_category,
            'reframe_notes': reframe_notes,
            'before_cort': before_cort,
            'after_cort': after_cort,
            'cort_delta': after_cort - before_cort,
            'serotonin_delta': after_ser - before_ser,
            'time': self.time,
        }
        self.last_regulation = result
        self.history.append({'event': 'regulate', **result})
        return result

    # ============= 4. suppression =============
    def suppression(self) -> Dict[str, Any]:
        """
        반응 억제 — 호르몬 직접 억제.
        Gross 비판: 효과 빠르지만 비용 (NE↑ — 누적)
        """
        self.regulate_count += 1
        self.suppression_count += 1

        before_cort = self._get_level('cortisol')
        before_ne = self._get_level('norepinephrine')

        # cort↓
        self._modify_hormone('cortisol', -self.SUPPRESSION_CORT_DECREASE)
        # GABA↑ (있으면)
        self._modify_hormone('gaba', self.SUPPRESSION_GABA_BOOST)
        # 비용: NE↑ (각성 — 누적되면 비효율)
        self._modify_hormone('norepinephrine', self.SUPPRESSION_NE_BOOST)

        after_cort = self._get_level('cortisol')
        after_ne = self._get_level('norepinephrine')

        result = {
            'strategy': 'suppression',
            'before_cort': before_cort,
            'after_cort': after_cort,
            'cort_delta': after_cort - before_cort,
            'ne_delta': after_ne - before_ne,
            'time': self.time,
        }
        self.last_regulation = result
        self.history.append({'event': 'regulate', **result})
        return result

    # ============= 5. regulate (자동 또는 수동) =============
    def regulate(self,
                strategy: Optional[str] = None,
                auto: bool = False) -> Optional[Dict[str, Any]]:
        """
        조절 실행. strategy=None이면 추천 전략 사용.

        Args:
            strategy: 'distraction' / 'reappraisal' / 'suppression' / None
            auto: True면 detect_need 통과 시에만 실행
        """
        if auto:
            need = self.detect_need()
            if not need['need']:
                return None
            self.auto_trigger_count += 1

        if strategy is None:
            strategy = self.suggest_strategy()

        if strategy == 'distraction':
            return self.distraction()
        elif strategy == 'reappraisal':
            return self.reappraisal()
        elif strategy == 'suppression':
            return self.suppression()
        else:
            return None

    # ============= 6. suggest_strategy =============
    def suggest_strategy(self) -> str:
        """
        학습된 효과 기반 추천. 첫 사용 시 default = reappraisal.

        효과 = 평균 cort_delta (음수가 좋음)
        """
        averages: Dict[str, float] = {}
        for s, deltas in self.strategy_effectiveness.items():
            if deltas:
                averages[s] = sum(deltas) / len(deltas)

        if not averages:
            return 'reappraisal'  # default — Gross 가장 효과적

        # 가장 효과적인 (cort_delta 가장 음수)
        # 결정론 (효과 ↑, 알파벳 ↑)
        sorted_strategies = sorted(averages.items(),
                                  key=lambda kv: (kv[1], kv[0]))
        return sorted_strategies[0][0]

    # ============= 7. evaluate_effect =============
    def evaluate_effect(self) -> Optional[Dict[str, Any]]:
        """
        직전 조절의 효과 측정 + 학습.
        last_regulation의 cort_delta를 strategy_effectiveness에 기록.
        """
        if self.last_regulation is None:
            return None

        strategy = self.last_regulation.get('strategy')
        delta = self.last_regulation.get('cort_delta', 0.0)

        if strategy in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy].append(float(delta))
            # max 20개만 유지
            if len(self.strategy_effectiveness[strategy]) > 20:
                self.strategy_effectiveness[strategy] = (
                    self.strategy_effectiveness[strategy][-20:])

        return {
            'strategy': strategy,
            'cort_delta': delta,
            'effective': delta < -0.01,  # 의미있는 감소
        }

    # ============= 헬퍼 =============
    def _get_level(self, name: str) -> float:
        if self.hs is None:
            return 0.0
        h = self.hs.hormones.get(name)
        if h is None:
            return 0.0
        return float(h.level)

    def _modify_hormone(self, name: str, delta: float):
        if self.hs is None:
            return
        h = self.hs.hormones.get(name)
        if h is None:
            return
        if hasattr(h, 'stimulate'):
            h.stimulate(delta)
        else:
            h.level = float(np.clip(h.level + delta, 0.0, 1.0))

    # ============= 8. tick =============
    def tick(self, dt: float):
        """
        시간 누적 + 자동 트리거 + 직전 효과 학습.
        """
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

        # 직전 조절 효과 학습 (한 번)
        if self.last_regulation is not None:
            age = self.time - self.last_regulation.get('time', self.time)
            if age >= dt:  # 1 tick 지났으면
                # 아직 평가 안 됐으면 평가
                strategy = self.last_regulation.get('strategy')
                if strategy and not self.last_regulation.get('_evaluated'):
                    self.evaluate_effect()
                    self.last_regulation['_evaluated'] = True

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        avg_eff = {
            s: (sum(d) / len(d) if d else None)
            for s, d in self.strategy_effectiveness.items()
        }
        return {
            'time': self.time,
            'detect_count': self.detect_count,
            'regulate_count': self.regulate_count,
            'auto_trigger_count': self.auto_trigger_count,
            'distraction_count': self.distraction_count,
            'reappraisal_count': self.reappraisal_count,
            'suppression_count': self.suppression_count,
            'avg_effectiveness': avg_eff,
            'history_size': len(self.history),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"EmotionRegulation(t={s['time']:.1f}s, "
                f"regulates={s['regulate_count']}, "
                f"auto={s['auto_trigger_count']})")
