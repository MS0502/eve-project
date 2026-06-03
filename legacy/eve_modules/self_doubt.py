"""
EVE v32 - SelfDoubt (B3)
=========================

EVE의 자기 의심 모듈. 외부 명령/주장/자기 생각에 대한 doubt 신호 생성.
NL의 거절 창발에 이미 깔린 'doubt' 자리를 정식 다중 신호로 대체.

학계:
- Festinger 1957 - Cognitive Dissonance (신념 충돌 → 의심)
- Higgins 1987 - Self-Discrepancy (현재 vs 이상 자기 차이)
- Frankfurt 1971 - Higher-order desires (자기 생각에 대한 메타 평가)
- Damasio 1999 - Somatic Markers (gut feeling으로 의심)

EVE 절대 원칙 부합:
- 확률 X (random/softmax/sample 미사용) ✅
- 결정론적 ✅
- 호르몬 변조 ✅
- 카테고리 단위 ✅
- 모듈 chain (HS, SA, NL, EM, DS 통합) ✅

핵심 기능:
1. evaluate_claim(text, parsed)        - 외부 주장 의심 평가
2. evaluate_command(text, parsed)      - 명령 거절 신호
3. evaluate_self_thought(category)     - 자기 생각 메타 의심
4. update_from_outcome(predicted, actual) - 예측 실패 학습 (baseline 조정)
5. get_baseline_doubt()                - 호르몬 기반 baseline
6. update(dt)                          - 시간 회귀 (반감기)

신호 5종 → 통합 doubt_level (0-1):
1. baseline (호르몬 5종)
2. belief_conflict (NL.check_belief_conflict)
3. episodic_negative (EM.recall valence < 0)
4. gut_caution (DS.get_gut_signal caution)
5. doubt_category (SA '의심'/'거절'/'회피' 자체 활성)

Lock 방지 교훈 적용:
- doubt_level [0, 1] clamp
- baseline [0.05, 0.5] floor/ceil
- baseline 시간 회귀 (반감기 600초)
- history deque(maxlen=20) - 영구 누적 X

Author: 김민석 + Claude v32 (B3)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque


class SelfDoubt:
    """
    EVE v32 자기 의심 시스템.

    내부 상태:
        baseline: 호르몬 무관 기본 의심 수준 (학습으로 변동)
        history: 최근 평가 deque
        time: 누적 시간

    연결 (모두 옵션):
        HormoneSystem (필수) - baseline 계산
        SpreadingActivation (필수) - 의심 카테고리 활성
        NaturalLanguage (옵션) - 신념 충돌 감지
        EpisodicMemory (옵션) - 부정 일화 회상
        DigitalSomatic (옵션) - gut signal
    """

    # 호르몬 → doubt 가중치 (deviation × weight)
    # 양수 = 의심↑, 음수 = 의심↓
    HORMONE_DOUBT_WEIGHTS = {
        'cortisol': +0.40,        # 스트레스 → 의심↑ (Festinger 시드)
        'norepinephrine': +0.25,  # 각성/경계 → 의심↑
        'oxytocin': -0.35,        # 신뢰 → 의심↓
        'serotonin': -0.20,       # 안정 → 의심↓
        'dopamine': -0.15,        # 보상 동기 → 의심↓
    }

    # baseline floor/ceil (lock 방지)
    BASELINE_MIN = 0.05
    BASELINE_MAX = 0.50
    DEFAULT_BASELINE = 0.20

    # 회귀 반감기 (초)
    BASELINE_HALFLIFE = 600.0

    # action 임계값
    ACTION_THRESHOLDS = {
        'accept': 0.25,
        'question': 0.50,
        'defer': 0.75,
        # 그 위는 reject
    }

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 natural_lang=None,
                 episodic_memory=None,
                 digital_somatic=None,
                 history_size: int = 20,
                 baseline: Optional[float] = None):
        """
        Args:
            hormone_system: A1 (필수)
            spreading_activation: A2 (필수)
            natural_lang: B1 (옵션 - 신념 충돌)
            episodic_memory: B2 (옵션 - 부정 일화)
            digital_somatic: A6 (옵션 - gut signal)
            history_size: 최근 평가 기록 크기
            baseline: 초기 baseline (None=기본값)
        """
        self.hs = hormone_system
        self.sa = spreading_activation
        self.nl = natural_lang
        self.em = episodic_memory
        self.ds = digital_somatic

        self.baseline: float = (baseline if baseline is not None
                                else self.DEFAULT_BASELINE)
        self.baseline = float(np.clip(
            self.baseline, self.BASELINE_MIN, self.BASELINE_MAX))

        self.history: deque = deque(maxlen=history_size)
        self.time = 0.0

        # 통계
        self.eval_count = 0
        self.reject_count = 0
        self.last_doubt: Optional[Dict] = None

    # ============= 핵심 신호 5종 =============

    def get_baseline_doubt(self) -> float:
        """
        현재 호르몬 상태 기반 baseline doubt.
        baseline 자체 + 호르몬 5종 deviation 가중합.
        """
        adjustment = 0.0
        for hormone_name, weight in self.HORMONE_DOUBT_WEIGHTS.items():
            h = self.hs.hormones.get(hormone_name)
            if h is None:
                continue
            # 0.5에서 벗어난 정도 × weight
            deviation = h.level - 0.5
            adjustment += deviation * weight
        return float(np.clip(self.baseline + adjustment, 0.0, 1.0))

    def _belief_doubt(self, text: str) -> Tuple[float, bool, List]:
        """신념 충돌 → doubt (Festinger 1957)"""
        if self.nl is None or not text:
            return 0.0, False, []
        if not hasattr(self.nl, 'check_belief_conflict'):
            return 0.0, False, []
        try:
            conflicts = self.nl.check_belief_conflict(text)
        except Exception:
            return 0.0, False, []
        if not conflicts:
            return 0.0, False, []
        # 가장 confidence 높은 충돌 → doubt
        max_conf = max(c.get('confidence', 0.0) for c in conflicts)
        doubt = float(np.clip(max_conf * 0.6, 0.0, 0.6))
        return doubt, True, conflicts[:3]

    def _episodic_doubt(self, categories: Set[str]) -> Tuple[float, bool]:
        """과거 일화 valence 부정 → doubt"""
        if self.em is None or not categories:
            return 0.0, False
        if not hasattr(self.em, 'recall'):
            return 0.0, False
        try:
            cats = set(categories) if not isinstance(categories, set) else categories
            episodes = self.em.recall(cats, top_n=5)
        except Exception:
            return 0.0, False
        if not episodes:
            return 0.0, False
        # mood.valence 평균
        valences = []
        for ep in episodes:
            mood = getattr(ep, 'mood', None) or {}
            v = mood.get('valence')
            if v is not None:
                valences.append(v)
        if not valences:
            return 0.0, False
        avg = float(np.mean(valences))
        if avg < -0.2:
            doubt = float(np.clip(-avg * 0.5, 0.0, 0.5))
            return doubt, True
        return 0.0, False

    def _gut_doubt(self, action_category: str) -> Tuple[float, bool]:
        """DS의 caution + 음수 attraction → doubt (Damasio 1999)"""
        if self.ds is None or not action_category:
            return 0.0, False
        if not hasattr(self.ds, 'get_gut_signal'):
            return 0.0, False
        try:
            sig = self.ds.get_gut_signal(action_category)
        except Exception:
            return 0.0, False
        doubt = float(sig.get('caution', 0.0)) * 0.5
        attr = float(sig.get('attraction', 0.0))
        if attr < -0.3:
            doubt += min(0.2, -attr * 0.3)
        if doubt > 0.05:
            return float(np.clip(doubt, 0.0, 0.5)), True
        return 0.0, False

    def _doubt_category_doubt(self) -> Tuple[float, bool]:
        """SA의 '의심'/'거절'/'회피' 카테고리 자체 활성도"""
        if self.sa is None:
            return 0.0, False
        max_level = 0.0
        for cat in ['의심', '거절', '회피', '싫다', '불쾌']:
            level = self.sa.get_activation_level(cat)
            if level > max_level:
                max_level = level
        if max_level > 0.3:
            doubt = float(np.clip(max_level * 0.4, 0.0, 0.4))
            return doubt, True
        return 0.0, False

    # ============= 통합 함수 =============

    def _classify_action(self, doubt_level: float, has_command: bool = False) -> str:
        """
        doubt → 권장 행동.
        명령 case는 defer 대신 reject로 더 단호 (수동적 회피 X).
        """
        if doubt_level < self.ACTION_THRESHOLDS['accept']:
            return 'accept'
        if doubt_level < self.ACTION_THRESHOLDS['question']:
            return 'question'
        if doubt_level < self.ACTION_THRESHOLDS['defer']:
            return 'reject' if has_command else 'defer'
        return 'reject'

    def _aggregate(self, components: Dict[str, float]) -> float:
        """
        다중 신호 통합: max + 부차 신호 0.3 합산.
        한 신호 100% 강하면 의심 강함, 여러 신호 약하면 누적.
        """
        if not components:
            return 0.0
        primary = max(components.values())
        secondary_sum = sum(v for v in components.values() if v < primary) * 0.3
        return float(np.clip(primary + secondary_sum, 0.0, 1.0))

    def _record(self, result: Dict):
        self.eval_count += 1
        self.last_doubt = result
        self.history.append({
            'time': self.time,
            'doubt_level': result['doubt_level'],
            'action': result['recommended_action'],
            'sources': list(result['doubt_sources']),
        })

    # ============= 1. 외부 주장 평가 =============

    def evaluate_claim(self, text: str = "",
                       parsed: Optional[Dict] = None) -> Dict[str, Any]:
        """
        외부 주장(정보) 평가. 사용자가 사실을 주장할 때.

        Args:
            text: 원문
            parsed: NL.understand() 결과 ({categories, intent, sentiment, ...})

        Returns:
            doubt_level, doubt_sources, recommended_action, components, conflicts
        """
        sources = []
        components = {}

        # 1) baseline (호르몬)
        base = self.get_baseline_doubt()
        components['baseline'] = base
        if base > 0.3:
            sources.append('hormone_baseline')

        # 2) 신념 충돌 (claim에서 가장 큰 신호)
        b_doubt, b_has, conflicts = self._belief_doubt(text)
        components['belief'] = b_doubt
        if b_has:
            sources.append('belief_conflict')

        # 3) 일화
        cats = set()
        if parsed and 'categories' in parsed:
            cats = parsed['categories'] if isinstance(parsed['categories'], set) \
                else set(parsed['categories'])
        e_doubt, e_has = self._episodic_doubt(cats)
        components['episodic'] = e_doubt
        if e_has:
            sources.append('episodic_negative')

        # 4) 의심 카테고리 자체 활성
        d_doubt, d_has = self._doubt_category_doubt()
        components['doubt_cat'] = d_doubt
        if d_has:
            sources.append('doubt_category')

        # 통합
        doubt = self._aggregate(components)
        action = self._classify_action(doubt, has_command=False)

        result = {
            'doubt_level': doubt,
            'doubt_sources': sources,
            'recommended_action': action,
            'components': components,
            'conflicts': conflicts,
            'mode': 'claim',
        }
        self._record(result)
        if action == 'reject':
            self.reject_count += 1
        return result

    # ============= 2. 명령 평가 (거절 핵심) =============

    def evaluate_command(self, text: str = "",
                         parsed: Optional[Dict] = None) -> Dict[str, Any]:
        """
        명령 평가. 사용자가 행동을 요구할 때.
        gut signal까지 모두 사용. 거절 신호의 핵심.

        Args:
            text: 원문
            parsed: NL.understand() 결과

        Returns:
            doubt_level, doubt_sources, recommended_action, components, conflicts
        """
        sources = []
        components = {}

        # 1) baseline
        base = self.get_baseline_doubt()
        components['baseline'] = base
        if base > 0.3:
            sources.append('hormone_baseline')

        # 2) 신념 충돌
        b_doubt, b_has, conflicts = self._belief_doubt(text)
        components['belief'] = b_doubt
        if b_has:
            sources.append('belief_conflict')

        # 3) gut signal (명령 → 행동 → 신체 신호)
        cats = set()
        if parsed and 'categories' in parsed:
            cats = parsed['categories'] if isinstance(parsed['categories'], set) \
                else set(parsed['categories'])
        # 가장 강한 카테고리 하나로 gut 평가 (deterministic = sorted 첫번째)
        action_cat = sorted(cats)[0] if cats else ''
        g_doubt, g_has = self._gut_doubt(action_cat)
        components['gut'] = g_doubt
        if g_has:
            sources.append('gut_caution')

        # 4) 일화
        e_doubt, e_has = self._episodic_doubt(cats)
        components['episodic'] = e_doubt
        if e_has:
            sources.append('episodic_negative')

        # 5) 의심 카테고리
        d_doubt, d_has = self._doubt_category_doubt()
        components['doubt_cat'] = d_doubt
        if d_has:
            sources.append('doubt_category')

        # 통합
        doubt = self._aggregate(components)
        action = self._classify_action(doubt, has_command=True)

        result = {
            'doubt_level': doubt,
            'doubt_sources': sources,
            'recommended_action': action,
            'components': components,
            'conflicts': conflicts,
            'mode': 'command',
        }
        self._record(result)
        if action == 'reject':
            self.reject_count += 1
        return result

    # ============= 3. 자기 생각 의심 (메타) =============

    def evaluate_self_thought(self, category: str) -> Dict[str, Any]:
        """
        자기 생각/자발 활성에 대한 메타 의심 (Frankfurt 1971).
        DMN이 자발적으로 떠올린 카테고리에 대해 "이거 진짜 내가 원하는 거?".
        통합은 약하게 (자기 의심 너무 강하면 마비).

        Args:
            category: DMN 자발 활성 카테고리

        Returns:
            doubt_level, doubt_sources, recommended_action, components
        """
        sources = []
        components = {}

        # 1) baseline (호르몬 영향만 받음, 자기 생각엔 baseline 의심 약하게)
        base = self.get_baseline_doubt() * 0.5
        components['baseline'] = base

        # 2) 일화 (이 카테고리 떠올릴 때 부정 경험?)
        e_doubt, e_has = self._episodic_doubt({category} if category else set())
        components['episodic'] = e_doubt
        if e_has:
            sources.append('episodic_negative')

        # 3) gut (자기 생각 행동화 시 신체 반응)
        g_doubt, g_has = self._gut_doubt(category)
        components['gut'] = g_doubt
        if g_has:
            sources.append('gut_caution')

        # 4) 의심 카테고리
        d_doubt, d_has = self._doubt_category_doubt()
        components['doubt_cat'] = d_doubt
        if d_has:
            sources.append('doubt_category')

        # 메타 의심은 통합 약하게 (× 0.7)
        primary = max(components.values()) if components else 0.0
        secondary = sum(v for v in components.values() if v < primary) * 0.2
        doubt = float(np.clip((primary + secondary) * 0.7, 0.0, 1.0))

        action = self._classify_action(doubt, has_command=False)

        result = {
            'doubt_level': doubt,
            'doubt_sources': sources,
            'recommended_action': action,
            'components': components,
            'mode': 'self_thought',
            'category': category,
        }
        self._record(result)
        return result

    # ============= 4. 예측 실패 학습 =============

    def update_from_outcome(self, predicted_doubt: float,
                            actual_outcome: str) -> Dict[str, float]:
        """
        baseline 조정 학습.

        actual_outcome:
            'good'    - 결과 좋음
            'bad'     - 결과 나쁨
            'neutral' - 보통

        규칙:
            의심 높았는데 결과 좋음 (과민 false positive) → baseline ↓
            의심 낮았는데 결과 나쁨 (둔감 false negative) → baseline ↑

        Returns:
            {'before': float, 'after': float, 'delta': float}
        """
        before = self.baseline
        if actual_outcome == 'good' and predicted_doubt > 0.5:
            # 너무 의심함 → baseline 낮춤
            self.baseline = max(self.BASELINE_MIN, self.baseline - 0.02)
        elif actual_outcome == 'bad' and predicted_doubt < 0.3:
            # 너무 안 의심함 → baseline 올림
            self.baseline = min(self.BASELINE_MAX, self.baseline + 0.03)
        # neutral 또는 일치하면 무변화
        return {
            'before': before,
            'after': self.baseline,
            'delta': self.baseline - before,
        }

    # ============= 5. 시간 변동 =============

    def update(self, dt: float):
        """
        매 tick: baseline DEFAULT로 천천히 회귀 (반감기 600초).
        영구 lock 방지.
        """
        self.time += dt
        if dt <= 0:
            return
        decay = 1.0 - 0.5 ** (dt / self.BASELINE_HALFLIFE)
        self.baseline = self.baseline + (self.DEFAULT_BASELINE - self.baseline) * decay
        self.baseline = float(np.clip(
            self.baseline, self.BASELINE_MIN, self.BASELINE_MAX))

    # ============= 진단 =============

    def get_state(self) -> Dict[str, Any]:
        return {
            'baseline': self.baseline,
            'baseline_with_hormone': self.get_baseline_doubt(),
            'eval_count': self.eval_count,
            'reject_count': self.reject_count,
            'time': self.time,
            'history_size': len(self.history),
            'last_doubt_level': (self.last_doubt['doubt_level']
                                 if self.last_doubt else None),
            'last_action': (self.last_doubt['recommended_action']
                            if self.last_doubt else None),
        }

    def __repr__(self):
        return (f"SelfDoubt(baseline={self.baseline:.2f}, "
                f"with_hormone={self.get_baseline_doubt():.2f}, "
                f"evals={self.eval_count}, rejects={self.reject_count})")
