"""
EVE v32 - B5: Agency (행동 결정 + 자기 귀속)
==============================================

EVE의 행동 선택 + "내가 했다" 인지.

학계:
- AURA 2025 — Agency Evaluation
- Wegner 2002 — Apparent Mental Causation
- Haggard 2017 — Sense of Agency (intentional binding)
- Frith 2012 — Will and Awareness
- Frankfurt 1971 — Hierarchical Will (2차 욕구)
- Damasio 1999 — Somatic Marker Hypothesis (gut signal → 결정)

역할:
- 6개 모듈 출력 통합 → 행동 선택 (do / hesitate / skip)
- DMN 모드 기반 attribution (intentional / spontaneous / reactive)
- 행동 history 관리

핵심 알고리즘:

1. confidence (결정론, [-1, 1]):
       confidence = (
           + W_GUT     × (attraction - aversion)         [DS]
           - W_DOUBT   × doubt_level                     [SD]
           + W_AI      × ai_prediction_bias              [B4]
           + W_INTENT  × (1 if mode == self_intent else 0)
           - W_THREAT  × threat_category_activation      [SA]
           + W_FOCUS   × focus_relevance                 [WM]
       )

   if confidence > do_threshold       → 'do'
   elif confidence > hesitate_threshold → 'hesitate'
   else                                → 'skip'

2. attribution (결정론):
       mode == 'self_intent'             → 'intentional'
       mode ∈ {mind_wandering,
               memory_recall,
               self_referential}          → 'spontaneous'
       source == 'external'              → 'reactive'
       (default)                         → 'spontaneous'

핵심 함수:
1. evaluate_action(action, source)   → {choice, confidence, attribution, ...}
2. compute_confidence(action)        → float in [-1, 1] (내부 helper, 공개도 OK)
3. attribute_action(action, ...)     → str
4. record_action(action_record)      → 기록 + history
5. get_recent_actions(n)             → list
6. tick(dt)                          → 시간 + history 정리

의존성 (모두 옵션 — None 안전):
- hormone_system, spreading_activation : 필수 추천
- working_memory, digital_somatic, self_doubt, dmn, active_inference : 옵션

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론 (같은 입력 → 같은 결정)
- 호르몬 변조 (DS gut + SD doubt + AI weight 모두 호르몬에서 옴)
- 카테고리 단위 (action_category)
- 모듈 chain (6개 모듈 통합)

Author: 김민석 + Claude v32 (B5)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


class Agency:
    """EVE v32 B5 Agency"""

    # 행동 선택 임계
    DO_THRESHOLD = 0.30          # confidence > 0.30 → do
    HESITATE_THRESHOLD = -0.10   # > -0.10 → hesitate, else skip

    # 가중치 (합 ~1.5, signed)
    W_GUT = 0.40
    W_DOUBT = 0.35
    W_AI = 0.20
    W_INTENT = 0.15
    W_THREAT = 0.30
    W_FOCUS = 0.10

    # threat 그룹 카테고리 (SA가 같은 그룹 정의 갖고 있음)
    THREAT_CATEGORIES = {'위험', '아프다', '무섭다', '칼', '죽음', '공격'}

    # mode → attribution
    INTENTIONAL_MODES = {'self_intent'}
    SPONTANEOUS_MODES = {'mind_wandering', 'memory_recall', 'self_referential'}

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 digital_somatic=None,
                 self_doubt=None,
                 dmn=None,
                 active_inference=None,
                 history_size: int = 100):
        """
        Args:
            hormone_system: A1 (필수)
            spreading_activation: A2 (필수)
            working_memory, digital_somatic, self_doubt, dmn, active_inference: 옵션
            history_size: 행동 기록 deque maxlen
        """
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.ds = digital_somatic
        self.sd = self_doubt
        self.dmn = dmn
        self.ai = active_inference

        # 행동 history (deque — lock 방지)
        self.actions: deque = deque(maxlen=history_size)
        self.time = 0.0

        # 통계
        self.eval_count = 0
        self.do_count = 0
        self.hesitate_count = 0
        self.skip_count = 0
        self.intentional_count = 0
        self.spontaneous_count = 0
        self.reactive_count = 0

    # ============= 1. compute_confidence =============
    def compute_confidence(self, action_category: str) -> Dict[str, Any]:
        """
        행동에 대한 confidence 계산.
        결정론적, [-1, 1] 범위.

        Returns:
            {'confidence': float, 'components': {...}}
        """
        components = {
            'gut_attract': 0.0,
            'gut_averse': 0.0,
            'doubt': 0.0,
            'ai_bias': 0.0,
            'intent_bonus': 0.0,
            'threat_penalty': 0.0,
            'focus_relevance': 0.0,
        }

        # 1) Gut signal (DS)
        if self.ds is not None and hasattr(self.ds, 'get_gut_signal'):
            gut = self.ds.get_gut_signal(action_category)
            components['gut_attract'] = float(gut.get('attraction', 0.0))
            components['gut_averse'] = float(gut.get('aversion', 0.0))

        # 2) Doubt (SD)
        if self.sd is not None and getattr(self.sd, 'last_doubt', None):
            components['doubt'] = float(self.sd.last_doubt.get('doubt_level', 0.0))

        # 3) AI prediction bias (B4)
        # 학습된 호르몬-outcome 매핑이 현재 호르몬 상태에 대해
        # good/bad 어느쪽 신호가 강한지 → bias
        if self.ai is not None and hasattr(self.ai, 'get_hormone_signature'):
            try:
                sig_good = self.ai.get_hormone_signature('good')
                sig_bad = self.ai.get_hormone_signature('bad')
                # 두 시그너처의 모든 호르몬 키 union
                all_hormones = set(sig_good.keys()) | set(sig_bad.keys())
                bias = 0.0
                if hasattr(self.hs, 'hormones'):
                    for h_name in all_hormones:
                        h = self.hs.hormones.get(h_name)
                        if h is None:
                            continue
                        deviation = h.level - 0.5  # baseline 0.5 가정
                        bias += deviation * (sig_good.get(h_name, 0.0)
                                             - sig_bad.get(h_name, 0.0))
                components['ai_bias'] = float(np.clip(bias, -1.0, 1.0))
            except Exception:
                components['ai_bias'] = 0.0

        # 4) Intent bonus (DMN self_intent)
        if self.dmn is not None:
            mode = getattr(self.dmn, 'current_mode', None)
            if mode in self.INTENTIONAL_MODES:
                components['intent_bonus'] = 1.0

        # 5) Threat penalty (SA threat 그룹 활성도)
        threat_act = 0.0
        if self.sa is not None and hasattr(self.sa, 'activations'):
            for cat in self.THREAT_CATEGORIES:
                threat_act += self.sa.activations.get(cat, 0.0)
            threat_act = min(1.0, threat_act)
        components['threat_penalty'] = threat_act

        # 6) Focus relevance (WM focus가 action_category와 관련 있나)
        if self.wm is not None and hasattr(self.wm, 'get_focus'):
            focus = self.wm.get_focus()
            if focus == action_category:
                components['focus_relevance'] = 1.0
            elif focus and self.sa is not None:
                # focus와 action 사이 graph weight
                if hasattr(self.sa, 'get_weight'):
                    w = self.sa.get_weight(focus, action_category)
                    components['focus_relevance'] = float(w)

        # ============= 통합 =============
        confidence = (
            self.W_GUT * (components['gut_attract'] - components['gut_averse'])
            - self.W_DOUBT * components['doubt']
            + self.W_AI * components['ai_bias']
            + self.W_INTENT * components['intent_bonus']
            - self.W_THREAT * components['threat_penalty']
            + self.W_FOCUS * components['focus_relevance']
        )
        confidence = float(np.clip(confidence, -1.0, 1.0))

        return {
            'confidence': confidence,
            'components': components,
        }

    # ============= 2. attribute_action =============
    def attribute_action(self,
                        action_category: str,
                        source: str = 'unknown',
                        mode_at_time: Optional[str] = None) -> str:
        """
        행동을 누가/어떻게 시작했는지 분류.

        Returns: 'intentional' / 'spontaneous' / 'reactive'

        - source='external' (사용자 명령 등) → reactive
        - mode == self_intent → intentional
        - mode ∈ DMN 자발 모드들 → spontaneous
        - default → spontaneous
        """
        if source == 'external':
            return 'reactive'

        mode = mode_at_time
        if mode is None and self.dmn is not None:
            mode = getattr(self.dmn, 'current_mode', None)

        if mode in self.INTENTIONAL_MODES:
            return 'intentional'
        if mode in self.SPONTANEOUS_MODES:
            return 'spontaneous'
        return 'spontaneous'

    # ============= 3. evaluate_action =============
    def evaluate_action(self,
                       action_category: str,
                       source: str = 'unknown') -> Dict[str, Any]:
        """
        행동 평가의 통합 인터페이스.

        Args:
            action_category: 평가할 행동 카테고리 (예: '대화', '쉬다')
            source: 'external' / 'spontaneous' / 'self_intent' / 'unknown'

        Returns:
            {
                'action': str,
                'choice': 'do' | 'hesitate' | 'skip',
                'confidence': float in [-1, 1],
                'components': dict (compute_confidence detail),
                'attribution': 'intentional' | 'spontaneous' | 'reactive',
                'time': float,
                'source': str,
            }
        """
        self.eval_count += 1

        # confidence
        conf_result = self.compute_confidence(action_category)
        confidence = conf_result['confidence']

        # choice
        if confidence > self.DO_THRESHOLD:
            choice = 'do'
            self.do_count += 1
        elif confidence > self.HESITATE_THRESHOLD:
            choice = 'hesitate'
            self.hesitate_count += 1
        else:
            choice = 'skip'
            self.skip_count += 1

        # attribution
        attribution = self.attribute_action(action_category, source=source)
        if attribution == 'intentional':
            self.intentional_count += 1
        elif attribution == 'reactive':
            self.reactive_count += 1
        else:
            self.spontaneous_count += 1

        result = {
            'action': action_category,
            'choice': choice,
            'confidence': confidence,
            'components': conf_result['components'],
            'attribution': attribution,
            'time': self.time,
            'source': source,
        }
        return result

    # ============= 4. record_action =============
    def record_action(self,
                     action_category: str,
                     choice: str,
                     attribution: str,
                     confidence: float = 0.0,
                     outcome: Optional[str] = None,
                     extra: Optional[Dict] = None) -> Dict[str, Any]:
        """
        행동 기록을 history에 추가.

        Args:
            outcome: 'good'/'bad'/'neutral'/None (None은 미평가)
            extra: 추가 정보

        Returns: 기록된 dict
        """
        record = {
            'action': action_category,
            'choice': choice,
            'attribution': attribution,
            'confidence': float(confidence),
            'outcome': outcome,
            'time': self.time,
        }
        if extra:
            record['extra'] = dict(extra)
        self.actions.append(record)
        return record

    # ============= 5. get_recent_actions =============
    def get_recent_actions(self, n: int = 10,
                          attribution: Optional[str] = None,
                          choice: Optional[str] = None) -> List[Dict]:
        """
        최근 N개 행동 (역순). 옵션 필터.
        """
        items = list(self.actions)
        if attribution is not None:
            items = [a for a in items if a.get('attribution') == attribution]
        if choice is not None:
            items = [a for a in items if a.get('choice') == choice]
        return items[-n:][::-1]  # 최신순

    # ============= 6. tick =============
    def tick(self, dt: float):
        """시간 누적. dt < 0이면 무시."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'eval_count': self.eval_count,
            'do_count': self.do_count,
            'hesitate_count': self.hesitate_count,
            'skip_count': self.skip_count,
            'intentional_count': self.intentional_count,
            'spontaneous_count': self.spontaneous_count,
            'reactive_count': self.reactive_count,
            'actions_recorded': len(self.actions),
            'last_action': dict(self.actions[-1]) if self.actions else None,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Agency(t={s['time']:.1f}s, evals={s['eval_count']}, "
                f"do/hes/skip={s['do_count']}/{s['hesitate_count']}/{s['skip_count']})")
