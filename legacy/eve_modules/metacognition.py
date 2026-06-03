"""
EVE v32 - B6: MetaCognition (자기 사고 관찰)
==============================================

"내가 왜 이렇게 생각하지?"

EVE의 메타 레벨 관찰. 다른 모듈들의 출력을 *읽기 전용*으로 관찰하며
사고/욕구/확신의 *원인*을 추적.

학계:
- Flavell 1979 — Metacognition (thinking about thinking)
- Frankfurt 1971 — Hierarchical Will (1차 vs 2차 욕구)
- Nelson & Narens 1990 — Meta-level vs Object-level
- Frith 2012 — Meta-awareness
- Fleming & Lau 2014 — Metacognitive bias
- Dehaene 2014 — Conscious access (글로벌 작업공간)

역할:
- 카테고리 활성의 *원인* 추적 (host driven? spread? spontaneous?)
- 1차 욕구의 endorse/override 결정 (Frankfurt 2차 욕구)
- Confidence calibration (SD vs AI vs Agency 일치성)
- 메타 관찰 history

EVE 절대 원칙 부합:
- 확률 X
- 결정론
- 호르몬 변조 (메타 인지도 세로토닌/cort 영향받음)
- 카테고리 단위
- 모듈 chain (HS+SA+WM+DS+SD+AI+DMN+EM 8개 read-only)
- read-only 관찰 (다른 모듈 수정 X)

핵심 함수 6개:
1. monitor_thought(category)     → 사고 원인 분석
2. endorse_or_override(need)     → Frankfurt 2차 욕구 결정
3. evaluate_confidence(action)   → 메타 calibration
4. meta_introspect()             → "지금 내 마음 상태"
5. record_observation(obs)       → history (deque)
6. tick(dt)                      → 시간 누적

Author: 김민석 + Claude v32 (B6)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


class MetaCognition:
    """EVE v32 B6 MetaCognition"""

    # 그룹별 주요 호르몬 (호르몬 driven 판단용)
    HORMONE_FOR_GROUP = {
        'reward':    ['dopamine', 'endorphin'],
        'social':    ['oxytocin'],
        'threat':    ['cortisol', 'norepinephrine'],
        'attention': ['norepinephrine', 'acetylcholine'],
        'memory':    ['acetylcholine', 'bdnf'],
        'rest':      ['adenosine', 'melatonin', 'gaba'],
        'curiosity': ['dopamine', 'norepinephrine'],
        'aversion':  ['cortisol'],
    }

    # 호르몬 high 임계 (이 이상이면 driving)
    HORMONE_HIGH_THRESHOLD = 0.6

    # WM focus → target 카테고리 spread 임계
    SPREAD_WEIGHT_THRESHOLD = 0.3

    # Frankfurt endorse 임계
    ENDORSE_SCORE_THRESHOLD = 0.4

    # confidence calibration: 모듈 간 차이 임계
    INCONSISTENCY_THRESHOLD = 0.5

    # 사고 원인 점수 (가중치)
    CAUSE_SCORES = {
        'self_intent':      0.95,  # DMN의 명시적 의도
        'external':         0.85,  # 외부 입력
        'hormone_driven':   0.80,  # 호르몬 그룹 활성
        'recall':           0.70,  # 일화 회상
        'spontaneous_dmn':  0.65,  # DMN 자발
        'spread':           0.55,  # graph spread
        'unknown':          0.10,
    }

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 digital_somatic=None,
                 self_doubt=None,
                 active_inference=None,
                 dmn=None,
                 episodic_memory=None,
                 agency=None,
                 history_size: int = 100):
        """
        Args:
            hormone_system, spreading_activation: 필수
            나머지 8개: 옵션 (None 안전)
        """
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.ds = digital_somatic
        self.sd = self_doubt
        self.ai = active_inference
        self.dmn = dmn
        self.em = episodic_memory
        self.ag = agency

        self.observations: deque = deque(maxlen=history_size)
        self.time = 0.0

        self.monitor_count = 0
        self.endorse_count = 0
        self.override_count = 0
        self.calibration_count = 0
        self.inconsistency_count = 0

    # ============= 1. monitor_thought =============
    def monitor_thought(self, category: str) -> Dict[str, Any]:
        """
        카테고리가 떠오른 원인 추적.

        가능한 원인 (점수 높은 순):
        - self_intent       (DMN current_mode + intent 매칭)
        - external          (WM source = external)
        - hormone_driven    (그룹 매칭 + 호르몬 ↑)
        - recall            (em.recall로 매칭되는 일화)
        - spontaneous_dmn   (DMN wandering_history 최근)
        - spread            (WM focus → graph weight)
        - unknown           (none above)

        Returns:
            {
                'category': str,
                'cause': str,           # 가장 강한 원인
                'evidence': str,        # 증거 텍스트
                'confidence': float,    # 메타 확신도
                'all_causes': list[(cause, evidence, score), ...]
            }
        """
        self.monitor_count += 1
        causes: List[Tuple[str, str, float]] = []

        # 1) DMN self_intent
        if self.dmn is not None:
            if getattr(self.dmn, 'current_mode', None) == 'self_intent':
                intent = getattr(self.dmn, 'current_intent', None)
                # NEED_TO_CATEGORIES 매핑 확인
                if intent and hasattr(self.dmn, 'NEED_TO_CATEGORIES'):
                    intent_cats = self.dmn.NEED_TO_CATEGORIES.get(intent, [])
                    if category in intent_cats:
                        causes.append((
                            'self_intent',
                            f"intent={intent}",
                            self.CAUSE_SCORES['self_intent'],
                        ))

        # 2) External (WM source)
        if self.wm is not None and hasattr(self.wm, 'slots'):
            slot = self.wm.slots.get(category)
            if slot is not None:
                src = getattr(slot, 'source', None)
                if src in ('external', 'dialogue'):
                    causes.append((
                        'external',
                        f"wm_source={src}",
                        self.CAUSE_SCORES['external'],
                    ))

        # 3) Hormone driven
        if self.hs is not None and hasattr(self.sa, 'CATEGORY_GROUPS'):
            group = self.sa.CATEGORY_GROUPS.get(category)
            if group:
                relevant_hormones = self.HORMONE_FOR_GROUP.get(group, [])
                high_horms = []
                for h_name in relevant_hormones:
                    h = self.hs.hormones.get(h_name)
                    if h is not None and h.level >= self.HORMONE_HIGH_THRESHOLD:
                        high_horms.append((h_name, h.level))
                if high_horms:
                    # 가장 높은 호르몬 1개로 evidence
                    h_name, h_lvl = max(high_horms, key=lambda x: x[1])
                    causes.append((
                        'hormone_driven',
                        f"group={group}, {h_name}={h_lvl:.2f}",
                        self.CAUSE_SCORES['hormone_driven'],
                    ))

        # 4) Episodic recall (em.recall로 직접 매칭)
        if self.em is not None and hasattr(self.em, 'recall'):
            try:
                eps = self.em.recall({category}, top_n=1)
                if eps and category in eps[0].categories:
                    causes.append((
                        'recall',
                        f"episode='{eps[0].summary[:30]}'",
                        self.CAUSE_SCORES['recall'],
                    ))
            except Exception:
                pass

        # 5) DMN spontaneous (wandering_history)
        if self.dmn is not None and getattr(self.dmn, 'wandering_history', None):
            recent = self.dmn.wandering_history[-5:]  # 최근 5개
            for entry in recent:
                # entry: (time, category, mode)
                if len(entry) >= 2 and entry[1] == category:
                    mode = entry[2] if len(entry) >= 3 else 'unknown'
                    if mode != 'self_intent':  # self_intent는 1번에서 처리
                        causes.append((
                            'spontaneous_dmn',
                            f"mode={mode}",
                            self.CAUSE_SCORES['spontaneous_dmn'],
                        ))
                    break

        # 6) WM focus spread
        if self.wm is not None and self.sa is not None:
            focus = self.wm.get_focus() if hasattr(self.wm, 'get_focus') else None
            if focus and focus != category and hasattr(self.sa, 'get_weight'):
                w = self.sa.get_weight(focus, category)
                if w >= self.SPREAD_WEIGHT_THRESHOLD:
                    causes.append((
                        'spread',
                        f"from focus={focus} (w={w:.2f})",
                        self.CAUSE_SCORES['spread'],
                    ))

        # 정렬: 점수 높은 순 (동점 시 cause 알파벳순으로 결정론)
        causes.sort(key=lambda x: (-x[2], x[0]))

        if causes:
            best = causes[0]
            return {
                'category': category,
                'cause': best[0],
                'evidence': best[1],
                'confidence': best[2],
                'all_causes': causes,
            }
        return {
            'category': category,
            'cause': 'unknown',
            'evidence': '',
            'confidence': self.CAUSE_SCORES['unknown'],
            'all_causes': [],
        }

    # ============= 2. endorse_or_override =============
    def endorse_or_override(self,
                           dominant_need: Optional[Tuple[str, float]] = None
                           ) -> Optional[Dict[str, Any]]:
        """
        Frankfurt 1971: 1차 욕구를 endorse(승인) / override(거부) 결정.

        예: dominant_need=('fatigue', 0.7)
            - endorse: "응, 쉬자"
            - override: "지금 안 쉴래"

        결정 요인:
        - 욕구 강도 (기본 점수)
        - 일주기 (밤이면 fatigue endorse, 낮이면 override)
        - 위협 카테고리 활성 (강하면 fatigue/warmth need override)
        - 호르몬 (cort↑ → 'fatigue' override 경향)

        Args:
            dominant_need: (need_name, strength) 또는 None
                None이면 self.ds.get_dominant_need() 사용

        Returns:
            {'need', 'strength', 'decision', 'score', 'reasons': list}
            또는 None (need 없으면)
        """
        if dominant_need is None and self.ds is not None:
            dominant_need = self.ds.get_dominant_need()
        if not dominant_need:
            return None

        need_name, strength = dominant_need
        score = float(strength)
        reasons: List[str] = [f"strength={strength:.2f}"]

        # 일주기
        sim_hour = getattr(self.hs, 'sim_hour', 12.0) if self.hs else 12.0

        if need_name == 'fatigue':
            # 밤(22-6시) → endorse
            if sim_hour >= 22 or sim_hour < 6:
                score += 0.30
                reasons.append(f"night({sim_hour:.0f}h) → +0.30")
            # 낮(9-18) → 약간 override 경향
            elif 9 <= sim_hour <= 18:
                score -= 0.20
                reasons.append(f"day({sim_hour:.0f}h) → -0.20")

        # 위협 활성 → fatigue/warmth override (강한 영향)
        if self.sa is not None and need_name in ('fatigue', 'warmth'):
            threat_active = 0.0
            for cat in ('위험', '아프다', '무섭다', '죽음'):
                threat_active += self.sa.activations.get(cat, 0.0)
            if threat_active > 0.3:
                # 위협은 휴식/사회 욕구를 강하게 억제 (생존 우선)
                score -= 0.60
                reasons.append(f"threat={threat_active:.2f} → -0.60")

        # cort 만성↑ → fatigue override (계속 긴장)
        if self.hs is not None and need_name == 'fatigue':
            cort = self.hs.hormones.get('cortisol')
            if cort and cort.level > 0.7:
                score -= 0.20
                reasons.append(f"cortisol={cort.level:.2f} → -0.20")

        # 결정
        if score >= self.ENDORSE_SCORE_THRESHOLD:
            decision = 'endorse'
            self.endorse_count += 1
        else:
            decision = 'override'
            self.override_count += 1

        result = {
            'need': need_name,
            'strength': strength,
            'decision': decision,
            'score': float(np.clip(score, -1.0, 2.0)),
            'reasons': reasons,
            'time': self.time,
        }
        return result

    # ============= 3. evaluate_confidence =============
    def evaluate_confidence(self,
                           action_record: Optional[Dict] = None
                           ) -> Dict[str, Any]:
        """
        Confidence calibration: SD doubt vs AI bias vs Agency confidence
        세 모듈의 시그널이 일치하는지.

        action_record: Agency.evaluate_action 결과 (없으면 last_action 사용)

        Returns:
            {
                'signals': {sd, ai, ag},     # 각 모듈 시그널 [-1, 1] 정규화
                'calibrated': bool,          # 일치 여부
                'inconsistency': float,      # max diff
                'meta_doubt': float,         # 메타 의심 (불일치 → 의심)
            }
        """
        self.calibration_count += 1

        # SD doubt → confidence 부호 반대 (high doubt → low confidence)
        sd_signal = 0.0
        if self.sd is not None and getattr(self.sd, 'last_doubt', None):
            doubt = self.sd.last_doubt.get('doubt_level', 0.0)
            sd_signal = float(np.clip(1.0 - 2.0 * doubt, -1.0, 1.0))

        # AI bias: 학습된 호르몬-outcome 매핑이 현재 상태에서 good 신호
        ai_signal = 0.0
        if self.ai is not None and hasattr(self.ai, 'get_hormone_signature'):
            try:
                sig_good = self.ai.get_hormone_signature('good')
                sig_bad = self.ai.get_hormone_signature('bad')
                # 두 시그너처의 모든 호르몬 키 union (한쪽만 있어도 평가)
                all_hormones = set(sig_good.keys()) | set(sig_bad.keys())
                bias = 0.0
                for h_name in all_hormones:
                    h = self.hs.hormones.get(h_name) if self.hs else None
                    if h is None:
                        continue
                    deviation = h.level - 0.5
                    bias += deviation * (sig_good.get(h_name, 0.0)
                                         - sig_bad.get(h_name, 0.0))
                ai_signal = float(np.clip(bias, -1.0, 1.0))
            except Exception:
                ai_signal = 0.0

        # Agency confidence: action_record에서
        ag_signal = 0.0
        if action_record is None and self.ag is not None and self.ag.actions:
            action_record = self.ag.actions[-1]
        if action_record:
            ag_signal = float(np.clip(action_record.get('confidence', 0.0),
                                     -1.0, 1.0))

        signals = {'sd': sd_signal, 'ai': ai_signal, 'ag': ag_signal}

        # 가장 큰 차이
        active_vals = [v for v in signals.values()]
        if len(active_vals) >= 2:
            inconsistency = float(max(active_vals) - min(active_vals))
        else:
            inconsistency = 0.0

        calibrated = inconsistency < self.INCONSISTENCY_THRESHOLD
        if not calibrated:
            self.inconsistency_count += 1

        # 메타 의심: 불일치 ↑ → 메타 의심 ↑
        meta_doubt = float(np.clip(inconsistency, 0.0, 1.0))

        return {
            'signals': signals,
            'calibrated': calibrated,
            'inconsistency': inconsistency,
            'meta_doubt': meta_doubt,
            'action': action_record.get('action') if action_record else None,
        }

    # ============= 4. meta_introspect =============
    def meta_introspect(self) -> Dict[str, Any]:
        """
        "지금 내 마음 상태" — 통합 메타 관찰.

        Returns:
            {
                'time', 'sim_hour',
                'focus', 'focus_cause' (monitor_thought of focus),
                'dominant_need', 'need_decision' (endorse/override),
                'confidence_state' (calibration),
                'recent_thoughts' (최근 wandering 원인 분석),
                'meta_doubt'
            }
        """
        result = {
            'time': self.time,
            'sim_hour': getattr(self.hs, 'sim_hour', None) if self.hs else None,
            'focus': None,
            'focus_cause': None,
            'dominant_need': None,
            'need_decision': None,
            'confidence_state': None,
            'recent_thoughts': [],
            'meta_doubt': 0.0,
        }

        # focus + cause
        if self.wm is not None:
            focus = self.wm.get_focus() if hasattr(self.wm, 'get_focus') else None
            result['focus'] = focus
            if focus:
                result['focus_cause'] = self.monitor_thought(focus)

        # dominant_need + decision
        if self.ds is not None:
            need = self.ds.get_dominant_need()
            result['dominant_need'] = need
            if need:
                result['need_decision'] = self.endorse_or_override(need)

        # confidence state
        if self.ag is not None and self.ag.actions:
            result['confidence_state'] = self.evaluate_confidence()
            result['meta_doubt'] = result['confidence_state']['meta_doubt']

        # 최근 자발 사고들 분석 (DMN wandering 최근 3개)
        if self.dmn is not None and getattr(self.dmn, 'wandering_history', None):
            recent = self.dmn.wandering_history[-3:]
            for entry in recent:
                cat = entry[1] if len(entry) >= 2 else None
                if cat:
                    cause = self.monitor_thought(cat)
                    result['recent_thoughts'].append({
                        'category': cat,
                        'cause': cause['cause'],
                        'evidence': cause['evidence'],
                    })

        return result

    # ============= 5. record_observation =============
    def record_observation(self, observation: Dict[str, Any]):
        """메타 관찰 history에 기록."""
        record = dict(observation)
        record['time'] = self.time
        self.observations.append(record)

    # ============= 6. tick =============
    def tick(self, dt: float):
        """시간 누적."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'monitor_count': self.monitor_count,
            'endorse_count': self.endorse_count,
            'override_count': self.override_count,
            'calibration_count': self.calibration_count,
            'inconsistency_count': self.inconsistency_count,
            'observations_recorded': len(self.observations),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"MetaCognition(t={s['time']:.1f}s, "
                f"monitors={s['monitor_count']}, "
                f"endorse/override={s['endorse_count']}/{s['override_count']})")
