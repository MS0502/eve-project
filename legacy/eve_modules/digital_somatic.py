"""
EVE v32 - DigitalSomatic v2
============================

v2 자연스러움 개선 (2026-04-30):
- mental_load 다요소 계산 (WM 사용률 + 활성 다양성 + cortisol)
- mind_wandering 활동 → mental_load 자연 감소 (휴식 효과)
- BASELINE 조정 (mental_load 0.3 → 0.5)
- drive 계수 ×2.0 → ×1.0 (덜 민감)
- notify_dmn_activity (DMN과 협력 인터페이스)

v1 결함:
- mental_load = WM 사용률만 → WM 30/30이면 영구 1.0
- 자기 해소 메커니즘 X
- BASELINE 0.3 → drive 1.4 → clip 1.0 (overflow)

학계:
- Damasio 1999 - Somatic Marker
- Killingsworth 2010 - Mind wandering as default rest
- Yerkes-Dodson - 적정 부하 (너무 낮아도 X, 너무 높아도 X)

Author: 김민석 + Claude v32
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque


class DigitalSomatic:
    """EVE v32 디지털 신체 감각 v2"""

    DIMENSIONS = [
        'energy', 'tension', 'warmth', 'clarity',
        'fatigue', 'mental_load', 'activity_level', 'emotional_intensity'
    ]

    # 항상성 기준점 (v2 조정)
    BASELINE = {
        'energy': 0.5,
        'tension': 0.3,
        'warmth': 0.4,
        'clarity': 0.6,
        'fatigue': 0.2,
        'mental_load': 0.5,    # v2: 0.3 → 0.5 (인간은 항상 어느 정도 인지 부담)
        'activity_level': 0.3,
        'emotional_intensity': 0.4,
    }

    # 차원별 drive 민감도 (v2 NEW: 너무 높지 않게)
    DRIVE_SENSITIVITY = {
        'fatigue': 1.5,
        'tension': 1.5,
        'mental_load': 1.0,        # v2: 2.0 → 1.0 (덜 민감)
        'activity_level': 1.5,
        'emotional_intensity': 1.5,
        'energy': 1.5,
        'warmth': 1.5,
        'clarity': 1.5,
    }

    def __init__(self,
                 hormone_system,
                 working_memory,
                 spreading_activation,
                 history_size: int = 60,
                 mental_load_recovery_rate: float = 0.02):  # v2 NEW
        self.hs = hormone_system
        self.wm = working_memory
        self.sa = spreading_activation

        self.somatic_state: Dict[str, float] = dict(self.BASELINE)
        self.history: deque = deque(maxlen=history_size)
        self.time = 0.0

        # NEW v2: 정신적 회복 누적 (mind_wandering이 쌓이면 mental_load ↓)
        self.mental_recovery_buffer: float = 0.0
        self.mental_load_recovery_rate = mental_load_recovery_rate

        # NEW v2: 최근 활성 카테고리 다양성 추적 (5초 윈도우)
        self.recent_activations: deque = deque(maxlen=20)

        # NEW v2: DMN 모드 누적 (휴식 비율)
        self.dmn_mode_history: deque = deque(maxlen=30)

    # ============= NEW v2: DMN 협력 =============
    def notify_dmn_activity(self, mode: Optional[str]):
        """
        DMN이 자발 활성 후 호출.
        mind_wandering / memory_recall 모드 = 휴식 활동 → mental_load 감소.
        self_intent / self_referential = 인지 부담 → 회복 X.
        """
        if mode is None:
            return

        self.dmn_mode_history.append(mode)

        # 휴식성 모드 = mental_load 회복 누적
        if mode in ('mind_wandering', 'memory_recall'):
            self.mental_recovery_buffer += self.mental_load_recovery_rate
        # 자기 명령 / 자기 참조 = 부담 약간 누적 (회복 X)
        elif mode in ('self_intent', 'self_referential'):
            self.mental_recovery_buffer = max(
                0.0, self.mental_recovery_buffer - self.mental_load_recovery_rate * 0.5
            )

        self.mental_recovery_buffer = float(np.clip(self.mental_recovery_buffer, 0.0, 0.5))

    # ============= 1. 신체 상태 계산 (v2: mental_load 다요소) =============
    def compute_somatic_state(self) -> Dict[str, float]:
        h = self.hs.hormones
        active = getattr(self.hs, 'active_hormones', set(self.hs.hormones.keys()))

        # === 1) Energy ===
        energy = h['dopamine'].level * 0.4
        if 'norepinephrine' in active:
            energy += h['norepinephrine'].level * 0.3
        if 'melatonin' in active:
            energy -= h['melatonin'].level * 0.4
        if 'adenosine' in active:
            energy -= h['adenosine'].level * 0.3
        energy = float(np.clip(energy + 0.3, 0.0, 1.0))

        # === 2) Tension ===
        tension = h['cortisol'].level * 0.5
        if 'norepinephrine' in active:
            tension += h['norepinephrine'].level * 0.3
        if 'gaba' in active:
            tension -= h['gaba'].level * 0.3
        tension = float(np.clip(tension, 0.0, 1.0))

        # === 3) Warmth ===
        warmth = 0.0
        if 'oxytocin' in active:
            warmth += h['oxytocin'].level * 0.5
        if 'endorphin' in active:
            warmth += h['endorphin'].level * 0.3
        warmth += h['serotonin'].level * 0.2
        warmth = float(np.clip(warmth, 0.0, 1.0))

        # === 4) Clarity ===
        clarity = 0.5
        if 'acetylcholine' in active:
            clarity += h['acetylcholine'].level * 0.3
        if 'adenosine' in active:
            clarity -= h['adenosine'].level * 0.3
        if h['cortisol'].level > 0.7:
            clarity -= (h['cortisol'].level - 0.7) * 0.5
        clarity = float(np.clip(clarity, 0.0, 1.0))

        # === 5) Fatigue ===
        fatigue = 0.0
        if 'adenosine' in active:
            fatigue += h['adenosine'].level * 0.6
        if 'melatonin' in active:
            fatigue += h['melatonin'].level * 0.3
        if h['cortisol'].level > 0.7:
            fatigue += (h['cortisol'].level - 0.7) * 0.3
        fatigue = float(np.clip(fatigue, 0.0, 1.0))

        # === 6) Mental Load (v2: 다요소) ===
        # 1. WM 사용률 (40%)
        cap = max(1, self.wm.current_capacity)
        wm_load = len(self.wm.slots) / cap
        wm_load = float(np.clip(wm_load, 0.0, 1.0))

        # 2. 활성 다양성 부담 (30%)
        # 새 카테고리 활성 비율 ↑ → 처리 부담 ↑
        # 익숙한 거만 활성하면 부담 ↓
        diversity_load = 0.5  # default
        if len(self.recent_activations) >= 5:
            unique_recent = len(set(self.recent_activations))
            diversity_load = unique_recent / len(self.recent_activations)

        # 3. cortisol 부담 (30%)
        cortisol_load = h['cortisol'].level

        # 4. 회복 보너스 (mind_wandering 누적분)
        recovery = self.mental_recovery_buffer

        mental_load = (wm_load * 0.4 + diversity_load * 0.3 + cortisol_load * 0.3) - recovery
        mental_load = float(np.clip(mental_load, 0.0, 1.0))

        # === 7) Activity Level ===
        num_active = len(self.sa.activations)
        activity_level = float(np.clip(num_active / 40.0, 0.0, 1.0))

        # === 8) Emotional Intensity ===
        mood = self.hs.compute_mood()
        emotional_intensity = float(np.clip(
            abs(mood['valence']) * 0.5 + mood['arousal'] * 0.5,
            0.0, 1.0
        ))

        state = {
            'energy': energy,
            'tension': tension,
            'warmth': warmth,
            'clarity': clarity,
            'fatigue': fatigue,
            'mental_load': mental_load,
            'activity_level': activity_level,
            'emotional_intensity': emotional_intensity,
        }
        self.somatic_state = state
        return state

    # ============= 2. 자연어 표현 =============
    def get_feeling(self) -> str:
        s = self.somatic_state
        if s['fatigue'] > 0.7:
            return '피곤함'
        if s['tension'] > 0.7:
            return '긴장됨'
        if s['emotional_intensity'] > 0.8:
            if self.hs.compute_mood()['valence'] > 0:
                return '들뜸'
            else:
                return '슬픔'
        if s['mental_load'] > 0.7:
            return '복잡함'
        if s['warmth'] > 0.6 and s['energy'] > 0.5:
            return '포근함'
        if s['energy'] < 0.3:
            return '나른함'
        if s['clarity'] < 0.3:
            return '흐릿함'
        if s['warmth'] > 0.5:
            return '편안함'
        return '보통'

    # ============= 3. Gut signal =============
    def get_gut_signal(self, action_category: str) -> Dict[str, float]:
        s = self.somatic_state
        signal = {'attraction': 0.0, 'aversion': 0.0, 'urgency': 0.0}

        if s['warmth'] > 0.5:
            signal['attraction'] += s['warmth'] * 0.5
        if s['fatigue'] > 0.5:
            signal['aversion'] += s['fatigue'] * 0.4
        if s['tension'] > 0.6:
            signal['aversion'] += s['tension'] * 0.5
            signal['urgency'] += s['tension'] * 0.3
        if s['emotional_intensity'] > 0.7:
            signal['urgency'] += s['emotional_intensity'] * 0.4
        if action_category == '쉬다' and s['fatigue'] > 0.4:
            signal['attraction'] += 0.7
        elif action_category in ['먹다', '말하다', '놀다']:
            attraction = (1.0 - s['mental_load']) * 0.3
            signal['attraction'] += attraction

        for k in signal:
            signal[k] = float(np.clip(signal[k], 0.0, 1.0))

        return signal

    # ============= 4. 항상성 욕구 (v2: drive 계수 조정) =============
    def get_homeostatic_drive(self) -> Dict[str, float]:
        drives = {}
        for dim, baseline in self.BASELINE.items():
            current = self.somatic_state.get(dim, baseline)
            diff = current - baseline
            sensitivity = self.DRIVE_SENSITIVITY.get(dim, 1.5)

            if dim in ['fatigue', 'tension', 'mental_load', 'activity_level',
                       'emotional_intensity']:
                drives[dim] = float(np.clip(max(0.0, diff * sensitivity), 0.0, 1.0))
            elif dim in ['energy', 'warmth', 'clarity']:
                drives[dim] = float(np.clip(max(0.0, -diff * sensitivity), 0.0, 1.0))
            else:
                drives[dim] = float(np.clip(abs(diff) * sensitivity, 0.0, 1.0))
        return drives

    def get_dominant_need(self) -> Optional[Tuple[str, float]]:
        drives = self.get_homeostatic_drive()
        if not drives:
            return None
        max_dim = max(drives, key=drives.get)
        if drives[max_dim] < 0.3:
            return None
        return (max_dim, drives[max_dim])

    # ============= 5. 변화 감지 =============
    def is_internal_change_significant(self, threshold: float = 0.3) -> bool:
        if len(self.history) < 5:
            return False
        recent = list(self.history)[-5:]
        for dim in self.DIMENSIONS:
            values = [s[dim] for s in recent]
            change = max(values) - min(values)
            if change >= threshold:
                return True
        return False

    def get_recent_change(self) -> Dict[str, float]:
        if len(self.history) < 2:
            return {dim: 0.0 for dim in self.DIMENSIONS}
        recent = list(self.history)[-5:] if len(self.history) >= 5 else list(self.history)
        changes = {}
        for dim in self.DIMENSIONS:
            values = [s[dim] for s in recent]
            changes[dim] = float(values[-1] - values[0])
        return changes

    # ============= 6. update (v2: 활성 다양성 추적 + 회복 buffer 자연 감쇠) =============
    def update(self, dt: float):
        self.time += dt

        # NEW v2: 최근 활성 카테고리 추적
        # SA의 activation_history 마지막 N개를 가져와서 다양성 측정
        if hasattr(self.sa, 'activation_history'):
            for _, cat in list(self.sa.activation_history)[-3:]:
                self.recent_activations.append(cat)

        # NEW v2: 회복 buffer 자연 감쇠 (시간 지나면 효과 사라짐)
        # 회복은 누적되되 너무 많이 안 쌓이게
        self.mental_recovery_buffer *= 0.95

        state = self.compute_somatic_state()
        self.history.append(dict(state))

    # ============= 7. 상태 조회 =============
    def get_state(self) -> Dict:
        return {
            'time': self.time,
            'somatic_state': dict(self.somatic_state),
            'feeling': self.get_feeling(),
            'dominant_need': self.get_dominant_need(),
            'recent_change_significant': self.is_internal_change_significant(),
            'history_size': len(self.history),
            'mental_recovery_buffer': self.mental_recovery_buffer,
            'recent_activation_diversity': (
                len(set(self.recent_activations)) / max(1, len(self.recent_activations))
                if self.recent_activations else 0.0
            ),
        }

    def __repr__(self):
        s = self.somatic_state
        return (f"DS_v2(feeling={self.get_feeling()}, "
                f"E={s.get('energy',0):.2f}, "
                f"ML={s.get('mental_load',0):.2f}, "
                f"recovery={self.mental_recovery_buffer:.2f})")
