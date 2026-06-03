"""
EVE v32 - WorkingMemory + GNW Broadcast (v2.1 lock 풀기)
=========================================================

WorkingMemory: 30+ 슬롯 (인간 7±2 무시, AGI 환영)
GNW (Global Neuronal Workspace): focus 카테고리가 모든 모듈 가중치 변조

v2 → v2.1 변경점 (영구 슬롯 lock 풀기):
1. decay_factor floor: max(0.3, ...) → access_count 무한 누적해도 최소 decay 보장
2. add() saturation-aware: salience > 0.7에서 둔감화 (호르몬 stimulate와 동일 패턴)
3. access_count 시간 감쇠: 옛날 access는 잊혀야 함 (생물학적 자연스러움)
4. salience floor 0.05 → 0.10 상향 (조기 evict 활성화)

핵심 원칙:
- 카테고리 단위 작동 (뉴런 X)
- 결정론적 (random/softmax X)
- 호르몬 변조 받음:
    NE ↑ → 특정 슬롯 강화 (Yerkes-Dodson)
    Cortisol 만성 ↑ → 용량 축소
- SpreadingActivation 위에 얹는 레이어

학계:
- Baddeley 1992 - Working Memory
- Baars 1988 / Dehaene 2014 - Global Workspace
- Cowan 2001 - Focus of Attention
- Brown 1958 - 시간 기반 망각 (decay theory)

Author: 김민석 + Claude v32.1
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class WMSlot:
    """WM 슬롯 (카테고리 + 메타데이터)

    v2.1: access_count를 float로 → 시간 감쇠 적용 가능
    (이전엔 int round로 인해 50.8 → 51로 복원되어 감쇠 안 됨)
    """
    category: str
    salience: float          # 0-1, attention weight
    entered_at: float        # 진입 시간
    last_accessed: float     # 마지막 접근
    access_count: float = 0.0  # v2.1: float (시간 감쇠 적용)
    source: str = "external" # external, spreading, recall, dmn

    def access(self, time: float):
        self.last_accessed = time
        self.access_count += 1.0


class WorkingMemory:
    """
    EVE v32 WorkingMemory + GNW

    내부 상태:
        slots: {category: WMSlot}
        max_capacity: 기본 30 (호르몬으로 변조)
        listeners: GNW broadcast 받을 모듈들
    """

    def __init__(self,
                 base_capacity: int = 30,
                 base_decay: float = 0.05,
                 focus_threshold: float = 0.4):
        """
        Args:
            base_capacity: 기본 슬롯 용량 (호르몬으로 변조)
            base_decay: 시간당 salience 감쇠
            focus_threshold: focus로 인정될 최소 salience
        """
        self.base_capacity = base_capacity
        self.current_capacity = base_capacity
        self.base_decay = base_decay
        self.focus_threshold = focus_threshold

        self.slots: Dict[str, WMSlot] = {}
        self.time = 0.0

        # GNW listeners
        self.listeners: List[Tuple[str, Callable]] = []

        # 호르몬 상태 캐시 (broadcast 시 사용)
        self.hormone_state: Dict[str, float] = {}
        self.attention_boost: float = 1.0  # NE 영향

    # ============= 1. 슬롯 관리 =============
    def add(self, category: str, salience: float = 0.5,
            source: str = "external") -> bool:
        """
        WM에 카테고리 추가.
        용량 초과 시: salience 가장 낮은 거 제거.

        v2.1: 기존 슬롯 강화 시 saturation-aware (수용체 다운레귤레이션 패턴)

        Returns: 성공 여부
        """
        if not category:
            return False

        if category in self.slots:
            # v2.1: 기존 슬롯 강화 - saturation 시 둔감화 (수용체 다운레귤레이션)
            slot = self.slots[category]
            if slot.salience > 0.5:
                # 0.5→0.9에서 sat_factor 1.0→0.0 (0.9 이상에선 강화 차단)
                sat_factor = max(0.0, 1.0 - (slot.salience - 0.5) / 0.4)
            else:
                sat_factor = 1.0
            increment = salience * 0.5 * sat_factor
            slot.salience = float(np.clip(slot.salience + increment, 0.0, 1.0))
            slot.access(self.time)
            return True

        # 용량 체크
        if len(self.slots) >= self.current_capacity:
            self._evict_lowest()

        self.slots[category] = WMSlot(
            category=category,
            salience=float(np.clip(salience, 0.0, 1.0)),
            entered_at=self.time,
            last_accessed=self.time,
            access_count=1,
            source=source,
        )
        return True

    def remove(self, category: str):
        if category in self.slots:
            del self.slots[category]

    def _evict_lowest(self):
        """salience 가장 낮은 슬롯 제거"""
        if not self.slots:
            return
        lowest = min(self.slots.items(), key=lambda x: x[1].salience)
        del self.slots[lowest[0]]

    def access(self, category: str) -> bool:
        """슬롯 접근 (last_accessed 갱신)"""
        if category not in self.slots:
            return False
        self.slots[category].access(self.time)
        # 접근 시 약간 강화
        self.slots[category].salience = float(
            np.clip(self.slots[category].salience + 0.05, 0.0, 1.0)
        )
        return True

    # ============= 2. SpreadingActivation 자동 입력 =============
    def update_from_activation(self, spreading_activation,
                              salience_factor: float = 0.8):
        """
        SpreadingActivation에서 활성 카테고리 자동으로 WM에 입력.

        Args:
            spreading_activation: SpreadingActivation 인스턴스
            salience_factor: 활성도 → salience 변환 계수
        """
        active_set = spreading_activation.get_active()
        for cat in active_set:
            level = spreading_activation.get_activation_level(cat)
            self.add(cat, salience=level * salience_factor, source="spreading")

    # ============= 3. Focus (가장 활성 높음) =============
    def get_focus(self) -> Optional[str]:
        """단일 focus 카테고리"""
        if not self.slots:
            return None
        focus_slot = max(self.slots.values(), key=lambda s: s.salience)
        if focus_slot.salience < self.focus_threshold:
            return None
        return focus_slot.category

    def get_focus_set(self, n: int = 5) -> List[Tuple[str, float]]:
        """top N salience 카테고리"""
        items = [(s.category, s.salience) for s in self.slots.values()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def get_all_active(self) -> Set[str]:
        """현재 WM에 있는 모든 카테고리"""
        return set(self.slots.keys())

    # ============= 4. GNW Broadcast ★★★ =============
    def register_listener(self, name: str, callback: Callable[[str, float, Dict], None]):
        """
        GNW broadcast 받을 모듈 등록.

        Args:
            name: 모듈 이름
            callback: focus_category, focus_salience, context를 받는 함수
                     context = {'all_active': set, 'hormone_state': dict, 'time': float}
        """
        self.listeners.append((name, callback))

    def unregister_listener(self, name: str):
        self.listeners = [(n, c) for n, c in self.listeners if n != name]

    def broadcast(self) -> Optional[Dict]:
        """
        GNW broadcast: focus를 모든 listener에게 전파.

        Dehaene 2014: 한 카테고리가 의식 표면에 떠오르면, 다른 모듈들은
        그 카테고리에 맞춰 자기 가중치를 조정한다 (selection-broadcast cycle).

        Returns: broadcast 정보 dict (없으면 None)
        """
        focus = self.get_focus()
        if focus is None:
            return None

        focus_salience = self.slots[focus].salience
        context = {
            'all_active': self.get_all_active(),
            'hormone_state': dict(self.hormone_state),
            'time': self.time,
            'attention_boost': self.attention_boost,
        }

        # 모든 listener에게 broadcast
        for name, callback in self.listeners:
            try:
                callback(focus, focus_salience, context)
            except Exception as e:
                # listener 오류는 broadcast 막지 않음
                pass

        return {
            'focus': focus,
            'salience': focus_salience,
            'num_listeners': len(self.listeners),
            'all_active': context['all_active'],
        }

    # ============= 5. 호르몬 변조 (A1 연결) =============
    def apply_hormone_state(self, hormone_state: Dict[str, float]):
        """
        호르몬 상태 받음 → WM 용량 + attention boost 변조.

        - NE ↑ → attention_boost ↑ (focus 강화)
        - Cortisol 만성 ↑ → 용량 ↓
        - 멜라토닌 ↑ → 용량 ↓ (수면)
        """
        self.hormone_state = dict(hormone_state)

        # NE → attention boost (Yerkes-Dodson 단순화)
        ne = hormone_state.get('norepinephrine', 0.3)
        # 0.3 baseline → 1.0, 1.0 max → 1.5
        self.attention_boost = float(np.clip(0.7 + ne * 1.0, 0.5, 1.5))

        # Cortisol 만성 ↑ → 용량 ↓
        cort = hormone_state.get('cortisol', 0.3)
        if cort > 0.7:
            # 0.7 → 30, 1.0 → 18
            shrink = (cort - 0.7) * 40  # 최대 12 감소
            self.current_capacity = max(10, int(self.base_capacity - shrink))
        else:
            self.current_capacity = self.base_capacity

        # 멜라토닌 → 용량 ↓ (수면)
        mel = hormone_state.get('melatonin', 0.1)
        if mel > 0.5:
            self.current_capacity = max(10, int(self.current_capacity * (1.0 - (mel - 0.5))))

        # 용량 초과 시 정리
        while len(self.slots) > self.current_capacity:
            self._evict_lowest()

        # NE 부스트로 focus salience 임시 강화
        if ne > 0.6:
            focus = self.get_focus()
            if focus and focus in self.slots:
                self.slots[focus].salience = float(
                    np.clip(self.slots[focus].salience * self.attention_boost, 0.0, 1.0)
                )

    # ============= 6. 시간 감쇠 (v2.1: lock 풀기) =============
    def decay(self, dt: float):
        """
        시간 기반 salience 감쇠.
        access_count 높은 슬롯은 천천히 감쇠 - 단, floor 있음 (v2.1).

        v2.1 변경:
        - access_count float 시간 감쇠 (반감기 ~30초)
          * Brown 1958 - decay theory (rehearsal 없으면 trace 약화)
          * int round 문제 해결: float 직접 감쇠
        - decay_factor floor: max(0.3, ...) → access_count 누적해도 최소 30% decay
        - access_count 영향력 약화 (* 0.1 → * 0.05): 영구 lock 방지
        - salience floor 0.05 → 0.10 (조기 evict)
        """
        self.time += dt
        alpha = 1.0 - np.exp(-self.base_decay * dt)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # access_count 시간 감쇠 (반감기 ~30초)
        # exp(-dt / 30) - rehearsal 없으면 trace 약화
        access_decay = float(np.exp(-dt / 30.0))

        to_remove = []
        for cat, slot in self.slots.items():
            # access_count float 직접 감쇠 (1.0 floor: 한 번이라도 access 됐으면 1)
            if slot.access_count > 1.0:
                slot.access_count = max(1.0, (slot.access_count - 1.0) * access_decay + 1.0)

            # decay_factor에 floor + access 영향력 약화 - 영구 lock 방지
            # floor 0.5: access_count 누적해도 최소 50% decay → 90초 내 evict 보장
            decay_factor = max(0.5, 1.0 / (1.0 + slot.access_count * 0.05))
            slot.salience = slot.salience * (1.0 - alpha * decay_factor)
            # salience floor 0.05 → 0.10 (조기 evict)
            if slot.salience < 0.10:
                to_remove.append(cat)

        for cat in to_remove:
            del self.slots[cat]

    # ============= 7. 상태 조회 =============
    def get_state(self) -> Dict:
        return {
            'time': self.time,
            'num_slots': len(self.slots),
            'capacity': self.current_capacity,
            'base_capacity': self.base_capacity,
            'focus': self.get_focus(),
            'attention_boost': self.attention_boost,
            'num_listeners': len(self.listeners),
            'top_5': self.get_focus_set(5),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"WorkingMemory(slots={s['num_slots']}/{s['capacity']}, "
                f"focus={s['focus']}, listeners={s['num_listeners']})")
