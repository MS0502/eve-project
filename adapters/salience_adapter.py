"""
DynamicValue + SalienceAdapter — 라운드 37

GPT 분석 (eve3.md): 모든 상태는 spike + decay 통일.

DynamicValue: 시간 + 중요도를 표현하는 단위
SalienceAdapter: 카테고리/생각의 중요도 추적

학계:
- Ebbinghaus 1885: forgetting curve
- Hebb 1949: 신경 강화-약화
- Damasio Somatic Marker
- Saliency Map (Itti & Koch 2001) — 어디에 주의 갈지
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DynamicValue — 모든 흐르는 상태의 기본 단위
# =============================================================================

@dataclass
class DynamicValue:
    """
    spike + decay 구조의 시간-가변 값.
    
    사용법:
        v = DynamicValue(base=0.0, decay_per_sec=0.02)
        v.spike(0.5)            # 자극
        v.tick(dt=1.0)          # 시간 흐름
        v.value                 # 현재 값
    """
    value: float = 0.0
    base: float = 0.0           # baseline (장기 회귀점)
    decay_per_sec: float = 0.02  # 초당 약화 비율 (linear). 0 = 영원.
    floor: float = 0.0           # 절대 최저
    ceiling: float = 5.0         # 절대 최대 (폭발 방지)
    last_tick_ts: float = field(default_factory=time.time)

    def spike(self, amount: float):
        """순간 증가."""
        self.value = min(self.ceiling, self.value + amount)

    def tick(self, dt: Optional[float] = None):
        """시간 흐름 — base 방향으로 회귀 + decay."""
        if dt is None:
            now = time.time()
            dt = now - self.last_tick_ts
            self.last_tick_ts = now
        # base로 자연 회귀 (homeostasis)
        diff = self.value - self.base
        # 지수 감소
        decay_factor = math.exp(-self.decay_per_sec * dt)
        self.value = self.base + diff * decay_factor
        # 경계
        self.value = max(self.floor, min(self.ceiling, self.value))

    def reset(self):
        self.value = self.base
        self.last_tick_ts = time.time()


# =============================================================================
# SalienceAdapter — 카테고리/생각의 중요도 추적
# =============================================================================

class SalienceAdapter:
    """
    각 카테고리/생각의 salience (중요도) 관리.
    
    원칙:
    - 자극 받으면 spike (이벤트로 떠오른 것)
    - 시간 지나면 decay (자연스럽게 잊힘)
    - 감정 강도 높을 때 추가 spike (감정적 사건은 더 오래 남음)
    
    DMN이 다음 생각 고를 때 사용 가능.
    EM이 인출 우선순위에 사용 가능.
    """

    def __init__(self, engine=None,
                 default_decay: float = 0.05,    # 초당 5% 감소
                 spike_normal: float = 0.3,
                 spike_emotional: float = 0.5):
        self.engine = engine
        self.default_decay = default_decay
        self.spike_normal = spike_normal
        self.spike_emotional = spike_emotional
        # 카테고리 → DynamicValue
        self.salience: dict = {}
        # 통계
        self.spike_count = 0
        self.tick_count = 0

    def spike(self, key: str, amount: Optional[float] = None,
              emotional: bool = False):
        """카테고리에 자극."""
        if amount is None:
            amount = self.spike_emotional if emotional else self.spike_normal
        if key not in self.salience:
            self.salience[key] = DynamicValue(
                base=0.0,
                decay_per_sec=self.default_decay,
                floor=0.0,
                ceiling=5.0,
            )
        self.salience[key].spike(amount)
        self.spike_count += 1

    def tick(self, dt: Optional[float] = None):
        """모든 salience 시간 흐름."""
        for v in self.salience.values():
            v.tick(dt)
        self.tick_count += 1

    def get(self, key: str) -> float:
        """현재 salience 값."""
        if key not in self.salience:
            return 0.0
        return self.salience[key].value

    def top(self, n: int = 5) -> list:
        """가장 salient 한 카테고리 N개."""
        items = [(k, v.value) for k, v in self.salience.items()]
        items.sort(key=lambda x: -x[1])
        return items[:n]

    def cleanup(self, threshold: float = 0.05):
        """매우 낮은 salience 제거 (메모리 절약)."""
        to_remove = [k for k, v in self.salience.items() if v.value < threshold]
        for k in to_remove:
            del self.salience[k]
        return len(to_remove)

    def stats(self) -> dict:
        return {
            "tracked_keys": len(self.salience),
            "spike_count": self.spike_count,
            "tick_count": self.tick_count,
            "top_5": self.top(5),
        }
