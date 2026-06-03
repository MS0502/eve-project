"""
EVE v35 - E6: ContinualRehearsal (정체성 보존)
================================================

목표: 새 학습 시 기존 정체성/핵심 기억 망각 방지 (catastrophic forgetting 방지).

문제:
- E1 Corpus + E3 APC가 매일 새 가중치 추가 → 옛 정체성 카테고리 묻힘 위험
- SA.forget이 protected_categories로 *수동 보호*만 가능
- 진짜 필요한 것: *능동적 재활성* (REM 수면 + spaced repetition)

E6 해결:
- NarrativeSelf.identity_stability에서 정체성 카테고리 자동 추출
- 주기적으로 약하게 자발 활성 (rehearse)
- 그래프가 그 주변으로 수렴 (인지 안정성)
- BDNF↑ 시 더 강한 보존, Cort↑ 시 약한 (스트레스 = 망각 가속)

EVE 절대 원칙:
- 결정론적 (rehearse 순서 = stability ↓ 알파벳 ↑)
- SA.activate 호출만 (가중치 직접 조작 X)
- 호르몬 자연 변조

학계:
- Schapiro 2025 — Combinatorial Continual Learning
- Skinner 1953 — Spaced repetition
- McClelland 1995 — Complementary Learning Systems
- Stickgold 2005 — Sleep consolidation (REM rehearsal)
- BDNF synaptic strengthening (Lu 2014)

핵심 함수:
1. mark_protected(category)              — 보호 등록 (수동)
2. rehearse_now(strength)                — 즉시 1회 재활성
3. detect_drift(category)                — 가중치 변화 감지
4. tick(dt)                              — 자동 주기 rehearse
5. stats()                               — 통계

Author: 김민석 + Claude v35
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class ContinualRehearsal:
    """
    정체성/핵심 카테고리 능동적 재활성.

    의존성:
        - SpreadingActivation (필수)
        - HormoneSystem (선택, 호르몬 변조)
        - NarrativeSelf (선택, identity_stability에서 자동 추출)
        - WorkingMemory (선택, focus 안 빼앗기 위해 약하게만)

    설계:
        - tick마다 카운터 증가, interval마다 rehearse 발동
        - 보호 카테고리: identity_stability top + 수동 등록
        - rehearse = SA.activate(cat, strength=낮음) — 다른 활동 방해 X
        - drift detection: 카테고리 가중치 평균 변화 추적
    """

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 narrative_self=None,
                 working_memory=None,
                 rehearse_interval_ticks: int = 60,
                 base_rehearse_strength: float = 0.15,
                 max_rehearse_count: int = 5,
                 use_hormone_modulation: bool = True):
        """
        Args:
            spreading_activation: SA (필수)
            hormone_system: HS (선택)
            narrative_self: NS (선택, identity 자동 추출)
            working_memory: WM (선택)
            rehearse_interval_ticks: 매 N tick마다 rehearse (60 = 30초@dt=0.5)
            base_rehearse_strength: 재활성 강도 (낮게 — 일상 활동 방해 X)
            max_rehearse_count: 한 번에 재활성할 카테고리 수
            use_hormone_modulation: BDNF/Cort 변조 사용
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.ns = narrative_self
        self.wm = working_memory

        self.rehearse_interval_ticks = int(rehearse_interval_ticks)
        self.base_rehearse_strength = float(base_rehearse_strength)
        self.max_rehearse_count = int(max_rehearse_count)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 수동 보호 카테고리
        self.manual_protected: Set[str] = set()

        # drift 추적: 카테고리 → 평균 가중치 history
        self.weight_history: Dict[str, List[float]] = defaultdict(list)
        self.history_max_len = 10

        # 카운터
        self.tick_count = 0
        self.rehearse_count = 0           # rehearse 호출 횟수
        self.categories_rehearsed = 0     # 누적 재활성 카테고리 수

        # 마지막 rehearse 결과
        self.last_rehearsed: List[str] = []

    # ============= 1. 보호 등록 =============
    def mark_protected(self, category: str):
        """카테고리 수동 보호 등록"""
        if category:
            self.manual_protected.add(category)

    def unmark_protected(self, category: str):
        """수동 보호 해제"""
        self.manual_protected.discard(category)

    # ============= 2. 보호 카테고리 수집 =============
    def get_protected_categories(self, top_n: int = 5) -> List[str]:
        """
        보호할 카테고리 결정.

        결정론적 정렬:
        1. 수동 등록 manual_protected (모두)
        2. NS identity_stability top N (stability ↓ 알파벳 ↑)

        중복 제거 후 결정론적 순서로 반환.
        """
        result: List[str] = []
        seen: Set[str] = set()

        # 1) 수동 등록 (알파벳 순)
        for cat in sorted(self.manual_protected):
            if cat not in seen:
                result.append(cat)
                seen.add(cat)

        # 2) NS identity (stability ↓, 알파벳 ↑)
        if self.ns is not None and hasattr(self.ns, 'identity_stability'):
            ns_top = sorted(
                self.ns.identity_stability.items(),
                key=lambda x: (-x[1], x[0])
            )[:top_n]
            for cat, _ in ns_top:
                if cat not in seen:
                    result.append(cat)
                    seen.add(cat)

        return result

    # ============= 3. 즉시 rehearse =============
    def rehearse_now(self, strength: Optional[float] = None,
                    max_count: Optional[int] = None) -> List[str]:
        """
        보호 카테고리들을 즉시 재활성.

        Args:
            strength: 활성 강도 (None이면 base_rehearse_strength)
            max_count: 활성할 최대 카테고리 수

        Returns:
            재활성된 카테고리 list
        """
        strength = strength if strength is not None else self.base_rehearse_strength
        max_count = max_count if max_count is not None else self.max_rehearse_count

        # 호르몬 변조
        if self.use_hormone_modulation and self.hs is not None:
            bdnf = self.hs.hormones.get('bdnf')
            cort = self.hs.hormones.get('cortisol')
            # BDNF↑ → 더 강하게 보존
            if bdnf and bdnf.level > 0.5:
                strength *= (1.0 + bdnf.level * 0.5)
            # 만성 cortisol↑ → 보존 약화 (스트레스가 정체성 흔듦)
            if cort and cort.level > 0.7:
                strength *= 0.5

        # 보호 카테고리 수집 + 자르기
        cats = self.get_protected_categories(top_n=max_count)
        cats = cats[:max_count]

        # 재활성 (낮은 강도, source 표시)
        rehearsed: List[str] = []
        for cat in cats:
            if cat:
                self.sa.activate(cat, strength)
                rehearsed.append(cat)

        # WM에는 약하게 add (focus 빼앗지 않게)
        if self.wm is not None:
            for cat in rehearsed[:2]:  # 최대 2개만 WM에
                if hasattr(self.wm, 'add'):
                    self.wm.add(cat, salience=strength * 0.3, source='rehearsal')

        # drift 추적 — 가중치 평균 기록
        for cat in rehearsed:
            if cat in self.sa.neighbors:
                weights = [self.sa.get_weight(cat, n)
                          for n in self.sa.neighbors[cat]]
                avg = sum(weights) / max(1, len(weights))
                hist = self.weight_history[cat]
                hist.append(avg)
                if len(hist) > self.history_max_len:
                    self.weight_history[cat] = hist[-self.history_max_len:]

        # 호르몬: rehearse 활동 → 약한 ACh (학습 신호 — Hasselmo)
        if self.use_hormone_modulation and self.hs is not None and rehearsed:
            self.hs.hormones['acetylcholine'].stimulate(0.02)
            # BDNF 약하게 자극 (자기 강화 사이클)
            if 'bdnf' in self.hs.hormones:
                self.hs.hormones['bdnf'].stimulate(0.02)

        self.rehearse_count += 1
        self.categories_rehearsed += len(rehearsed)
        self.last_rehearsed = rehearsed

        return rehearsed

    # ============= 4. drift 감지 =============
    def detect_drift(self, category: str,
                    threshold: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        카테고리의 평균 가중치 변화 감지.

        Args:
            category: 검사할 카테고리
            threshold: 변화 임계 (절대값)

        Returns:
            {'drift': 'increase'/'decrease'/'stable', 'magnitude': float}
            또는 None (history 부족)
        """
        hist = self.weight_history.get(category, [])
        if len(hist) < 3:
            return None

        # 최근 vs 처음 비교
        recent = sum(hist[-3:]) / 3
        early = sum(hist[:3]) / 3
        delta = recent - early

        if abs(delta) < threshold:
            kind = 'stable'
        elif delta > 0:
            kind = 'increase'
        else:
            kind = 'decrease'

        return {
            'drift': kind,
            'magnitude': float(abs(delta)),
            'early_avg': float(early),
            'recent_avg': float(recent),
            'delta': float(delta),
        }

    # ============= 5. tick (자동) =============
    def tick(self, dt: float = 0.5) -> Optional[List[str]]:
        """
        매 tick 호출. interval마다 자동 rehearse.

        Returns:
            rehearse한 카테고리 list 또는 None
        """
        self.tick_count += 1

        # interval 도달 시 rehearse
        if self.tick_count % self.rehearse_interval_ticks == 0:
            return self.rehearse_now()

        return None

    # ============= 6. 통계 =============
    def stats(self) -> Dict[str, Any]:
        protected = self.get_protected_categories(top_n=10)
        return {
            'tick_count': self.tick_count,
            'rehearse_count': self.rehearse_count,
            'categories_rehearsed_total': self.categories_rehearsed,
            'manual_protected_size': len(self.manual_protected),
            'current_protected': protected,
            'last_rehearsed': list(self.last_rehearsed),
            'tracked_drift_categories': len(self.weight_history),
        }

    def __repr__(self):
        s = self.stats()
        return (f"ContinualRehearsal(rehearses={s['rehearse_count']}, "
                f"protected={len(s['current_protected'])}, "
                f"tracked={s['tracked_drift_categories']})")
