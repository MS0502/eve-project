"""
EVE v35 - F2: AllostaticLearning (호르몬 칵테일 적응 학습)
==========================================================

목표: 반복 경험 → 새 호르몬 칵테일 자동 발견 + 학습.

기존 한계:
- HormoneSystem: 13 hardcode 칵테일 (eat, sleep, stress 등)
- 새 상황에 맞는 패턴 X
- 사용자 고유 컨텍스트 학습 X (예: '민석과의 코딩 세션' → 고유 칵테일)

F2 해결:
- 트리거-반응 페어 관찰
- 자주 동시 출현하는 (카테고리 → 호르몬 변화) 패턴 학습
- 임계 도달 시 새 cocktail 자동 등록
- 다음 같은 카테고리 트리거 시 학습된 칵테일 적용

학계:
- McEwen 2000 — Allostasis: 안정 점수가 변하는 것
- Sterling 2012 — Allostatic Theory of Adaptation
- Schulkin 2003 — Adaptation and Allostasis
- Sapolsky 2005 — Glucocorticoids and Plasticity

핵심 데이터:
- AllostaticPattern: trigger_cats + hormone_response + evidence_count
- patterns: Dict[pattern_id, AllostaticPattern]
- pending_observations: deque (최근 관찰 페어, 패턴 추출 대기)

EVE 절대 원칙:
- 결정론적 (sorted, no random)
- read-only on 호르몬 baseline (학습된 *추가* 변조만)
- 카테고리 의미 단위 유지
- hardcode 칵테일과 공존 (학습된 칵테일은 **augmentation**)

핵심 함수:
1. observe_response(trigger_cats, hs_state)  → 호르몬 반응 관찰
2. extract_patterns()                         → 반복 페어 → 패턴 학습
3. get_learned_cocktail(category)             → 학습된 칵테일 (있으면)
4. apply_learned(category)                    → 호르몬에 적용
5. tick(dt)                                   → 시간 감쇠 + 자동 추출
6. introspect_patterns()                      → 학습된 패턴들 보기

Author: 김민석 + Claude v35
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


@dataclass
class AllostaticPattern:
    """학습된 호르몬 칵테일 패턴"""
    id: str
    trigger_cats: Set[str]                      # 트리거 카테고리들
    hormone_changes: Dict[str, float]           # 호르몬 → delta
    evidence_count: int = 1                     # 반복 횟수
    strength: float = 0.5                       # 학습 강도
    last_observed: float = 0.0
    last_applied: float = 0.0
    created_at: float = 0.0
    apply_count: int = 0


class AllostaticLearning:
    """
    호르몬 칵테일 적응 학습.

    의존:
        hormone_system: HS (필수)
        spreading_activation: SA (필수)
        working_memory: WM (선택)

    플로우:
        매 tick:
            1. 현재 활성 카테고리 + 호르몬 상태 스냅샷
            2. 이전 스냅샷과 차이 계산 (delta)
            3. delta가 의미 있으면 pending_observations에 추가
            4. 패턴 추출 (매 N tick)

        패턴 발견:
            같은 trigger_cats가 N번 반복 + 비슷한 hormone delta → 패턴 학습

        적용:
            현재 trigger 매치 → 학습된 칵테일을 호르몬에 작은 영향으로 추가
    """

    # 의미 있는 호르몬 변화 임계
    SIGNIFICANT_DELTA = 0.05  # ±0.05 이상이면 기록

    # 핵심 호르몬만 추적 (모든 26개 추적은 노이즈)
    TRACKED_HORMONES = [
        'dopamine', 'cortisol', 'oxytocin', 'serotonin',
        'norepinephrine', 'acetylcholine', 'endorphin', 'gaba',
        'melatonin', 'adenosine',
    ]

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 internalize_threshold: int = 3,  # 3번 반복하면 패턴 학습
                 pattern_decay_per_day: float = 0.05,
                 max_patterns: int = 50,
                 extract_interval_ticks: int = 60):
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.internalize_threshold = int(internalize_threshold)
        self.pattern_decay_per_day = float(pattern_decay_per_day)
        self.max_patterns = int(max_patterns)
        self.extract_interval_ticks = int(extract_interval_ticks)

        # 핵심 상태
        self.patterns: Dict[str, AllostaticPattern] = {}
        self.cat_to_patterns: Dict[str, Set[str]] = defaultdict(set)

        # 관찰 버퍼 (제한 크기, 메모리 관리)
        self.pending_observations: deque = deque(maxlen=200)

        # 이전 호르몬 스냅샷 (delta 계산)
        self._prev_hormone_snapshot: Optional[Dict[str, float]] = None
        self._prev_active_cats: Optional[Set[str]] = None

        # 시간/통계
        self.time = 0.0
        self.tick_count = 0
        self.observe_count = 0
        self.extract_count = 0
        self.learned_count = 0
        self.apply_count = 0

    # ============= 1. 호르몬 반응 관찰 =============
    def observe_response(self,
                        trigger_cats: Set[str],
                        prev_state: Dict[str, float],
                        curr_state: Dict[str, float]):
        """
        트리거 카테고리 + 호르몬 변화 (delta) 관찰.

        Args:
            trigger_cats: 현재 활성 카테고리들
            prev_state: 이전 호르몬 levels {name: level}
            curr_state: 현재 호르몬 levels
        """
        if not trigger_cats:
            return

        # delta 계산
        deltas = {}
        for h_name in self.TRACKED_HORMONES:
            prev = prev_state.get(h_name, 0.0)
            curr = curr_state.get(h_name, 0.0)
            d = curr - prev
            if abs(d) >= self.SIGNIFICANT_DELTA:
                deltas[h_name] = round(d, 3)

        if not deltas:
            return  # 의미 있는 변화 없음

        # pending에 추가
        observation = {
            'time': self.time,
            'trigger': frozenset(trigger_cats),
            'deltas': deltas,
        }
        self.pending_observations.append(observation)
        self.observe_count += 1

    # ============= 2. 패턴 추출 =============
    def extract_patterns(self):
        """
        pending_observations에서 반복 패턴 추출.

        같은 trigger가 internalize_threshold번 이상 + 비슷한 delta → 학습.
        """
        if len(self.pending_observations) < self.internalize_threshold:
            return

        # trigger별 그룹화
        by_trigger: Dict[frozenset, List[Dict]] = defaultdict(list)
        for obs in self.pending_observations:
            by_trigger[obs['trigger']].append(obs)

        for trigger_set, obs_list in by_trigger.items():
            if len(obs_list) < self.internalize_threshold:
                continue

            # 호르몬별 평균 delta
            hormone_deltas: Dict[str, List[float]] = defaultdict(list)
            for obs in obs_list:
                for h, d in obs['deltas'].items():
                    hormone_deltas[h].append(d)

            avg_deltas = {}
            for h, deltas in hormone_deltas.items():
                # 같은 부호 비율이 높을 때만 신뢰 (60% 이상)
                pos = sum(1 for d in deltas if d > 0)
                neg = sum(1 for d in deltas if d < 0)
                consistency = max(pos, neg) / len(deltas)
                if consistency >= 0.6 and len(deltas) >= self.internalize_threshold:
                    # 평균 delta 저장
                    avg_deltas[h] = round(sum(deltas) / len(deltas), 3)

            if not avg_deltas:
                continue

            # 패턴 등록 또는 강화
            pattern_id = self._make_pattern_id(trigger_set)

            if pattern_id in self.patterns:
                # 강화
                p = self.patterns[pattern_id]
                p.evidence_count += len(obs_list)
                # delta 평균 갱신 (가중)
                for h, new_d in avg_deltas.items():
                    if h in p.hormone_changes:
                        # 가중 평균 (이전 30%, 새 70% — 적응)
                        p.hormone_changes[h] = round(
                            p.hormone_changes[h] * 0.3 + new_d * 0.7, 3)
                    else:
                        p.hormone_changes[h] = new_d
                # strength saturation
                boost = 0.1 * (1.0 - p.strength)
                p.strength = min(1.0, p.strength + boost)
                p.last_observed = self.time
            else:
                # 신규 패턴
                if len(self.patterns) >= self.max_patterns:
                    self._evict_weakest()

                p = AllostaticPattern(
                    id=pattern_id,
                    trigger_cats=set(trigger_set),
                    hormone_changes=avg_deltas,
                    evidence_count=len(obs_list),
                    strength=0.5,
                    last_observed=self.time,
                    created_at=self.time,
                )
                self.patterns[pattern_id] = p
                for cat in trigger_set:
                    self.cat_to_patterns[cat].add(pattern_id)
                self.learned_count += 1

        # 처리한 obs는 비움 — 강화 누적 방지 (이미 학습된 패턴은 새 obs로만 강화)
        self.pending_observations.clear()

        self.extract_count += 1

    def _make_pattern_id(self, trigger_set: frozenset) -> str:
        """결정론적 ID"""
        return "TRIG|" + "|".join(sorted(trigger_set))

    def _evict_weakest(self):
        if not self.patterns:
            return
        weakest = min(self.patterns.values(),
                     key=lambda p: (p.strength, p.evidence_count))
        self._remove_pattern(weakest.id)

    def _remove_pattern(self, pattern_id: str):
        if pattern_id not in self.patterns:
            return
        p = self.patterns[pattern_id]
        del self.patterns[pattern_id]
        for cat in p.trigger_cats:
            self.cat_to_patterns[cat].discard(pattern_id)

    # ============= 3. 학습된 칵테일 조회 =============
    def get_learned_cocktail(self, category: str) -> Optional[Dict[str, float]]:
        """
        해당 카테고리에 학습된 호르몬 패턴 (있으면).

        Returns:
            {hormone: delta, ...} 또는 None
        """
        pattern_ids = self.cat_to_patterns.get(category, set())
        if not pattern_ids:
            return None

        # 가장 강한 패턴 (strength × evidence)
        best_pattern = None
        best_score = 0.0
        for pid in pattern_ids:
            p = self.patterns.get(pid)
            if not p:
                continue
            score = p.strength * (1.0 + p.evidence_count * 0.1)
            if score > best_score:
                best_score = score
                best_pattern = p

        if best_pattern is None or best_pattern.strength < 0.3:
            return None

        return dict(best_pattern.hormone_changes)

    # ============= 4. 적용 =============
    def apply_learned(self, category: str, scale: float = 0.3) -> Optional[Dict[str, float]]:
        """
        학습된 칵테일을 호르몬에 적용 (작은 비율로).

        Args:
            category: 트리거 카테고리
            scale: 적용 비율 (0-1, 작게 시작 — 안정성)

        Returns:
            적용된 변화 또는 None
        """
        cocktail = self.get_learned_cocktail(category)
        if not cocktail:
            return None

        applied = {}
        for h_name, delta in cocktail.items():
            scaled_delta = delta * scale
            if h_name in self.hs.hormones:
                if scaled_delta > 0:
                    self.hs.hormones[h_name].stimulate(scaled_delta)
                elif scaled_delta < 0:
                    # 감쇠는 직접 적용 (음수 stimulate 대신)
                    self.hs.hormones[h_name].level = max(
                        0.0, self.hs.hormones[h_name].level + scaled_delta)
                applied[h_name] = round(scaled_delta, 3)

        # 패턴 적용 통계
        pattern_ids = self.cat_to_patterns.get(category, set())
        for pid in pattern_ids:
            p = self.patterns.get(pid)
            if p:
                p.apply_count += 1
                p.last_applied = self.time

        self.apply_count += 1
        return applied

    # ============= 5. tick =============
    def tick(self, dt: float = 0.5):
        """
        시간 진행 + 자동 관찰 + 자동 패턴 추출 + 시간 감쇠.
        """
        self.time += dt
        self.tick_count += 1

        # 1) 현재 호르몬 스냅샷
        curr_snapshot = {}
        for h_name in self.TRACKED_HORMONES:
            h = self.hs.hormones.get(h_name)
            if h:
                curr_snapshot[h_name] = h.level

        # 2) 활성 카테고리 (top 5)
        try:
            top_active = self.sa.get_top_active(5) if hasattr(self.sa, 'get_top_active') else []
            curr_active_cats = set(c for c, _ in top_active)
        except Exception:
            curr_active_cats = set()

        # 3) 이전 스냅샷과 비교 → 관찰
        if (self._prev_hormone_snapshot is not None
                and self._prev_active_cats):
            self.observe_response(
                trigger_cats=self._prev_active_cats,
                prev_state=self._prev_hormone_snapshot,
                curr_state=curr_snapshot,
            )

        # 4) 스냅샷 갱신 (다음 tick용)
        self._prev_hormone_snapshot = curr_snapshot
        self._prev_active_cats = curr_active_cats

        # 5) 자동 패턴 추출
        if self.tick_count % self.extract_interval_ticks == 0:
            self.extract_patterns()

        # 6) 자동 적용 (현재 활성 카테고리에 학습된 패턴 있으면)
        if self.tick_count % 30 == 0:  # 매 15초
            for cat in curr_active_cats:
                if cat in self.cat_to_patterns:
                    self.apply_learned(cat, scale=0.1)  # 약하게
                    break  # 한 번에 1개만

        # 7) 시간 감쇠 (안 쓰는 패턴 약화)
        if self.tick_count % 600 == 0:  # 매 5분
            decay_per = self.pattern_decay_per_day * (dt * 600 / 86400.0)
            to_remove = []
            for pid, p in self.patterns.items():
                time_unused = self.time - max(p.last_observed, p.last_applied)
                if time_unused < 3600:
                    continue
                decay = decay_per / (1.0 + p.strength * 2)
                p.strength = max(0.0, p.strength - decay)
                if p.strength < 0.05 and p.evidence_count < self.internalize_threshold:
                    to_remove.append(pid)
            for pid in to_remove:
                self._remove_pattern(pid)

    # ============= 6. introspect =============
    def introspect_patterns(self) -> List[Dict[str, Any]]:
        """학습된 호르몬 패턴들"""
        patterns_list = list(self.patterns.values())
        # 결정론 정렬 (strength ↓, evidence ↓, id ↑)
        patterns_list.sort(key=lambda p: (-p.strength, -p.evidence_count, p.id))

        result = []
        for p in patterns_list[:20]:
            result.append({
                'id': p.id,
                'trigger_cats': sorted(p.trigger_cats),
                'hormone_changes': dict(p.hormone_changes),
                'evidence': p.evidence_count,
                'strength': round(p.strength, 2),
                'apply_count': p.apply_count,
            })
        return result

    # ============= stats =============
    def stats(self) -> Dict[str, Any]:
        return {
            'pattern_count': len(self.patterns),
            'observe_count': self.observe_count,
            'extract_count': self.extract_count,
            'learned_count': self.learned_count,
            'apply_count': self.apply_count,
            'pending_observations': len(self.pending_observations),
        }

    def __repr__(self):
        s = self.stats()
        return (f"AllostaticLearning(patterns={s['pattern_count']}, "
                f"observed={s['observe_count']}, "
                f"learned={s['learned_count']})")
