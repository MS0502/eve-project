"""
EVE v32 - SpreadingActivation
==============================

카테고리 활성화 메커니즘 (구 real_snn / SpontaneousSNN의 v32 정정)

핵심 원칙:
- 뉴런 시뮬 X. 카테고리(=의미 단위)가 활성화 대상
- 시냅스 X. 카테고리 간 가중 연결 (graph)
- 발화 X. 카테고리 활성도 (continuous)
- 학계: Spreading Activation Network (Anderson 1983, Quillian 1968)

기능:
1. activate(cat, strength)              - 카테고리 활성
2. decay(dt)                            - 시간 기반 감쇠
3. spread(steps)                        - 활성 전파 (이웃으로 확산)
4. get_active(threshold)                - 임계 넘는 카테고리
5. apply_hormone_modulation(mods)        - 호르몬 → 임계 변조 (A1 연결)
6. learn_pair(a, b, strength)           - Hebbian 연결 강화
7. learn_chain([a, b, c], strength)     - 순차 chain
8. pattern_complete(partial)            - 부분 → 연관 카테고리
9. is_similar(a, b)                     - 카테고리 유사도

EVE 절대 원칙 부합:
- 확률 X (random/softmax/sample 사용 안 함) ✅
- 결정론적 (같은 입력 → 같은 출력) ✅
- 호르몬 modulation 받음 ✅
- 카테고리 의미 단위 ✅

Author: 김민석 + Claude v32
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class SpreadingActivation:
    """
    EVE v32 카테고리 활성화 시스템

    내부 상태:
        activations: {category: float}     # 현재 활성도 (0-1)
        weights: {(a, b): float}           # 카테고리 간 연결 가중치
        category_groups: {cat: group}      # 카테고리 → 그룹 (호르몬 변조용)

    사용:
        sa = SpreadingActivation()
        sa.activate('기쁨', 0.8)
        sa.spread(steps=2)  # 연결된 카테고리로 활성 전파
        active = sa.get_active(threshold=0.3)
    """

    # 카테고리 → 그룹 매핑 (HormoneSystem과 동일)
    CATEGORY_GROUPS = {
        # reward
        '기쁨': 'reward', '성취': 'reward', '쾌감': 'reward',
        '맛있다': 'reward', '좋다': 'reward', '재밌다': 'reward',
        # social
        '민석': 'social', '친구': 'social', '대화': 'social',
        '유대': 'social', '사랑': 'social', '함께': 'social',
        # threat
        '위험': 'threat', '아프다': 'threat', '무섭다': 'threat',
        '칼': 'threat', '죽음': 'threat', '공격': 'threat',
        # attention
        '주의': 'attention', '집중': 'attention', '중요': 'attention',
        '관찰': 'attention', '경계': 'attention',
        # memory
        '기억': 'memory', '회상': 'memory', '과거': 'memory', '경험': 'memory',
        # rest
        '잠': 'rest', '휴식': 'rest', '졸림': 'rest', '편안': 'rest',
        # curiosity
        '왜': 'curiosity', '어떻게': 'curiosity', '새로운': 'curiosity',
        '모르는': 'curiosity', '궁금': 'curiosity',
        # aversion
        '싫다': 'aversion', '거절': 'aversion', '피하다': 'aversion',
        '회피': 'aversion', '불쾌': 'aversion',
    }

    def __init__(self,
                 base_threshold: float = 0.3,
                 decay_rate: float = 0.2,
                 spread_factor: float = 0.5,
                 max_categories: int = 10000):
        """
        Args:
            base_threshold: 기본 활성 임계 (호르몬으로 변조됨)
            decay_rate: 시간당 감쇠율 (1/s)
            spread_factor: 활성 전파 시 감쇠 (0-1)
            max_categories: 최대 카테고리 수
        """
        self.base_threshold = base_threshold
        self.decay_rate = decay_rate
        self.spread_factor = spread_factor
        self.max_categories = max_categories

        # 핵심 상태
        self.activations: Dict[str, float] = {}
        self.weights: Dict[Tuple[str, str], float] = {}
        self.neighbors: Dict[str, Set[str]] = defaultdict(set)

        # 호르몬 변조 (A1 연결)
        self.hormone_modulation: Dict[str, float] = {}  # 그룹 → 변조 계수

        # 활성 history (자발 활성용)
        self.activation_history: List[Tuple[float, str]] = []
        self.time = 0.0

    # ============= 1. 활성 =============
    def activate(self, category: str, strength: float = 0.5):
        """
        카테고리 활성화. strength 누적.

        Args:
            category: 카테고리 이름 (의미 단위)
            strength: 활성 강도 (0-1)
        """
        if not category:
            return
        cur = self.activations.get(category, 0.0)
        self.activations[category] = float(np.clip(cur + strength, 0.0, 1.0))
        self.activation_history.append((self.time, category))
        # history 크기 제한
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-500:]

    def deactivate(self, category: str):
        """카테고리 비활성"""
        if category in self.activations:
            del self.activations[category]

    # ============= 2. 감쇠 =============
    def decay(self, dt: float):
        """
        시간 기반 감쇠. 호르몬 modulation 반영.
        활성도가 임계 이하면 자동 제거.
        """
        self.time += dt

        # dt 안전 (지수 감쇠)
        alpha = 1.0 - np.exp(-self.decay_rate * dt)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        to_remove = []
        for cat, val in self.activations.items():
            new_val = val * (1.0 - alpha)
            if new_val < 0.01:
                to_remove.append(cat)
            else:
                self.activations[cat] = new_val

        for cat in to_remove:
            del self.activations[cat]

    # ============= 3. 전파 (Spreading Activation 핵심) =============
    def spread(self, steps: int = 1):
        """
        활성이 연결된 카테고리로 전파.

        Anderson 1983: 활성도 × 가중치 × spread_factor → 이웃 카테고리

        Args:
            steps: 전파 단계 수
        """
        for _ in range(steps):
            new_activations = dict(self.activations)

            for cat, val in list(self.activations.items()):
                if cat not in self.neighbors:
                    continue
                # 이웃들에게 활성 전파
                for neighbor in self.neighbors[cat]:
                    weight = self.weights.get(self._key(cat, neighbor), 0.0)
                    if weight > 0:
                        delta = val * weight * self.spread_factor
                        cur = new_activations.get(neighbor, 0.0)
                        new_activations[neighbor] = float(np.clip(cur + delta, 0.0, 1.0))

            self.activations = new_activations

    # ============= 4. 활성 조회 =============
    def get_active(self, threshold: Optional[float] = None) -> Set[str]:
        """
        임계를 넘는 카테고리 반환.
        호르몬 변조 적용된 임계 사용.
        """
        if threshold is None:
            threshold = self.base_threshold

        active = set()
        for cat, val in self.activations.items():
            actual_threshold = self._effective_threshold(cat, threshold)
            if val >= actual_threshold:
                active.add(cat)
        return active

    def _effective_threshold(self, category: str, base: float) -> float:
        """호르몬 변조 적용된 실제 임계"""
        group = self.CATEGORY_GROUPS.get(category)
        if group and group in self.hormone_modulation:
            return base * self.hormone_modulation[group]
        return base

    def get_activation_level(self, category: str) -> float:
        """특정 카테고리 활성도"""
        return self.activations.get(category, 0.0)

    def get_top_active(self, n: int = 5) -> List[Tuple[str, float]]:
        """가장 활성 높은 n개"""
        items = list(self.activations.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    # ============= 5. 호르몬 변조 (A1 연결) =============
    def apply_hormone_modulation(self, modulation: Dict[str, float]):
        """
        HormoneSystem.get_category_threshold_modulation() 결과 받음.

        Args:
            modulation: {group_name: factor}
                factor < 1.0 → 임계 ↓ (활성 쉬움)
                factor > 1.0 → 임계 ↑ (활성 어려움)
        """
        self.hormone_modulation = dict(modulation)

    # ============= 6. Hebbian 학습 =============
    def learn_pair(self, a: str, b: str, strength: float = 0.1):
        """
        카테고리 간 연결 강화 (Hebbian: 함께 활성된 것은 함께 강화).

        Args:
            a, b: 카테고리 이름
            strength: 연결 강화 양
        """
        if not a or not b or a == b:
            return
        key = self._key(a, b)
        cur = self.weights.get(key, 0.0)
        self.weights[key] = float(np.clip(cur + strength, 0.0, 1.0))
        self.neighbors[a].add(b)
        self.neighbors[b].add(a)

    def learn_chain(self, chain: List[str], strength: float = 0.1):
        """순차 chain 학습 (a→b→c)"""
        for i in range(len(chain) - 1):
            self.learn_pair(chain[i], chain[i+1], strength)

    def get_weight(self, a: str, b: str) -> float:
        """두 카테고리 간 연결 가중치"""
        return self.weights.get(self._key(a, b), 0.0)

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        """대칭 키 (순서 무관)"""
        return (a, b) if a < b else (b, a)

    # ============= 7. Pattern Completion =============
    def pattern_complete(self, partial: Set[str], depth: int = 2,
                        min_weight: float = 0.2) -> Set[str]:
        """
        부분 활성 → 연관 카테고리 찾기 (Hopfield 영감).
        단, 활성 자체는 변경 X. 후보만 반환.

        Args:
            partial: 부분 활성 카테고리
            depth: 탐색 깊이
            min_weight: 최소 연결 강도

        Returns:
            연관 카테고리 set
        """
        completed = set(partial)
        frontier = set(partial)

        for _ in range(depth):
            new_frontier = set()
            for cat in frontier:
                if cat not in self.neighbors:
                    continue
                for neighbor in self.neighbors[cat]:
                    if neighbor in completed:
                        continue
                    weight = self.weights.get(self._key(cat, neighbor), 0.0)
                    if weight >= min_weight:
                        new_frontier.add(neighbor)
            completed.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break

        return completed - partial  # 새로 완성된 것만

    # ============= 8. 카테고리 유사도 =============
    def is_similar(self, a: str, b: str, min_overlap: int = 1) -> bool:
        """
        두 카테고리 유사도 (공유 이웃 기반).

        Args:
            min_overlap: 최소 공유 이웃 수
        """
        if a == b:
            return True
        if a not in self.neighbors or b not in self.neighbors:
            return False
        overlap = self.neighbors[a] & self.neighbors[b]
        return len(overlap) >= min_overlap

    def similarity_score(self, a: str, b: str) -> float:
        """Jaccard 유사도 (0-1)"""
        if a == b:
            return 1.0
        if a not in self.neighbors and b not in self.neighbors:
            return 0.0
        n_a = self.neighbors.get(a, set())
        n_b = self.neighbors.get(b, set())
        if not n_a and not n_b:
            return 0.0
        intersection = len(n_a & n_b)
        union = len(n_a | n_b)
        return intersection / union if union > 0 else 0.0

    # ============= 9. STDP-등가 (시간 기반 가중치 강화) =============
    def stdp_update(self, window: float = 2.0, strength: float = 0.05):
        """
        STDP 등가: 최근 함께 활성된 카테고리들의 연결 강화.
        실제 시간 차이는 카테고리 레벨에선 단순화 (window 안에서 함께 활성된 것).

        Args:
            window: 시간 창 (초)
            strength: 강화 양
        """
        now = self.time
        recent = [(t, c) for t, c in self.activation_history if now - t <= window]

        # 같은 window 안의 모든 페어 강화
        cats_in_window = list({c for _, c in recent})
        for i, a in enumerate(cats_in_window):
            for b in cats_in_window[i+1:]:
                self.learn_pair(a, b, strength)

    # ============= 10. 카테고리 자기 진화 (v32) =============
    # 학계 기반:
    # - Ebbinghaus 1885: R = e^(-t/S) 지수 감쇠
    # - Bjork & Bjork 1992 (New Theory of Disuse):
    #   storage strength (불변) vs retrieval strength (감쇠)
    #   → 우리 모델: stability = storage, weight = retrieval
    # - Spaced repetition: 자주 활성 → stability ↑ → 망각 느림
    # - Cortisol 만성 → 망각 가속 (해마 손상)
    # - BDNF → 망각 둔화 (시냅스 강화)

    def discover_category(self, new_cat: str,
                         context_categories: Set[str] = None,
                         initial_strength: float = 0.3,
                         link_strength: float = 0.2) -> bool:
        """
        새 카테고리 발견 (자연어 처리 시 모르는 단어 등).
        EVE가 스스로 그래프에 추가.

        Args:
            new_cat: 새 카테고리 이름
            context_categories: 함께 등장한 맥락 카테고리들 (Hebbian으로 연결)
            initial_strength: 초기 활성도
            link_strength: 맥락 연결 가중치

        Returns:
            새로 추가됐으면 True, 이미 있어서 강화만 했으면 False
        """
        if not new_cat:
            return False

        # 이미 그래프에 있으면 활성/연결만 강화
        already_known = (new_cat in self.neighbors) or (new_cat in self.activations)

        # 활성
        self.activate(new_cat, initial_strength)

        # 맥락과 Hebbian 연결
        if context_categories:
            for ctx in context_categories:
                if ctx and ctx != new_cat:
                    self.learn_pair(new_cat, ctx, link_strength)

        return not already_known

    def _compute_stability(self, category: str) -> float:
        """
        카테고리의 stability (Bjork & Bjork 1992).
        자주 활성된 카테고리일수록 stability ↑ → 망각 느림.

        stability = 1 + log(1 + access_count_in_history)

        항상 ≥ 1.0 보장 (분모 안전)
        """
        # activation_history에서 빈도 계산
        count = sum(1 for _, c in self.activation_history if c == category)
        return 1.0 + float(np.log1p(count))

    def forget(self,
              dt: float = 1.0,
              base_decay: float = 0.001,
              removal_threshold: float = 0.05,
              hormone_modifier: float = 1.0,
              protected_groups: Set[str] = None,
              protected_categories: Set[str] = None) -> Dict[str, int]:
        """
        Ebbinghaus + Bjork 망각.

        매 tick에 작은 양으로 점진적 약화.
        가능한 자주 호출 권장 (예: 매 10 tick).

        알고리즘:
            1. 각 페어 (a, b) 가중치 점진 감쇠:
               stability = (stability(a) + stability(b)) / 2
               new_w = w × exp(-dt × base_decay × hormone_modifier / stability)
            2. new_w < removal_threshold → 페어 제거
            3. 카테고리의 모든 페어가 제거되고 활성도 약하면 → 카테고리 그래프에서 제거
            4. 보호된 그룹/카테고리는 망각 X

        Args:
            dt: 시간 진행 (초)
            base_decay: 기본 감쇠율 (1/s, 작게 설정)
            removal_threshold: 이 가중치 미만 페어 제거
            hormone_modifier: 호르몬 영향 배수 (cortisol↑ → >1.0, BDNF↑ → <1.0)
            protected_groups: 망각 안 할 카테고리 그룹들
            protected_categories: 망각 안 할 카테고리들 (개별)

        Returns:
            {'pairs_removed': N, 'categories_removed': N}
        """
        if protected_groups is None:
            protected_groups = {'reward', 'social', 'threat'}  # 핵심 그룹 보호
        if protected_categories is None:
            protected_categories = set()

        # 1) 페어 약화
        pairs_to_remove = []
        # stability 캐시 (성능)
        stability_cache: Dict[str, float] = {}

        for pair, w in list(self.weights.items()):
            a, b = pair

            # 보호 체크
            if a in protected_categories or b in protected_categories:
                continue
            ga = self.CATEGORY_GROUPS.get(a)
            gb = self.CATEGORY_GROUPS.get(b)
            if ga in protected_groups or gb in protected_groups:
                continue

            # stability (캐시)
            if a not in stability_cache:
                stability_cache[a] = self._compute_stability(a)
            if b not in stability_cache:
                stability_cache[b] = self._compute_stability(b)
            stability = (stability_cache[a] + stability_cache[b]) / 2.0

            # Ebbinghaus 지수 감쇠 (Bjork stability 보정)
            # cortisol 만성 → hormone_modifier > 1 → 더 빨리 망각
            decay_amount = dt * base_decay * hormone_modifier / stability
            new_w = float(w * np.exp(-decay_amount))

            if new_w < removal_threshold:
                pairs_to_remove.append(pair)
            else:
                self.weights[pair] = new_w

        # 2) 약한 페어 제거
        for pair in pairs_to_remove:
            a, b = pair
            del self.weights[pair]
            self.neighbors[a].discard(b)
            self.neighbors[b].discard(a)

        # 3) 고립된 카테고리 제거
        # 조건: 이웃 0개 + 활성도 낮음 + 보호 X
        isolated = []
        all_cats = set(self.neighbors.keys()) | set(self.activations.keys())
        for cat in all_cats:
            # 보호 체크
            if cat in protected_categories:
                continue
            group = self.CATEGORY_GROUPS.get(cat)
            if group in protected_groups:
                continue

            # 이웃 있나
            has_neighbors = bool(self.neighbors.get(cat))
            if has_neighbors:
                continue

            # 활성도 낮나
            activation = self.activations.get(cat, 0.0)
            if activation >= 0.05:
                continue

            isolated.append(cat)

        for cat in isolated:
            self.neighbors.pop(cat, None)
            self.activations.pop(cat, None)

        # 4) 빈 neighbors 정리 (defaultdict라 None 비교 X)
        empty_keys = [k for k, v in self.neighbors.items() if not v]
        for k in empty_keys:
            if k in isolated:
                del self.neighbors[k]
            # 그 외는 보호된 거니 유지

        return {
            'pairs_removed': len(pairs_to_remove),
            'categories_removed': len(isolated),
        }

    # ============= 상태 조회 =============
    def get_state(self) -> Dict:
        return {
            'time': self.time,
            'num_categories': len(self.activations),
            'num_connections': len(self.weights),
            'top_active': self.get_top_active(5),
            'hormone_modulation_active': bool(self.hormone_modulation),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"SpreadingActivation(t={s['time']:.1f}, "
                f"active={s['num_categories']}, "
                f"connections={s['num_connections']})")
