"""
EVE v32 - C1: CausalGraph (Pearl 인과 추론)
=============================================

EVE의 인과 사슬. "왜?" "그러면?" 질의 + do-intervention.

학계:
- Pearl 2009 — Causality: Models, Reasoning, Inference
- Pearl & Mackenzie 2018 — Book of Why (인과 추론 3단계)
- Spirtes, Glymour, Scheines — Causal discovery from observation
- Granger 1969 — Granger causality (시간 우선)

Pearl 3단계 인과 추론:
1. Association  P(Y|X)         — "보는 것" (passive)
2. Intervention P(Y|do(X))     — "하는 것" (active)
3. Counterfactual               — "상상" (Counterfactual 모듈에서)

이 모듈은 1+2단계.
3단계는 v37+ Counterfactual 모듈에서 (이 그래프 기반으로).

핵심 메커니즘:

A. 인과 학습 (관찰):
   - SA activation_history 시간차 윈도우 (2초) 내 A→B 패턴 감지
   - 반복되면 edge weight ↑
   - cycle 방지 (DFS check) — DAG 무결성 보존
   - 호르몬 변조: ACh↑ + dopamine↑ → 학습 가속

B. 인과 질의:
   - parents(X) — X의 직접 원인들
   - children(X) — X의 직접 결과들
   - ancestors(X) — 모든 transitive 원인 (BFS)
   - descendants(X) — 모든 transitive 결과 (BFS)
   - is_cause_of(A, B) — A에서 B로 경로 존재?
   - why(X) — "왜 X?" 강한 부모들 정렬
   - what_if(X) — "X 하면?" 강한 자식들

C. Intervention (간단 do-calculus):
   - do(X) — X를 강제 활성, X의 부모는 disconnect
   - 결과로 children에게 인과 영향 전파

D. 인과 vs 상관 구분:
   - SA Hebbian (양방향) ≠ CausalGraph (방향성)
   - Granger: A가 B보다 *시간상 먼저* + 반복 → A → B

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용) ✅
- 결정론 (같은 history → 같은 그래프) ✅
- 호르몬 변조 (학습률 + 망각률) ✅
- 카테고리 단위 (노드 = 의미 단위) ✅
- 모듈 chain (SA + HS + (선택) EM/AI) ✅

핵심 함수 10+:
1. learn_causal(cause, effect, evidence_weight)  - 직접 학습
2. observe_temporal(window, min_evidence)        - SA history에서 자동 학습
3. parents(cat) / children(cat)                  - 직접 이웃
4. ancestors(cat) / descendants(cat)             - transitive
5. is_cause_of(a, b)                              - 경로 존재?
6. why(cat, top_n)                                - "왜?" 정렬된 원인
7. what_if(cat, top_n)                            - "그러면?" 정렬된 결과
8. intervene(category, depth)                     - do-intervention
9. forget(dt, hormone_modifier)                   - 약한 엣지 망각
10. tick(dt)

Author: 김민석 + Claude v32 (C1)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


class CausalGraph:
    """EVE v32 C1 CausalGraph (Pearl 인과 DAG)"""

    # 시간차 학습 윈도우 (이 시간 내에 A→B면 인과 후보)
    DEFAULT_WINDOW = 2.0  # 초
    
    # 학습 임계
    MIN_EVIDENCE_FOR_EDGE = 2  # 최소 N번 관찰돼야 엣지 생성
    EDGE_WEIGHT_INC = 0.10     # 매 관찰당 가중치 증가
    EDGE_MIN_WEIGHT = 0.05     # 이 가중치 미만 엣지 제거
    
    # Cycle 처리
    CYCLE_REPLACE_RATIO = 0.7  # 새 엣지 weight가 기존 reverse edge × 0.7 이상이면 reverse 제거 후 추가

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 episodic_memory=None,
                 active_inference=None,
                 max_nodes: int = 5000,
                 max_edges_per_node: int = 50):
        """
        Args:
            spreading_activation: 필수 (history 읽음)
            hormone_system: 옵션 (학습률 변조)
            episodic_memory, active_inference: 옵션 (미래 확장용)
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.em = episodic_memory
        self.ai = active_inference

        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node

        # DAG 구조 (양방향 인덱스로 빠른 조회)
        # effects[cause][effect] = {weight, evidence, last_observed}
        self.effects: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        # causes[effect][cause] = same dict (역방향)
        self.causes: Dict[str, Dict[str, Dict]] = defaultdict(dict)

        self.time = 0.0
        self._last_observe_index = 0  # 마지막 observe_temporal 처리한 SA history idx

        # 통계
        self.learn_count = 0
        self.cycle_rejected = 0
        self.cycle_replaced = 0
        self.observe_count = 0
        self.query_count = 0

    # ============= 1. 직접 학습 =============
    def learn_causal(self, cause: str, effect: str,
                    evidence_weight: float = None) -> Dict[str, Any]:
        """
        직접 인과 학습 (외부 호출 가능, observe_temporal이 자동 호출도).

        Args:
            cause, effect: 카테고리
            evidence_weight: None이면 기본값 (호르몬 변조 적용)

        Returns: {'added': bool, 'cycle_action': str|None, 'new_weight': float}
        """
        # self-loop 방지
        if not cause or not effect or cause == effect:
            return {'added': False, 'reason': 'self_loop'}

        # 호르몬 변조 학습률
        if evidence_weight is None:
            evidence_weight = self._compute_learning_rate()

        # Cycle 검사: effect → cause로 가는 경로 있는지
        cycle_action = None
        if self._has_path(effect, cause):
            # 사이클이 생김. 두 가지 처리:
            # - 새 엣지가 충분히 강하면 reverse 제거 후 추가
            # - 아니면 거부
            existing_reverse = self.effects.get(effect, {}).get(cause)
            if existing_reverse and evidence_weight >= existing_reverse['weight'] * self.CYCLE_REPLACE_RATIO:
                # reverse 제거
                self._remove_edge(effect, cause)
                cycle_action = 'replaced'
                self.cycle_replaced += 1
            else:
                self.cycle_rejected += 1
                return {'added': False, 'reason': 'cycle',
                       'cycle_action': 'rejected'}

        # 엣지 추가 또는 강화
        existing = self.effects.get(cause, {}).get(effect)
        if existing:
            existing['weight'] = float(np.clip(
                existing['weight'] + evidence_weight, 0.0, 1.0))
            existing['evidence'] += 1
            existing['last_observed'] = self.time
            edge = existing
        else:
            # max_edges_per_node 체크 (cause의 effects 수)
            if len(self.effects[cause]) >= self.max_edges_per_node:
                # 가장 약한 effect 제거 (LRU + low weight)
                weakest_effect = min(
                    self.effects[cause].items(),
                    key=lambda kv: (kv[1]['weight'], kv[1]['last_observed'])
                )[0]
                self._remove_edge(cause, weakest_effect)

            edge = {
                'weight': float(np.clip(evidence_weight, 0.0, 1.0)),
                'evidence': 1,
                'last_observed': self.time,
                'created': self.time,
            }
            self.effects[cause][effect] = edge
            self.causes[effect][cause] = edge

        self.learn_count += 1
        return {
            'added': True,
            'cycle_action': cycle_action,
            'new_weight': edge['weight'],
            'evidence': edge['evidence'],
        }

    def _compute_learning_rate(self) -> float:
        """호르몬 변조 학습률"""
        rate = self.EDGE_WEIGHT_INC  # 기본 0.10
        if self.hs is not None and hasattr(self.hs, 'hormones'):
            ach = self.hs.hormones.get('acetylcholine')
            da = self.hs.hormones.get('dopamine')
            cort = self.hs.hormones.get('cortisol')
            if ach is not None:
                rate *= (1.0 + (ach.level - 0.4) * 0.5)
            if da is not None:
                rate *= (1.0 + (da.level - 0.5) * 0.3)
            if cort is not None and cort.level > 0.7:
                rate *= 0.7  # 만성 스트레스 → 학습 둔화
        return float(np.clip(rate, 0.01, 0.5))

    def _remove_edge(self, cause: str, effect: str):
        """엣지 제거 (양쪽 인덱스)"""
        if effect in self.effects.get(cause, {}):
            del self.effects[cause][effect]
        if cause in self.causes.get(effect, {}):
            del self.causes[effect][cause]
        # 빈 dict 정리 (defaultdict라 신경)
        if cause in self.effects and not self.effects[cause]:
            del self.effects[cause]
        if effect in self.causes and not self.causes[effect]:
            del self.causes[effect]

    # ============= 2. SA history 자동 학습 =============
    def observe_temporal(self, window: float = None,
                        min_strength: float = 0.0,
                        max_pairs_per_call: int = 100) -> Dict[str, int]:
        """
        SA activation_history에서 시간차 패턴 추출 → 자동 학습.

        Granger: A가 B보다 시간상 먼저 활성됐고 윈도우 내면 A → B 후보.

        Args:
            window: 시간차 임계 (초)
            min_strength: 최소 활성 강도 (현재는 사용 X — history에 활성 강도 X)
            max_pairs_per_call: 한 번 호출당 최대 페어 수

        Returns: {'pairs_observed': N, 'edges_learned': N}
        """
        if window is None:
            window = self.DEFAULT_WINDOW

        history = self.sa.activation_history
        if len(history) < 2:
            return {'pairs_observed': 0, 'edges_learned': 0}

        # 마지막 처리 인덱스 이후만
        start_idx = max(0, self._last_observe_index)
        new_entries = history[start_idx:]

        pairs_observed = 0
        edges_learned = 0

        for i, (t_a, cat_a) in enumerate(new_entries):
            global_i = start_idx + i
            # 이후 엔트리들과 페어
            for j in range(i + 1, len(new_entries)):
                t_b, cat_b = new_entries[j]
                if t_b - t_a > window:
                    break  # 윈도우 벗어남 (history 정렬됨 가정)
                if cat_a == cat_b:
                    continue
                pairs_observed += 1
                if pairs_observed > max_pairs_per_call:
                    break
                # 학습 시도
                result = self.learn_causal(cat_a, cat_b)
                if result.get('added'):
                    edges_learned += 1

        # 다음 호출은 history 끝부터
        self._last_observe_index = len(history)
        self.observe_count += 1

        return {
            'pairs_observed': pairs_observed,
            'edges_learned': edges_learned,
        }

    # ============= 3. 직접 이웃 =============
    def parents(self, category: str) -> Dict[str, float]:
        """직접 원인들 + 가중치"""
        return {p: e['weight'] for p, e in self.causes.get(category, {}).items()}

    def children(self, category: str) -> Dict[str, float]:
        """직접 결과들 + 가중치"""
        return {c: e['weight'] for c, e in self.effects.get(category, {}).items()}

    # ============= 4. Transitive (BFS) =============
    def ancestors(self, category: str, max_depth: int = 5) -> Set[str]:
        """모든 조상 (transitive parents)"""
        visited = set()
        frontier = {category}
        for _ in range(max_depth):
            new_frontier = set()
            for cat in frontier:
                for p in self.causes.get(cat, {}).keys():
                    if p not in visited and p != category:
                        visited.add(p)
                        new_frontier.add(p)
            frontier = new_frontier
            if not frontier:
                break
        return visited

    def descendants(self, category: str, max_depth: int = 5) -> Set[str]:
        """모든 후손 (transitive children)"""
        visited = set()
        frontier = {category}
        for _ in range(max_depth):
            new_frontier = set()
            for cat in frontier:
                for c in self.effects.get(cat, {}).keys():
                    if c not in visited and c != category:
                        visited.add(c)
                        new_frontier.add(c)
            frontier = new_frontier
            if not frontier:
                break
        return visited

    # ============= 5. 경로 / Cycle 검사 =============
    def _has_path(self, source: str, target: str, max_depth: int = 10) -> bool:
        """source → target 경로 존재? (BFS, cycle 체크용)"""
        if source == target:
            return True
        visited = {source}
        frontier = {source}
        for _ in range(max_depth):
            new_frontier = set()
            for cat in frontier:
                for c in self.effects.get(cat, {}).keys():
                    if c == target:
                        return True
                    if c not in visited:
                        visited.add(c)
                        new_frontier.add(c)
            frontier = new_frontier
            if not frontier:
                break
        return False

    def is_cause_of(self, a: str, b: str) -> bool:
        """A가 B의 원인인가? (transitive)"""
        self.query_count += 1
        return self._has_path(a, b)

    # ============= 6. "왜?" / "그러면?" =============
    def why(self, category: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        "왜 X 일어났지?" — X의 부모들을 가중치 순으로.
        결정론: 동점 시 알파벳 순.
        """
        self.query_count += 1
        ps = self.parents(category)
        if not ps:
            return []
        items = sorted(ps.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:top_n]

    def what_if(self, category: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        "X 하면 어떻게 되지?" — X의 자식들을 가중치 순으로.
        """
        self.query_count += 1
        cs = self.children(category)
        if not cs:
            return []
        items = sorted(cs.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:top_n]

    # ============= 7. Intervention (단순 do) =============
    def intervene(self, category: str,
                 strength: float = 0.7,
                 depth: int = 2) -> Dict[str, Any]:
        """
        do(category) — 강제 활성. 부모와 인과적으로 disconnect (Pearl).

        효과:
        1. SA에 category 활성 (강제)
        2. depth 만큼 자식들도 비례 활성 (인과 사슬)
        3. 결과 dict 반환

        Note: 진짜 do-calculus는 부모 노드의 영향을 무효화하지만,
        EVE 단순 버전은 자식 활성에 집중. 정식 do는 v37+.
        """
        if not category or self.sa is None:
            return {'activated': [], 'depth_reached': 0}

        # 1) 자기 자신 강제 활성
        self.sa.activate(category, strength)
        activated = [(category, strength, 0)]

        # 2) 자식들 인과 활성 (decay)
        current_layer = {category}
        cur_strength = strength
        for d in range(1, depth + 1):
            next_layer = set()
            cur_strength *= 0.6  # decay per depth
            for parent in current_layer:
                for child, edge in self.effects.get(parent, {}).items():
                    propagated = cur_strength * edge['weight']
                    if propagated >= 0.05:  # 너무 약한 건 스킵
                        self.sa.activate(child, propagated)
                        activated.append((child, propagated, d))
                        next_layer.add(child)
            current_layer = next_layer
            if not current_layer:
                break

        return {
            'activated': activated,
            'depth_reached': len(set(x[2] for x in activated)) - 1,
            'category': category,
        }

    # ============= 8. 망각 (Ebbinghaus) =============
    def forget(self, dt: float = 1.0,
              base_decay: float = 0.001,
              hormone_modifier: float = 1.0) -> Dict[str, int]:
        """
        오래 안 쓴 엣지 점진 약화. 임계 미만 제거.

        - last_observed 오래된 + low weight → 약화
        - cort 만성 ↑ → 가속 (hormone_modifier > 1)
        - bdnf ↑ → 둔화 (hormone_modifier < 1)

        Returns: {'edges_weakened': N, 'edges_removed': N}
        """
        edges_weakened = 0
        edges_removed = 0
        to_remove = []

        for cause, effects_map in list(self.effects.items()):
            for effect, edge in list(effects_map.items()):
                age = self.time - edge['last_observed']
                if age <= 1.0:
                    continue  # 너무 최근 거 보존
                # 지수 망각
                decay_amount = dt * base_decay * hormone_modifier * (1.0 + age / 100.0)
                old_weight = edge['weight']
                new_weight = float(old_weight * np.exp(-decay_amount))
                if new_weight < self.EDGE_MIN_WEIGHT:
                    to_remove.append((cause, effect))
                    edges_removed += 1
                else:
                    edge['weight'] = new_weight
                    edges_weakened += 1

        for cause, effect in to_remove:
            self._remove_edge(cause, effect)

        return {
            'edges_weakened': edges_weakened,
            'edges_removed': edges_removed,
        }

    # ============= 9. tick =============
    def tick(self, dt: float):
        """시간 누적."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def edge_count(self) -> int:
        return sum(len(es) for es in self.effects.values())

    def node_count(self) -> int:
        return len(set(self.effects.keys()) | set(self.causes.keys()))

    def get_edge(self, cause: str, effect: str) -> Optional[Dict]:
        return self.effects.get(cause, {}).get(effect)

    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'nodes': self.node_count(),
            'edges': self.edge_count(),
            'learn_count': self.learn_count,
            'cycle_rejected': self.cycle_rejected,
            'cycle_replaced': self.cycle_replaced,
            'observe_count': self.observe_count,
            'query_count': self.query_count,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"CausalGraph(t={s['time']:.1f}s, "
                f"nodes={s['nodes']}, edges={s['edges']}, "
                f"learns={s['learn_count']})")
