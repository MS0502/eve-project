"""
EVE v35 - E4: DeepReasoning (멀티홉 path-based 추론)
=====================================================

목표: 1단계 인과를 넘어 **5+ 단계 경로 추론**.

기존 한계:
- CG.parents/children = 직접 1-hop만
- CG.is_cause_of = boolean (경로 자체 없음)
- SA spread = 활성 전파 (path 추적 X)

E4 해결:
- BFS 기반 path search (결정론)
- SA (의미 연결) + CG (인과 방향) 통합
- 멀티홉 추론: "민석 → ? → ? → 기쁨"
- 가중치 곱 점수 = path strength

EVE 절대 원칙 부합:
- 결정론적 (BFS + sorted, heapq tie-break)
- read-only (SA/CG 변경 X — 추론은 관찰일 뿐)
- 호르몬 변조 (선택, 추론 시 ACh↑/Glu↑)
- 카테고리 의미 단위

학계:
- Knowledge Graph Reasoning (Lin 2018) — path-based
- PRoH 2026 — 동적 BFS planning + multi-hop
- Mental Models (Johnson-Laird 1983) — 마음 속 모델 추론
- Pearl 2009 — Causal chain reasoning

핵심 함수:
1. find_path(start, target, mode)         → 최강 path
2. find_paths_top_k(start, target, k)     → 상위 k개 path
3. why_chain(target, depth)               → 인과 체인 (역방향)
4. predict_chain(start, depth)            → 결과 체인 (정방향)
5. multi_hop_reason(start, target)        → 추론 결과 + 설명
6. explain_path(path)                     → 자연어 설명
7. tick(dt)                               → 통계만 (no-op)

Author: 김민석 + Claude v35
"""

import heapq
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class DeepReasoning:
    """
    멀티홉 path-based 추론.

    의존성:
        - SpreadingActivation (의미 그래프, neighbors/weights)
        - CausalGraph (인과 그래프, parents/children/effects/causes)
        - HormoneSystem (선택, 추론 호르몬 변조)
        - NaturalLanguage (선택, explain_path 자연어)

    설계:
        - BFS + 우선순위 큐 (heapq)
        - 점수 = -log(weight 곱) → 짧고 강한 경로 우선
          (-log이라 더 작을수록 강함, heapq 자연 우선순위)
        - 결정론: 동점 시 알파벳 순 노드 선호
        - read-only: SA/CG 가중치/엣지 변경 X
    """

    # mode: 'semantic' (SA only), 'causal' (CG only), 'hybrid' (둘 다)
    MODES = ('semantic', 'causal', 'hybrid')

    def __init__(self,
                 spreading_activation,
                 causal_graph,
                 hormone_system=None,
                 natural_lang=None,
                 default_max_depth: int = 6,
                 min_edge_weight: float = 0.1,
                 max_explored_nodes: int = 5000,
                 use_hormone_modulation: bool = True):
        """
        Args:
            spreading_activation: SA (의미 graph)
            causal_graph: CG (인과 graph)
            hormone_system: HS (선택)
            natural_lang: NL (선택, explain_path용)
            default_max_depth: BFS 기본 깊이
            min_edge_weight: 이 가중치 미만 엣지는 path에 포함 X
            max_explored_nodes: BFS 안전 cap
            use_hormone_modulation: 추론 시 호르몬 자극
        """
        self.sa = spreading_activation
        self.cg = causal_graph
        self.hs = hormone_system
        self.nl = natural_lang

        self.default_max_depth = int(default_max_depth)
        self.min_edge_weight = float(min_edge_weight)
        self.max_explored_nodes = int(max_explored_nodes)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 통계
        self.query_count = 0
        self.path_count = 0       # 발견된 path 수
        self.no_path_count = 0    # 경로 못 찾음
        self.explored_total = 0
        self.deepest_path = 0

    # ============= 1. 핵심 path search =============
    def find_paths_top_k(self,
                         start: str,
                         target: str,
                         k: int = 3,
                         mode: str = 'hybrid',
                         max_depth: Optional[int] = None
                        ) -> List[Dict[str, Any]]:
        """
        Top-k 경로 (가장 짧고 강한 순).

        BFS + priority queue (heapq):
            queue: (cost, depth, last_node, path_tuple)
            cost = -log(weight 곱) — 작을수록 강함
            동점 시 path 알파벳 순 (결정론)

        Returns:
            [{'path': [n1, n2, ...], 'score': float, 'depth': int,
              'edges': [(a, b, w, mode), ...]}]
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if max_depth is None:
            max_depth = self.default_max_depth
        if not start or not target:
            return []
        if start == target:
            return [{'path': [start], 'score': 1.0, 'depth': 0, 'edges': []}]

        self.query_count += 1
        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['acetylcholine'].stimulate(0.02)

        # 우선순위 큐: (cost, depth, path_tuple, edges_tuple)
        # path_tuple = ('start', 'a', 'b', 'target')
        # edges_tuple = (('start','a',0.5,'sem'), ('a','b',0.6,'cau'), ...)
        # heapq 자연 정렬 — cost 작은 거 우선, 동점이면 path tuple 사전순
        import math
        queue: List[Tuple[float, int, Tuple, Tuple]] = []
        heapq.heappush(queue, (0.0, 0, (start,), ()))

        # 발견된 경로 (k개까지)
        found: List[Dict[str, Any]] = []
        # 같은 path 중복 방지
        seen_paths: Set[Tuple] = set()
        explored = 0

        while queue and len(found) < k:
            if explored > self.max_explored_nodes:
                break
            explored += 1

            cost, depth, path, edges = heapq.heappop(queue)
            current = path[-1]

            # 도착!
            if current == target:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                # cost = -log(score) → score = exp(-cost)
                score = math.exp(-cost) if cost > 0 else 1.0
                found.append({
                    'path': list(path),
                    'score': float(score),
                    'depth': depth,
                    'edges': list(edges),
                })
                if depth > self.deepest_path:
                    self.deepest_path = depth
                continue

            if depth >= max_depth:
                continue

            # 이웃 후보 (mode에 따라)
            neighbors_with_weight = self._get_neighbors(current, mode)

            # 결정론 정렬 (가중치 ↓, 알파벳 ↑) — 동점 path tuple 사전순 보장
            sorted_neighbors = sorted(
                neighbors_with_weight.items(),
                key=lambda x: (-x[1][0], x[0])
            )

            for next_node, (weight, edge_mode) in sorted_neighbors:
                # 사이클 방지
                if next_node in path:
                    continue
                if weight < self.min_edge_weight:
                    continue

                new_cost = cost - math.log(max(weight, 1e-9))
                new_path = path + (next_node,)
                new_edges = edges + ((current, next_node, weight, edge_mode),)
                heapq.heappush(queue, (new_cost, depth + 1, new_path, new_edges))

        self.explored_total += explored
        if found:
            self.path_count += len(found)
        else:
            self.no_path_count += 1

        return found

    def find_path(self,
                  start: str,
                  target: str,
                  mode: str = 'hybrid',
                  max_depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """가장 강한 단일 경로 (Top-1)"""
        results = self.find_paths_top_k(start, target, k=1,
                                        mode=mode, max_depth=max_depth)
        return results[0] if results else None

    def _get_neighbors(self, node: str, mode: str) -> Dict[str, Tuple[float, str]]:
        """
        node의 이웃 + 가중치 + edge mode.

        mode='semantic': SA neighbors (양방향)
        mode='causal': CG children (정방향, 결과)
        mode='hybrid': SA + CG children (강한 거 우선)

        Returns:
            {neighbor: (weight, edge_mode)}
        """
        result: Dict[str, Tuple[float, str]] = {}

        # CG children (인과 정방향)
        if mode in ('causal', 'hybrid'):
            for child, w in self.cg.children(node).items():
                if child != node:
                    if child not in result or w > result[child][0]:
                        result[child] = (w, 'causal')

        # SA neighbors (양방향 의미)
        if mode in ('semantic', 'hybrid'):
            if node in self.sa.neighbors:
                for neighbor in self.sa.neighbors[node]:
                    if neighbor == node:
                        continue
                    w = self.sa.get_weight(node, neighbor)
                    if w <= 0:
                        continue
                    # hybrid에서 같은 노드면 강한 가중치 + 더 가치 있는 edge mode
                    if neighbor not in result or w > result[neighbor][0]:
                        result[neighbor] = (w, 'semantic')

        return result

    # ============= 2. 인과 체인 (역방향 BFS) =============
    def why_chain(self,
                  target: str,
                  depth: int = 3,
                  branch: int = 3) -> List[List[Tuple[str, float]]]:
        """
        "왜 target?" — CG 역방향 BFS로 원인 체인.

        Args:
            target: 결과 카테고리
            depth: 최대 깊이
            branch: 각 단계 branch factor

        Returns:
            [[(원인, 가중치), ...], ...] — 가장 강한 chain들
        """
        if not target:
            return []

        chains: List[List[Tuple[str, float]]] = []
        # BFS 시작
        # 큐: (depth, current, chain_so_far)
        queue: List[Tuple[int, str, List[Tuple[str, float]]]] = [(0, target, [])]

        while queue:
            d, cur, chain = queue.pop(0)
            if d >= depth:
                if chain:
                    chains.append(chain)
                continue

            parents = self.cg.parents(cur)
            if not parents:
                if chain:
                    chains.append(chain)
                continue

            # 결정론적 정렬 (가중치 ↓, 알파벳 ↑)
            sorted_parents = sorted(parents.items(),
                                   key=lambda x: (-x[1], x[0]))[:branch]

            for parent, w in sorted_parents:
                # 사이클 방지
                if any(p == parent for p, _ in chain):
                    continue
                new_chain = chain + [(parent, w)]
                queue.append((d + 1, parent, new_chain))

        # 가장 강한 chain 우선 (점수 = 가중치 평균)
        chains.sort(key=lambda c: (-sum(w for _, w in c) / max(1, len(c)),
                                   tuple(p for p, _ in c)))
        self.query_count += 1
        return chains

    def predict_chain(self,
                     start: str,
                     depth: int = 3,
                     branch: int = 3) -> List[List[Tuple[str, float]]]:
        """
        "그러면 무슨 일이?" — CG 정방향 BFS로 결과 체인.
        """
        if not start:
            return []

        chains: List[List[Tuple[str, float]]] = []
        queue: List[Tuple[int, str, List[Tuple[str, float]]]] = [(0, start, [])]

        while queue:
            d, cur, chain = queue.pop(0)
            if d >= depth:
                if chain:
                    chains.append(chain)
                continue

            children = self.cg.children(cur)
            if not children:
                if chain:
                    chains.append(chain)
                continue

            sorted_children = sorted(children.items(),
                                    key=lambda x: (-x[1], x[0]))[:branch]

            for child, w in sorted_children:
                if any(c == child for c, _ in chain):
                    continue
                new_chain = chain + [(child, w)]
                queue.append((d + 1, child, new_chain))

        chains.sort(key=lambda c: (-sum(w for _, w in c) / max(1, len(c)),
                                   tuple(c for c, _ in c)))
        self.query_count += 1
        return chains

    # ============= 3. 통합 추론 =============
    def multi_hop_reason(self,
                        start: str,
                        target: str,
                        max_depth: int = 5) -> Dict[str, Any]:
        """
        통합 추론. start → target 경로 + 의미 분석.

        Returns:
            {
                'has_path': bool,
                'best_path': dict,
                'alt_paths': List[dict],
                'why_chain': List[List[(cat, w)]],
                'predict_chain': List[List[(cat, w)]],
                'depth': int,
            }
        """
        # 1) 멀티홉 경로 탐색
        paths = self.find_paths_top_k(start, target, k=3,
                                     mode='hybrid', max_depth=max_depth)

        # 2) target 인과 체인 (왜 target?)
        why = self.why_chain(target, depth=3, branch=2)

        # 3) start 결과 체인 (start면 무슨 일?)
        predict = self.predict_chain(start, depth=3, branch=2)

        if self.use_hormone_modulation and self.hs is not None:
            # 추론 활동 → ACh + Glutamate
            self.hs.hormones['acetylcholine'].stimulate(0.03)
            if 'glutamate' in self.hs.hormones:
                self.hs.hormones['glutamate'].stimulate(0.02)

        return {
            'has_path': bool(paths),
            'best_path': paths[0] if paths else None,
            'alt_paths': paths[1:] if len(paths) > 1 else [],
            'why_chain': why[:3],
            'predict_chain': predict[:3],
            'depth': paths[0]['depth'] if paths else 0,
            'start': start,
            'target': target,
        }

    # ============= 4. 자연어 설명 =============
    def explain_path(self,
                    path: List[str],
                    edges: Optional[List[Tuple]] = None) -> str:
        """
        path → 자연어 설명.

        Args:
            path: ['민석', '대화', '기쁨']
            edges: optional [(a, b, w, mode), ...]

        Returns:
            '민석 → 대화 → 기쁨 (의미 연결)' 형태
        """
        if not path:
            return ""
        if len(path) == 1:
            return path[0]

        # mode-aware 화살표
        arrows = []
        if edges:
            for (a, b, w, mode) in edges:
                # causal → "이끈다"/"→", semantic → "연관"/"~"
                if mode == 'causal':
                    arrows.append(f"{a} → {b}")
                else:
                    arrows.append(f"{a} ~ {b}")
            return ' → '.join(p for p in path)
        else:
            return ' → '.join(path)

    # ============= 5. 통계 / tick =============
    def tick(self, dt: float = 0.5):
        """no-op (read-only 모듈, 외부 자극으로만 작동)"""
        pass

    def stats(self) -> Dict[str, Any]:
        return {
            'query_count': self.query_count,
            'path_count': self.path_count,
            'no_path_count': self.no_path_count,
            'explored_total': self.explored_total,
            'deepest_path': self.deepest_path,
        }

    def __repr__(self):
        s = self.stats()
        return (f"DeepReasoning(queries={s['query_count']}, "
                f"paths_found={s['path_count']}, "
                f"deepest={s['deepest_path']})")
