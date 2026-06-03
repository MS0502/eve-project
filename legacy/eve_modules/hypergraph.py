"""
EVE v35 - E2: HyperGraph (n-ary 관계)
======================================

목표: 카테고리 페어 연결을 넘어 **다중 entity 관계** 표현.

기존 한계:
- SA: 무방향 binary edge (a-b 페어)
- CG: 방향 binary edge (cause→effect)
- 둘 다 페어 연결 → 다중 entity 관계 표현 불가
  예: "민석이 어제 학교에서 사과를 먹었다"
      → agent=민석, patient=사과, location=학교, time=어제, action=먹다
      → 5 entity 관계 한 묶음으로 표현 X

E2 해결:
- HyperEdge 클래스: nodes(Set) + roles(Dict) + temporal/epistemic
- HyperGraph: edges 저장 + node→edges 인덱싱
- query: 부분 매칭, role 기반 검색
- expand_to_pairs: SA에 자동 페어 등록 (호환성 유지)

EVE 절대 원칙:
- 결정론적 (edge_id 자동 생성: nodes 정렬 + role hash)
- read-only on SA/CG (add-only 모듈)
- 카테고리 = 의미 단위 유지 (hyperedge는 *관계*, 카테고리는 그대로)
- 호르몬 변조 (학습 시 ACh↑, 검색 시 약하게)

학계:
- HyperGraphRAG (Luo 2025) — n-ary 관계 직접 표현
- Hex-Tuple (Gadgil 2025) — 결정론적 traversal
- PRoH 2026 — F1 19.73% 향상 (멀티홉)
- Knowledge Hypergraph (Wen 2016) — n-ary KB

핵심 함수:
1. add_edge(nodes, roles, ...)     → edge_id
2. find_edges_with(*nodes)         → List[HyperEdge]
3. find_edges_by_role(role, value) → List[HyperEdge]
4. neighbors_via_edges(node)       → Set[str] (hyperedge 통한 이웃)
5. query(role_filler)              → 부분 매칭
6. expand_to_pairs(min_weight)     → SA 등록할 페어들
7. multi_hop_via_edges(s, t)       → hyperedge traversal 경로

Author: 김민석 + Claude v35
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class HyperEdge:
    """
    n-ary 관계 (multi-entity).

    예: "민석이 어제 학교에서 사과를 먹었다"
        nodes = {민석, 학교, 사과, 먹다, 어제}
        roles = {민석: 'agent', 사과: 'patient',
                 학교: 'location', 먹다: 'action', 어제: 'time'}
    """
    edge_id: str
    nodes: Set[str]              # 관여 카테고리들 (의미 단위)
    roles: Dict[str, str]        # {category: role_name}
    weight: float = 0.5          # 신뢰도/강도 (0-1)
    temporal: Optional[Dict] = None  # {'when': str, 'duration': float, ...}
    epistemic: str = 'belief'    # 'fact' / 'belief' / 'hypothesis'
    source: str = 'manual'       # 'corpus' / 'manual' / 'inference' / 'dialogue'
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    def involves(self, *cats: str) -> bool:
        """주어진 카테고리 모두 이 edge에 있나?"""
        return all(c in self.nodes for c in cats)

    def role_of(self, cat: str) -> Optional[str]:
        """카테고리의 role"""
        return self.roles.get(cat)

    def value_of_role(self, role: str) -> Optional[str]:
        """특정 role의 카테고리"""
        for cat, r in self.roles.items():
            if r == role:
                return cat
        return None

    def __repr__(self):
        # 결정론적 표현 (정렬)
        node_str = ', '.join(sorted(self.nodes))
        return f"HE({self.edge_id[:8]}: {{{node_str}}}, w={self.weight:.2f})"


class HyperGraph:
    """
    n-ary 관계 저장소.

    의존성:
        - SpreadingActivation (선택, expand_to_pairs로 호환성)
        - HormoneSystem (선택)

    설계:
        - edges: edge_id → HyperEdge
        - node_to_edges: 카테고리 → 그 카테고리가 포함된 edge_ids set
        - role_index: role → set of edge_ids (검색 빠르게)
        - 결정론적 edge_id (sha256(sorted_nodes + sorted_roles))
    """

    def __init__(self,
                 spreading_activation=None,
                 hormone_system=None,
                 use_hormone_modulation: bool = True):
        """
        Args:
            spreading_activation: SA (선택, expand_to_pairs로 동기화)
            hormone_system: HS (선택)
            use_hormone_modulation: 호르몬 자극 사용
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 핵심 저장소
        self.edges: Dict[str, HyperEdge] = {}
        self.node_to_edges: Dict[str, Set[str]] = defaultdict(set)
        self.role_index: Dict[str, Set[str]] = defaultdict(set)  # role → edge_ids

        self.time = 0.0

        # 통계
        self.add_count = 0
        self.query_count = 0
        self.expand_count = 0
        self.removed_count = 0

    # ============= 1. edge_id 생성 (결정론) =============
    def _make_edge_id(self,
                     nodes: Set[str],
                     roles: Dict[str, str]) -> str:
        """
        결정론적 edge_id (sha256 hash).
        같은 (nodes, roles) → 같은 id.
        """
        node_str = '|'.join(sorted(nodes))
        role_items = sorted(roles.items())
        role_str = '|'.join(f"{c}:{r}" for c, r in role_items)
        seed = f"{node_str}__{role_str}"
        h = hashlib.sha256(seed.encode('utf-8')).hexdigest()
        return h[:16]  # 16-char hex prefix

    # ============= 2. add_edge =============
    def add_edge(self,
                 nodes: Set[str],
                 roles: Optional[Dict[str, str]] = None,
                 weight: float = 0.5,
                 temporal: Optional[Dict] = None,
                 epistemic: str = 'belief',
                 source: str = 'manual') -> str:
        """
        Hyperedge 추가.

        Args:
            nodes: 관여 카테고리 set (최소 2개)
            roles: {cat: role_name} 매핑 (선택, default empty)
            weight: 0-1
            temporal: 시간 정보
            epistemic: 'fact'/'belief'/'hypothesis'
            source: 'corpus'/'manual'/'inference'/'dialogue'

        Returns:
            edge_id (이미 있으면 weight 강화 후 같은 id)
        """
        if not nodes or len(nodes) < 2:
            raise ValueError("Hyperedge needs at least 2 nodes")

        nodes = set(nodes)  # 복사
        roles = dict(roles) if roles else {}

        edge_id = self._make_edge_id(nodes, roles)

        if edge_id in self.edges:
            # 이미 있음 → weight 강화 (Hebbian 누적)
            edge = self.edges[edge_id]
            edge.weight = min(1.0, edge.weight + weight * 0.3)  # 보존하며 강화
            edge.last_accessed = self.time
            edge.access_count += 1
            return edge_id

        # 새 edge
        edge = HyperEdge(
            edge_id=edge_id,
            nodes=nodes,
            roles=roles,
            weight=float(min(1.0, max(0.0, weight))),
            temporal=temporal,
            epistemic=epistemic,
            source=source,
            created_at=self.time,
            last_accessed=self.time,
        )
        self.edges[edge_id] = edge

        # 인덱스
        for node in nodes:
            self.node_to_edges[node].add(edge_id)
        for role in roles.values():
            self.role_index[role].add(edge_id)

        # 호르몬 (학습 신호)
        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['acetylcholine'].stimulate(0.02)

        self.add_count += 1
        return edge_id

    # ============= 3. find_edges_with — 카테고리 포함 =============
    def find_edges_with(self, *cats: str,
                       require_all: bool = True) -> List[HyperEdge]:
        """
        주어진 카테고리(들)을 포함한 edges.

        Args:
            *cats: 검색 카테고리들
            require_all: True면 모두 포함, False면 하나라도

        Returns:
            HyperEdge list (weight ↓ 정렬)
        """
        if not cats:
            return []

        # 각 cat의 edge set 교집합/합집합
        edge_sets = [self.node_to_edges.get(c, set()) for c in cats]

        if require_all:
            result_ids = edge_sets[0]
            for s in edge_sets[1:]:
                result_ids = result_ids & s
        else:
            result_ids = set()
            for s in edge_sets:
                result_ids = result_ids | s

        # 결정론 정렬 (weight ↓, edge_id ↑)
        result_edges = [self.edges[eid] for eid in result_ids if eid in self.edges]
        result_edges.sort(key=lambda e: (-e.weight, e.edge_id))

        # 접근 기록
        for e in result_edges:
            e.access_count += 1
            e.last_accessed = self.time

        self.query_count += 1
        return result_edges

    # ============= 4. find_edges_by_role =============
    def find_edges_by_role(self,
                          role: str,
                          value: Optional[str] = None) -> List[HyperEdge]:
        """
        특정 role을 가진 edges.

        Args:
            role: role 이름 (예: 'agent')
            value: 그 role의 값 카테고리 (None이면 모든)

        Returns:
            HyperEdge list
        """
        edge_ids = self.role_index.get(role, set())
        edges = [self.edges[eid] for eid in edge_ids if eid in self.edges]

        if value is not None:
            edges = [e for e in edges if e.value_of_role(role) == value]

        edges.sort(key=lambda e: (-e.weight, e.edge_id))
        self.query_count += 1
        return edges

    # ============= 5. neighbors_via_edges — hyperedge 기반 이웃 =============
    def neighbors_via_edges(self,
                           node: str,
                           via_roles: Optional[Set[str]] = None) -> Dict[str, float]:
        """
        node와 같은 hyperedge에 있는 다른 카테고리들.

        Args:
            node: 시작 카테고리
            via_roles: 특정 role을 통해서만 (예: {'agent'} 만)

        Returns:
            {neighbor: max_weight}  — 가장 강한 hyperedge weight
        """
        if node not in self.node_to_edges:
            return {}

        result: Dict[str, float] = {}
        for eid in self.node_to_edges[node]:
            edge = self.edges.get(eid)
            if edge is None:
                continue

            # via_roles 필터
            if via_roles is not None:
                node_role = edge.role_of(node)
                if node_role not in via_roles:
                    continue

            for other in edge.nodes:
                if other == node:
                    continue
                cur = result.get(other, 0.0)
                if edge.weight > cur:
                    result[other] = edge.weight

        return result

    # ============= 6. query — 부분 매칭 =============
    def query(self,
             role_filler: Dict[str, str],
             top_k: int = 5) -> List[HyperEdge]:
        """
        role-filler 부분 매칭.
        예: query({'agent': '민석', 'action': '먹다'})
            → 민석=agent, 먹다=action 인 모든 edges

        Args:
            role_filler: {role: value}
            top_k: 상위 k개

        Returns:
            점수 높은 순 HyperEdge list
        """
        if not role_filler:
            return []

        # 각 (role, value) 페어가 있는 edges
        candidate_sets = []
        for role, value in role_filler.items():
            edges_for_role = self.role_index.get(role, set())
            matching = {eid for eid in edges_for_role
                       if eid in self.edges and
                       self.edges[eid].value_of_role(role) == value}
            candidate_sets.append(matching)

        # 모든 조건 만족 (intersection)
        if not candidate_sets:
            return []
        result_ids = candidate_sets[0]
        for s in candidate_sets[1:]:
            result_ids = result_ids & s

        edges = [self.edges[eid] for eid in result_ids if eid in self.edges]
        edges.sort(key=lambda e: (-e.weight, e.edge_id))
        self.query_count += 1
        return edges[:top_k]

    # ============= 7. expand_to_pairs — SA 등록 =============
    def expand_to_pairs(self,
                       min_weight: float = 0.3,
                       link_strength: float = 0.2) -> int:
        """
        Hyperedge를 binary pair로 분해해서 SA에 등록.
        호환성 유지 — SA 기반 다른 모듈도 hyperedge 정보 활용 가능.

        Args:
            min_weight: 이 weight 이상 hyperedge만 expand
            link_strength: SA에 등록할 페어 강도 (각 페어)

        Returns:
            추가된 페어 수
        """
        if self.sa is None:
            return 0

        added = 0
        for edge in self.edges.values():
            if edge.weight < min_weight:
                continue
            nodes_list = sorted(edge.nodes)  # 결정론
            for i, a in enumerate(nodes_list):
                for b in nodes_list[i+1:]:
                    self.sa.learn_pair(a, b, link_strength * edge.weight)
                    added += 1

        self.expand_count += 1
        return added

    # ============= 8. multi_hop_via_edges =============
    def multi_hop_via_edges(self,
                           start: str,
                           target: str,
                           max_depth: int = 4) -> Optional[List[str]]:
        """
        Hyperedge traversal로 start → target 경로 찾기.
        BFS, 결정론 정렬.

        Returns:
            [start, ..., target] 또는 None
        """
        if start == target:
            return [start]
        if start not in self.node_to_edges:
            return None

        # BFS
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            cur, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue

            # cur의 hyperedge 이웃들 (가중치 높은 순, 알파벳 순)
            neighbors = self.neighbors_via_edges(cur)
            sorted_neighbors = sorted(neighbors.items(),
                                     key=lambda x: (-x[1], x[0]))

            for neighbor, w in sorted_neighbors:
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    # ============= 9. remove =============
    def remove_edge(self, edge_id: str) -> bool:
        """edge 제거. True/False 반환."""
        if edge_id not in self.edges:
            return False
        edge = self.edges[edge_id]
        # 인덱스 제거
        for node in edge.nodes:
            self.node_to_edges[node].discard(edge_id)
            if not self.node_to_edges[node]:
                del self.node_to_edges[node]
        for role in edge.roles.values():
            self.role_index[role].discard(edge_id)
            if not self.role_index[role]:
                del self.role_index[role]
        del self.edges[edge_id]
        self.removed_count += 1
        return True

    # ============= 10. tick + stats =============
    def tick(self, dt: float = 0.5):
        """시간만 증가 (no-op for edges)"""
        self.time += dt

    def stats(self) -> Dict[str, Any]:
        # 가장 connected node
        node_counts = [(n, len(eids)) for n, eids in self.node_to_edges.items()]
        node_counts.sort(key=lambda x: (-x[1], x[0]))
        return {
            'edge_count': len(self.edges),
            'node_count': len(self.node_to_edges),
            'role_count': len(self.role_index),
            'add_count': self.add_count,
            'query_count': self.query_count,
            'expand_count': self.expand_count,
            'removed_count': self.removed_count,
            'top_connected': node_counts[:5],
        }

    def __repr__(self):
        s = self.stats()
        return (f"HyperGraph(edges={s['edge_count']}, "
                f"nodes={s['node_count']}, "
                f"roles={s['role_count']})")
