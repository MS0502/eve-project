"""
EVE v35 - E9: SemanticDistance (의미 거리)
============================================

목표: 카테고리 간 의미적 거리/유사도 계산.

기존 한계:
- SA: 가중치(weight)만, 의미 거리 별도 추상화 X
- VSA: vector cosine, 합성 카테고리만 사용
- 다른 모듈 (Creative, Analogy, DR)이 각자 ad-hoc 거리 계산

E9 해결:
- 다중 신호 통합:
  1. SA path distance (그래프 최단 경로)
  2. SA neighbor overlap (Jaccard)
  3. VSA cosine similarity (있으면)
  4. CategoryGroup 일치 (HormoneSystem 그룹)
  5. HyperGraph 공유 사건 (있으면)
- 가중 평균으로 0~1 거리 계산
- 결정론적, read-only

EVE 절대 원칙:
- 결정론적 (정렬 + 가중 평균, no random)
- read-only (모든 의존 그래프 변경 X)
- 카테고리 = 의미 단위 유지
- 호르몬 변조 (선택, 검색 시 ACh)

학계:
- Rada et al. 1989 — Path-based semantic distance
- Resnik 1995 — Information Content
- Mikolov 2013 — word2vec (영감, 결정론적 적응)
- Lin 1998 — Information-theoretic similarity
- Pedersen 2004 — WordNet::Similarity

핵심 함수:
1. distance(a, b)                → 0~1 (0=같음, 1=완전 다름)
2. similarity(a, b)              → 1 - distance
3. find_similar(target, k)       → [(cat, sim)] top-k
4. find_dissimilar(target, k)    → [(cat, dist)] top-k
5. cluster_by_similarity(thres)  → List[Set] (의미 클러스터)
6. antonym_candidates(target)    → 같은 그룹 X 가장 먼
7. evaluate_path(path)           → DR path 신뢰도 평가

Author: 김민석 + Claude v35
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class SemanticDistance:
    """
    카테고리 간 의미 거리 다중 신호 통합 모듈.

    의존:
        sa: SpreadingActivation (필수)
        vsa: VSABinding (선택)
        hg: HyperGraph (선택)
        hs: HormoneSystem (선택)
    """

    def __init__(self,
                 spreading_activation,
                 vsa_binding=None,
                 hypergraph=None,
                 hormone_system=None,
                 # 신호 가중치 (합 = 1.0이 권장)
                 w_sa_path: float = 0.30,
                 w_neighbor_overlap: float = 0.25,
                 w_vsa: float = 0.20,
                 w_group: float = 0.15,
                 w_hypergraph: float = 0.10,
                 max_path_depth: int = 5,
                 use_hormone_modulation: bool = True):
        self.sa = spreading_activation
        self.vsa = vsa_binding
        self.hg = hypergraph
        self.hs = hormone_system

        # 가중치 정규화 (vsa/hg 없으면 비례 재분배)
        self.w_sa_path = float(w_sa_path)
        self.w_neighbor_overlap = float(w_neighbor_overlap)
        self.w_vsa = float(w_vsa) if vsa_binding else 0.0
        self.w_group = float(w_group)
        self.w_hypergraph = float(w_hypergraph) if hypergraph else 0.0

        # 정규화 (합=1)
        total_w = (self.w_sa_path + self.w_neighbor_overlap +
                  self.w_vsa + self.w_group + self.w_hypergraph)
        if total_w > 0:
            self.w_sa_path /= total_w
            self.w_neighbor_overlap /= total_w
            self.w_vsa /= total_w
            self.w_group /= total_w
            self.w_hypergraph /= total_w

        self.max_path_depth = int(max_path_depth)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 캐시 (성능 — 동일 페어 반복)
        self._distance_cache: Dict[Tuple[str, str], float] = {}
        self._cache_max = 10000

        # 통계
        self.distance_count = 0
        self.cache_hits = 0
        self.find_similar_count = 0
        self.find_dissimilar_count = 0
        self.cluster_count = 0

    # ============= 1. 신호별 거리 =============
    def _sa_path_distance(self, a: str, b: str) -> float:
        """
        SA 그래프 BFS 최단 경로 거리.
        - 같음 = 0, 직접 연결 = 1/(1+weight), 못 찾음 = 1.0
        """
        if a == b:
            return 0.0
        if a not in self.sa.neighbors or b not in self.sa.neighbors:
            return 1.0

        # BFS
        from collections import deque
        visited = {a}
        queue = deque([(a, 0, 1.0)])  # (node, depth, accumulated_weight_product)

        best_w = 0.0
        while queue:
            cur, depth, w_prod = queue.popleft()
            if depth >= self.max_path_depth:
                continue
            for neighbor in sorted(self.sa.neighbors.get(cur, set())):
                if neighbor in visited:
                    continue
                w = self.sa.get_weight(cur, neighbor)
                if w <= 0:
                    continue
                new_prod = w_prod * w
                if neighbor == b:
                    if new_prod > best_w:
                        best_w = new_prod
                    continue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1, new_prod))

        if best_w > 0:
            # 거리 = 1 - 누적 가중치
            return float(1.0 - min(0.99, best_w))
        return 1.0

    def _neighbor_overlap_distance(self, a: str, b: str) -> float:
        """
        SA 이웃 Jaccard 거리.
        공유 이웃 많을수록 거리 ↓.
        """
        if a == b:
            return 0.0
        n_a = self.sa.neighbors.get(a, set())
        n_b = self.sa.neighbors.get(b, set())
        if not n_a and not n_b:
            return 1.0
        intersection = len(n_a & n_b)
        union = len(n_a | n_b)
        if union == 0:
            return 1.0
        jaccard = intersection / union
        return float(1.0 - jaccard)

    def _vsa_distance(self, a: str, b: str) -> float:
        """VSA cosine 거리 (있으면)"""
        if self.vsa is None or a == b:
            return 0.0 if a == b else 0.5
        try:
            sim = self.vsa.similarity(a, b)
            # cosine -1~1 → 0~1 거리 (cosine 0 또는 음수 = 거리 1)
            return float(1.0 - max(0.0, sim))
        except Exception:
            return 0.5  # 실패 시 중립

    def _group_distance(self, a: str, b: str) -> float:
        """카테고리 그룹 일치 → 0, 다름 → 1"""
        if a == b:
            return 0.0
        ga = self.sa.CATEGORY_GROUPS.get(a)
        gb = self.sa.CATEGORY_GROUPS.get(b)
        if ga is None and gb is None:
            return 0.5  # 둘 다 그룹 없음 = 중립
        if ga == gb:
            return 0.0
        return 1.0

    def _hypergraph_distance(self, a: str, b: str) -> float:
        """HG 공유 사건 — 많을수록 거리 ↓"""
        if self.hg is None or a == b:
            return 0.0 if a == b else 0.5

        edges_a = self.hg.node_to_edges.get(a, set())
        edges_b = self.hg.node_to_edges.get(b, set())
        if not edges_a and not edges_b:
            return 0.5
        shared = edges_a & edges_b
        if shared:
            # 공유 사건 가중치 평균
            ratio = len(shared) / max(len(edges_a), len(edges_b))
            return float(1.0 - min(1.0, ratio))
        return 1.0

    # ============= 2. distance / similarity =============
    def distance(self, a: str, b: str) -> float:
        """
        다중 신호 가중 평균 거리. 0=같음, 1=다름.
        """
        if not a or not b:
            return 1.0
        if a == b:
            return 0.0

        # 캐시
        key = (a, b) if a < b else (b, a)
        if key in self._distance_cache:
            self.cache_hits += 1
            return self._distance_cache[key]

        # 각 신호
        d_path = self._sa_path_distance(a, b)
        d_overlap = self._neighbor_overlap_distance(a, b)
        d_vsa = self._vsa_distance(a, b) if self.vsa else 0.0
        d_group = self._group_distance(a, b)
        d_hg = self._hypergraph_distance(a, b) if self.hg else 0.0

        # 가중 평균
        total = (
            self.w_sa_path * d_path +
            self.w_neighbor_overlap * d_overlap +
            self.w_vsa * d_vsa +
            self.w_group * d_group +
            self.w_hypergraph * d_hg
        )
        total = float(max(0.0, min(1.0, total)))

        # 캐시
        if len(self._distance_cache) < self._cache_max:
            self._distance_cache[key] = total

        # 호르몬 (검색 활동 — 약한 ACh)
        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['acetylcholine'].stimulate(0.005)

        self.distance_count += 1
        return total

    def similarity(self, a: str, b: str) -> float:
        """1 - distance(a, b)"""
        return 1.0 - self.distance(a, b)

    # ============= 3. find_similar / dissimilar =============
    def find_similar(self,
                    target: str,
                    k: int = 5,
                    exclude: Optional[Set[str]] = None,
                    candidates: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """
        target과 가장 비슷한 카테고리 top-k.

        Args:
            target: 기준
            k: top k
            exclude: 제외 set (None이면 target만)
            candidates: 검색 풀 (None이면 SA 모든 카테고리)

        Returns:
            [(category, similarity)] 결정론 정렬 (sim ↓, cat ↑)
        """
        exclude = exclude or set()
        exclude = exclude | {target}

        if candidates is None:
            candidates = set(self.sa.neighbors.keys())

        # 타겟이 그래프에 없으면 빈 결과
        candidates = candidates - exclude

        results = []
        for cat in candidates:
            sim = self.similarity(target, cat)
            results.append((cat, sim))

        # 결정론 정렬
        results.sort(key=lambda x: (-x[1], x[0]))

        self.find_similar_count += 1
        return results[:k]

    def find_dissimilar(self,
                       target: str,
                       k: int = 5,
                       exclude: Optional[Set[str]] = None,
                       candidates: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """target과 가장 먼 카테고리 top-k"""
        exclude = exclude or set()
        exclude = exclude | {target}

        if candidates is None:
            candidates = set(self.sa.neighbors.keys())

        candidates = candidates - exclude

        results = []
        for cat in candidates:
            d = self.distance(target, cat)
            results.append((cat, d))

        # 거리 ↓ 정렬 (먼 거 우선)
        results.sort(key=lambda x: (-x[1], x[0]))

        self.find_dissimilar_count += 1
        return results[:k]

    # ============= 4. 클러스터링 =============
    def cluster_by_similarity(self,
                             threshold: float = 0.5,
                             candidates: Optional[Set[str]] = None) -> List[Set[str]]:
        """
        유사도 임계 이상끼리 클러스터링.
        Single-linkage hierarchical (간단).

        Args:
            threshold: 이 유사도 이상이면 같은 클러스터
            candidates: 클러스터링 대상 (None이면 모두)

        Returns:
            List[Set] — 결정론 정렬 (크기 ↓, 첫 카테고리 알파벳 ↑)
        """
        if candidates is None:
            candidates = set(self.sa.neighbors.keys())

        cats = sorted(candidates)
        if not cats:
            return []

        # union-find
        parent = {c: c for c in cats}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        # 모든 페어 검사
        for i, a in enumerate(cats):
            for b in cats[i+1:]:
                if self.similarity(a, b) >= threshold:
                    union(a, b)

        # 클러스터 모음
        clusters: Dict[str, Set[str]] = defaultdict(set)
        for c in cats:
            clusters[find(c)].add(c)

        # 크기 ↓ 정렬
        cluster_list = list(clusters.values())
        cluster_list.sort(key=lambda s: (-len(s), sorted(s)[0] if s else ''))

        self.cluster_count += 1
        return cluster_list

    # ============= 5. antonym_candidates =============
    def antonym_candidates(self,
                          target: str,
                          k: int = 3,
                          require_different_group: bool = False) -> List[Tuple[str, float]]:
        """
        target과 의미적으로 가장 *반대*인 후보들.

        반의어 ≠ 단순 거리. '반대'는:
        - target과 충분한 의미적 *연관* (완전 무관 X)
        - 그러나 핵심 차이 (그룹 다름 또는 group 정 반대)

        간단한 휴리스틱:
        - distance ≥ 0.5 (충분히 다름)
        - distance ≤ 0.95 (완전 무관 X)
        - require_different_group=True면 다른 group만

        Returns:
            [(cat, distance)] 정렬
        """
        if not target:
            return []

        candidates_data = []
        all_cats = set(self.sa.neighbors.keys()) - {target}

        target_group = self.sa.CATEGORY_GROUPS.get(target)

        for cat in all_cats:
            d = self.distance(target, cat)
            if d < 0.5 or d > 0.95:
                continue
            if require_different_group:
                cat_group = self.sa.CATEGORY_GROUPS.get(cat)
                if target_group and cat_group == target_group:
                    continue
            candidates_data.append((cat, d))

        # 적당히 먼 거 우선 (0.7~0.85 sweet spot)
        # 점수 = 1 - |d - 0.78|  (0.78이 sweet spot)
        candidates_data.sort(key=lambda x: (-(1.0 - abs(x[1] - 0.78)), x[0]))
        return candidates_data[:k]

    # ============= 6. evaluate_path =============
    def evaluate_path(self, path: List[str]) -> Dict[str, Any]:
        """
        DR path의 의미 일관성 평가.

        path 인접 페어 평균 거리 + 양 끝 거리.
        일관된 경로 = 인접 거리 작음 + 양 끝 거리 점진 증가.

        Returns:
            {
                'avg_step_distance': float,
                'end_distance': float,
                'monotonic': bool,
                'coherence_score': float (0-1, 높을수록 일관성)
            }
        """
        if len(path) < 2:
            return {'avg_step_distance': 0.0, 'end_distance': 0.0,
                   'monotonic': True, 'coherence_score': 1.0}

        # 인접 페어 거리
        step_distances = []
        for i in range(len(path) - 1):
            step_distances.append(self.distance(path[i], path[i+1]))

        avg_step = sum(step_distances) / len(step_distances)
        end_dist = self.distance(path[0], path[-1])

        # 단조성 — 시작에서 멀어지는 방향?
        progressive_distances = [self.distance(path[0], p) for p in path]
        monotonic = all(progressive_distances[i] <= progressive_distances[i+1]
                       for i in range(len(progressive_distances) - 1))

        # 일관성 = 작은 step + 큰 end
        coherence = (1.0 - avg_step) * 0.6 + end_dist * 0.4

        return {
            'avg_step_distance': float(avg_step),
            'end_distance': float(end_dist),
            'monotonic': bool(monotonic),
            'coherence_score': float(coherence),
            'step_distances': [float(s) for s in step_distances],
        }

    # ============= 7. tick + stats =============
    def tick(self, dt: float = 0.5):
        """no-op"""
        pass

    def clear_cache(self):
        """캐시 초기화 (그래프 변경 시)"""
        self._distance_cache.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            'distance_count': self.distance_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self._distance_cache),
            'find_similar_count': self.find_similar_count,
            'find_dissimilar_count': self.find_dissimilar_count,
            'cluster_count': self.cluster_count,
            'weights': {
                'sa_path': round(self.w_sa_path, 3),
                'neighbor_overlap': round(self.w_neighbor_overlap, 3),
                'vsa': round(self.w_vsa, 3),
                'group': round(self.w_group, 3),
                'hypergraph': round(self.w_hypergraph, 3),
            },
        }

    def __repr__(self):
        s = self.stats()
        return (f"SemanticDistance(distance={s['distance_count']}, "
                f"cache_hits={s['cache_hits']}, "
                f"clusters={s['cluster_count']})")
