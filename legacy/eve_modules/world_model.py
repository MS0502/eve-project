"""
EVE v32 - C2: WorldModel (세계 지식 구조)
============================================

EVE의 메타 레이어 — 다른 모듈의 정보를 통합한 *세계 지식* view.

티어 C의 마지막 모듈. 다른 모듈들 위의 메타 read-only view.

학계:
- Forbus & Gentner 1986 — Qualitative Process Theory
- Lakoff & Johnson 1980 — Metaphors We Live By (image schemas)
- Friston 2010 — Generative Model in FEP
- Tenenbaum et al. 2011 — How to grow a mind
- Lake et al. 2017 — Building machines that learn and think
- Battaglia et al. 2018 — Relational inductive biases (graph as world model)

핵심 통찰:
- 인지 시스템 = "세계의 모델"을 가진 시스템
- EVE의 세계 = 카테고리 + 관계 + 일화 + 패턴
- WorldModel은 *새 데이터 구조 X* — 기존 모듈 통합 read view
- "EVE가 아는 모든 것"의 구조화된 view

EVE 특화:
- Concept = 카테고리 (SA 그룹 + CG 부모/자식 + EM 일화 + AN 유추)
- Domain = 같은 그룹 + 인접 카테고리들의 클러스터
- Schema = CG에서 반복되는 인과 패턴
- 도메인 = 동적 (CATEGORY_GROUPS + CG 클러스터 + 새 카테고리 자동 분류)

핵심 함수:

A. describe_concept(category):
   카테고리의 모든 측면 통합:
   - group (속성)
   - parents/children (인과)
   - episodes (일화 빈도)
   - analogies (비슷한 카테고리)
   - last_seen (시간)

B. list_domains():
   도메인별 카테고리 클러스터.
   - 명시적 그룹 (SA.CATEGORY_GROUPS)
   - 동적 그룹 (CG에서 강하게 연결된 카테고리들)

C. categorize(category):
   카테고리의 도메인 추론 (그룹 미정 시).
   - 그룹 있으면 그대로
   - 없으면 인접 카테고리들의 그룹 다수결

D. find_schema(seed=None):
   반복되는 패턴 찾기.
   - "X → Y → Z" 길이 ≥ 2 인과 사슬
   - seed가 등장하는 사슬 (있을 때만)

E. knowledge_overview():
   전체 지식 요약 dict.
   - 도메인 수, 카테고리 수, 인과 엣지, 일화 수, 평균 카테고리당 관계 등

F. relate(a, b):
   A와 B의 관계 종합 분석.
   - 같은 그룹? 인과 있음? 유추 가능? 일화 공유?

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론
- 호르몬 변조 X (knowledge view에 영향 없음)
- 카테고리 단위
- Read-only on 모든 의존 모듈

핵심 함수 6:
1. describe_concept(category)
2. list_domains()
3. categorize(category)
4. find_schema(seed=None, max_chains=5)
5. knowledge_overview()
6. relate(a, b)
+ tick(dt)

Author: 김민석 + Claude v32 (C2)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter


class WorldModel:
    """EVE v32 C2 WorldModel — 통합 메타 view"""

    def __init__(self,
                 spreading_activation,
                 causal_graph=None,
                 episodic_memory=None,
                 analogy=None,
                 temporal=None,
                 hormone_system=None):
        """
        Args:
            spreading_activation: 필수 (그룹 + 카테고리)
            causal_graph: 옵션 (인과 분석)
            episodic_memory: 옵션 (일화)
            analogy: 옵션 (유추)
            temporal: 옵션 (시간 거리)
            hormone_system: 옵션 (현재 상태 추가)
        """
        self.sa = spreading_activation
        self.cg = causal_graph
        self.em = episodic_memory
        self.an = analogy
        self.tm = temporal
        self.hs = hormone_system

        self.time = 0.0

        # 통계
        self.describe_count = 0
        self.list_domains_count = 0
        self.categorize_count = 0
        self.schema_count = 0
        self.relate_count = 0

    # ============= 1. describe_concept =============
    def describe_concept(self, category: str) -> Dict[str, Any]:
        """
        카테고리의 통합 묘사.

        Returns: {
            'category', 'group',
            'parents', 'children',     # CG (있으면)
            'episode_count',            # EM (있으면)
            'similar_concepts',         # AN (있으면)
            'last_seen_distance',       # TM (있으면)
            'in_sa_graph',
        }
        """
        self.describe_count += 1
        if not category:
            return {'category': category, 'error': 'empty'}

        result: Dict[str, Any] = {
            'category': category,
            'group': self._get_group(category),
            'in_sa_graph': category in self.sa.neighbors,
        }

        # CG
        if self.cg is not None:
            try:
                result['parents'] = dict(self.cg.parents(category))
                result['children'] = dict(self.cg.children(category))
            except Exception:
                result['parents'] = {}
                result['children'] = {}
        else:
            result['parents'] = {}
            result['children'] = {}

        # EM
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                count = sum(1 for ep in self.em.episodes.values()
                           if category in getattr(ep, 'categories', set()))
                result['episode_count'] = count
            except Exception:
                result['episode_count'] = 0
        else:
            result['episode_count'] = 0

        # AN
        if self.an is not None:
            try:
                similar = self.an.explain_with_analogy(
                    category, top_k=3, min_similarity=0.3)
                result['similar_concepts'] = [
                    {'category': s['category'],
                     'similarity': s['similarity']}
                    for s in similar
                ]
            except Exception:
                result['similar_concepts'] = []
        else:
            result['similar_concepts'] = []

        # TM
        if self.tm is not None:
            try:
                dist = self.tm.time_distance(category)
                result['last_seen_distance'] = dist
            except Exception:
                result['last_seen_distance'] = None
        else:
            result['last_seen_distance'] = None

        # SA 활성도
        result['current_activation'] = self.sa.activations.get(category, 0.0)

        return result

    def _get_group(self, category: str) -> Optional[str]:
        """SA.CATEGORY_GROUPS에서"""
        if not hasattr(self.sa, 'CATEGORY_GROUPS'):
            return None
        return self.sa.CATEGORY_GROUPS.get(category)

    # ============= 2. list_domains =============
    def list_domains(self) -> Dict[str, List[str]]:
        """
        도메인별 카테고리 클러스터.

        명시적 그룹 (SA.CATEGORY_GROUPS) + 미분류 카테고리들.
        결정론 정렬 (그룹 알파벳, 카테고리 알파벳).

        Returns: {group_name: [category, ...], '_unclassified': [...]}
        """
        self.list_domains_count += 1

        domains: Dict[str, List[str]] = defaultdict(list)

        # CATEGORY_GROUPS 등록된 거
        if hasattr(self.sa, 'CATEGORY_GROUPS'):
            for cat, group in self.sa.CATEGORY_GROUPS.items():
                domains[group].append(cat)

        # SA에 있지만 그룹 미등록인 거
        all_in_sa = set(self.sa.neighbors.keys()) | set(self.sa.activations.keys())
        registered = (set(self.sa.CATEGORY_GROUPS.keys())
                     if hasattr(self.sa, 'CATEGORY_GROUPS') else set())
        unclassified = sorted(all_in_sa - registered)

        if unclassified:
            domains['_unclassified'] = unclassified

        # 정렬 (결정론)
        sorted_domains = {}
        for group in sorted(domains.keys()):
            sorted_domains[group] = sorted(domains[group])
        return sorted_domains

    # ============= 3. categorize =============
    def categorize(self, category: str) -> Dict[str, Any]:
        """
        카테고리의 도메인 추론.

        - 명시적 그룹 있으면 그대로
        - 없으면 인접 카테고리들 다수결
          (SA.neighbors → CG.parents/children → CATEGORY_GROUPS)

        Returns: {'category', 'group', 'inferred': bool, 'evidence'}
        """
        self.categorize_count += 1

        # 1) 명시적
        explicit = self._get_group(category)
        if explicit:
            return {
                'category': category,
                'group': explicit,
                'inferred': False,
                'evidence': 'explicit_group',
            }

        # 2) 인접 카테고리 다수결
        neighbor_groups: List[str] = []

        # SA neighbors
        if category in self.sa.neighbors:
            for n in self.sa.neighbors[category]:
                g = self._get_group(n)
                if g:
                    neighbor_groups.append(g)

        # CG parents + children
        if self.cg is not None:
            try:
                for p in self.cg.parents(category).keys():
                    g = self._get_group(p)
                    if g:
                        neighbor_groups.append(g)
                for c in self.cg.children(category).keys():
                    g = self._get_group(c)
                    if g:
                        neighbor_groups.append(g)
            except Exception:
                pass

        if not neighbor_groups:
            return {
                'category': category,
                'group': None,
                'inferred': False,
                'evidence': 'no_neighbors',
            }

        # 다수결 (동률 시 알파벳)
        counter = Counter(neighbor_groups)
        max_count = max(counter.values())
        top_groups = sorted([g for g, c in counter.items() if c == max_count])

        return {
            'category': category,
            'group': top_groups[0],
            'inferred': True,
            'evidence': f'{max_count}/{len(neighbor_groups)} neighbors',
        }

    # ============= 4. find_schema =============
    def find_schema(self,
                   seed: Optional[str] = None,
                   max_chains: int = 5,
                   min_length: int = 2) -> List[Dict[str, Any]]:
        """
        반복 인과 패턴 추출.

        seed=None이면 CG에서 모든 chain 中 강한 거.
        seed 있으면 seed 등장 chain만.

        Returns: [{'chain': ['A', 'B', 'C'], 'avg_weight': float, 'length': N}]
        """
        self.schema_count += 1

        if self.cg is None:
            return []

        chains: List[Tuple[List[str], float]] = []

        # CG의 모든 (cause, effect) 페어를 시작점으로
        starts: List[Tuple[str, str, float]] = []
        if hasattr(self.cg, 'effects'):
            for cause, effects_dict in self.cg.effects.items():
                for effect, edge_data in effects_dict.items():
                    weight = edge_data.get('weight', 0.0) if isinstance(
                        edge_data, dict) else float(edge_data)
                    starts.append((cause, effect, weight))

        if not starts:
            return []

        # weight 내림차순 (강한 것부터)
        starts.sort(key=lambda x: (-x[2], x[0], x[1]))

        # 각 시작에서 최대 길이 chain 펼침
        for cause, effect, w in starts:
            # seed 필터
            if seed is not None and seed not in (cause, effect):
                continue

            chain = [cause, effect]
            weights = [w]
            visited = {cause, effect}
            current = effect

            # 자식 사슬 펼침
            while True:
                children = self.cg.children(current) if hasattr(
                    self.cg, 'children') else {}
                if not children:
                    break
                # 결정론: weight ↓, 알파벳 ↑
                sorted_kids = sorted(children.items(),
                                    key=lambda kv: (-kv[1], kv[0]))
                next_node = None
                for k, kw in sorted_kids:
                    if k not in visited:
                        next_node = k
                        next_w = kw
                        break
                if next_node is None:
                    break
                chain.append(next_node)
                weights.append(next_w)
                visited.add(next_node)
                current = next_node
                if len(chain) >= 5:  # 최대 길이
                    break

            if len(chain) >= min_length:
                avg_w = sum(weights) / len(weights)
                chains.append((chain, avg_w))

        # 중복 제거 (같은 chain)
        seen_chains: Set[Tuple[str, ...]] = set()
        unique = []
        for chain, avg_w in chains:
            key = tuple(chain)
            if key in seen_chains:
                continue
            seen_chains.add(key)
            unique.append({'chain': chain, 'avg_weight': float(avg_w),
                          'length': len(chain)})

        # 평균 가중치 ↓ 정렬
        unique.sort(key=lambda d: (-d['avg_weight'], d['length'], d['chain']))
        return unique[:max_chains]

    # ============= 5. knowledge_overview =============
    def knowledge_overview(self) -> Dict[str, Any]:
        """
        EVE의 전체 지식 요약.

        Returns: {
            'total_categories', 'classified_categories',
            'domains_count', 'causal_edges', 'episodes',
            'avg_relations_per_category', 'top_connected', ...
        }
        """
        # SA
        all_cats = set(self.sa.neighbors.keys()) | set(self.sa.activations.keys())

        # 도메인
        domains = self.list_domains()
        classified = sum(len(cats) for g, cats in domains.items()
                        if g != '_unclassified')

        result: Dict[str, Any] = {
            'total_categories': len(all_cats),
            'classified_categories': classified,
            'unclassified_categories': len(domains.get('_unclassified', [])),
            'domains_count': len([g for g in domains.keys()
                                 if g != '_unclassified']),
            'domains': {g: len(cats) for g, cats in domains.items()},
        }

        # CG
        if self.cg is not None:
            try:
                result['causal_edges'] = self.cg.edge_count() if hasattr(
                    self.cg, 'edge_count') else 0
                result['causal_nodes'] = self.cg.node_count() if hasattr(
                    self.cg, 'node_count') else 0
            except Exception:
                result['causal_edges'] = 0
                result['causal_nodes'] = 0
        else:
            result['causal_edges'] = 0
            result['causal_nodes'] = 0

        # EM
        if self.em is not None and hasattr(self.em, 'episodes'):
            result['episodes'] = len(self.em.episodes)
        else:
            result['episodes'] = 0

        # 평균 인과 관계 수
        if result['causal_nodes'] > 0:
            result['avg_relations_per_node'] = (
                result['causal_edges'] / result['causal_nodes'])
        else:
            result['avg_relations_per_node'] = 0.0

        # 가장 연결 많은 카테고리 (CG)
        if self.cg is not None and hasattr(self.cg, 'effects') and hasattr(
                self.cg, 'causes'):
            connection_counts: Dict[str, int] = defaultdict(int)
            for c in self.cg.effects.keys():
                connection_counts[c] += len(self.cg.effects[c])
            for c in self.cg.causes.keys():
                connection_counts[c] += len(self.cg.causes[c])
            top = sorted(connection_counts.items(),
                        key=lambda kv: (-kv[1], kv[0]))[:3]
            result['top_connected'] = [
                {'category': c, 'connections': n} for c, n in top
            ]
        else:
            result['top_connected'] = []

        # 활성 카테고리 (현재)
        active = [c for c, v in self.sa.activations.items() if v > 0.1]
        result['currently_active'] = sorted(active)[:10]

        return result

    # ============= 6. relate =============
    def relate(self, a: str, b: str) -> Dict[str, Any]:
        """
        A와 B의 관계 종합 분석.

        Returns: {
            'a', 'b',
            'same_group': bool,
            'group_a', 'group_b',
            'causal': str ('a→b' / 'b→a' / 'transitive_ab' / 'transitive_ba' / None),
            'sa_neighbors': bool,
            'sa_weight': float,
            'shared_episodes': int,
            'analogy_similarity': float,
        }
        """
        self.relate_count += 1

        result: Dict[str, Any] = {'a': a, 'b': b}

        # 그룹 비교
        ga = self._get_group(a)
        gb = self._get_group(b)
        result['group_a'] = ga
        result['group_b'] = gb
        result['same_group'] = (ga is not None and ga == gb)

        # 인과
        causal = None
        if self.cg is not None:
            try:
                if b in self.cg.children(a):
                    causal = f'{a}→{b}'
                elif a in self.cg.children(b):
                    causal = f'{b}→{a}'
                elif hasattr(self.cg, 'is_cause_of'):
                    if self.cg.is_cause_of(a, b):
                        causal = f'transitive_{a}→{b}'
                    elif self.cg.is_cause_of(b, a):
                        causal = f'transitive_{b}→{a}'
            except Exception:
                pass
        result['causal'] = causal

        # SA neighbors
        sa_w = 0.0
        sa_neighbors = False
        if hasattr(self.sa, 'neighbors'):
            sa_neighbors = (a in self.sa.neighbors
                          and b in self.sa.neighbors.get(a, set()))
        if hasattr(self.sa, 'get_weight'):
            try:
                sa_w = self.sa.get_weight(a, b)
            except Exception:
                pass
        result['sa_neighbors'] = sa_neighbors
        result['sa_weight'] = float(sa_w)

        # 공유 일화
        shared_episodes = 0
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                for ep in self.em.episodes.values():
                    cats = getattr(ep, 'categories', set())
                    if a in cats and b in cats:
                        shared_episodes += 1
            except Exception:
                pass
        result['shared_episodes'] = shared_episodes

        # 유추 유사도
        analogy_sim = 0.0
        if self.an is not None:
            try:
                analogy_sim = self.an.structural_similarity(a, b)
            except Exception:
                pass
        result['analogy_similarity'] = float(analogy_sim)

        # 종합 strength
        score = 0.0
        if result['same_group']:
            score += 0.3
        if causal:
            score += 0.4 if 'transitive' not in causal else 0.2
        score += min(sa_w, 1.0) * 0.2
        score += min(shared_episodes / 5.0, 1.0) * 0.1
        score += analogy_sim * 0.2
        result['strength'] = float(np.clip(score, 0.0, 1.0))

        return result

    # ============= tick =============
    def tick(self, dt: float):
        """시간 누적."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'describe_count': self.describe_count,
            'list_domains_count': self.list_domains_count,
            'categorize_count': self.categorize_count,
            'schema_count': self.schema_count,
            'relate_count': self.relate_count,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"WorldModel(t={s['time']:.1f}s, "
                f"describes={s['describe_count']}, "
                f"relates={s['relate_count']})")
