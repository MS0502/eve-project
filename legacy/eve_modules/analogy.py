"""
EVE v32 - C4: Analogy (유추)
==============================

EVE의 유추 능력. "A:B::C:D" — 관계 매핑.

학계:
- Gentner 1983 — Structure-Mapping Theory (관계 vs 속성)
- Hofstadter 2001 — Analogy as the Core of Cognition
- Holyoak & Thagard 1989 — Analogical Mapping (Multi-constraint)
- Gentner & Markman 1997 — Structure mapping in analogy
- Penn, Holyoak & Povinelli 2008 — Darwin's Mistake

핵심 통찰 (Gentner):
- 표면적 유사 (속성) ≠ 관계적 유사 (구조)
- 진정한 유추는 관계 패턴이 같은 것
- 예: "전자가 핵 주위 돈다" :: "행성이 태양 주위 돈다"
  → 표면 다르지만, '돈다' 관계가 같음

EVE 매핑:
- SA 그룹 = 속성 (표면)
- CG 인과 관계 = 관계 (구조)
- analogy는 *CG 관계* 매칭이 우선

핵심 메커니즘:

A. find_analogy(a, b, c):
   "a:b :: c:?" — D를 찾는다.
   알고리즘:
   1. a→b 관계 추출 (CG에서 a의 children 中 b면 → 'cause-effect')
   2. c의 children에서 b와 비슷한 weight의 자식 찾기
   3. 같은 그룹의 카테고리 우선

B. map_relation(source_pair, target_anchor):
   source가 어떤 관계인지 분석 → target에서 같은 관계 적용.

C. structural_similarity(x, y):
   x와 y의 *인과 구조* 비교.
   - parents/children 그룹 분포
   - relation strength 패턴
   - 같은 그룹 children 비율

D. explain_with_analogy(target, top_k):
   target과 가장 비슷한 인과 구조 가진 카테고리들 찾기.
   "X는 Y와 비슷해" 형태.

E. complete_proportion(a, b, c):
   언어적 단순 형태 (find_analogy 별칭).

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론 (점수 ↓, 알파벳 ↑ 정렬)
- 카테고리 단위
- 호르몬 변조: 옵션 (acetylcholine ↑ → 더 깊은 매핑)
- Read-only on CG/SA

핵심 함수 6:
1. find_analogy(a, b, c)
2. map_relation(source_pair, target_anchor)
3. structural_similarity(x, y)
4. explain_with_analogy(target, top_k)
5. complete_proportion(a, b, c)  # find_analogy 별칭
6. tick(dt)

Author: 김민석 + Claude v32 (C4)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict


class Analogy:
    """EVE v32 C4 Analogy — Gentner Structure-Mapping"""

    # 매핑 점수 가중치
    W_RELATION_TYPE = 0.5      # 관계 타입 일치 (parent/child)
    W_WEIGHT_SIMILARITY = 0.3  # 가중치 유사도
    W_GROUP_MATCH = 0.2        # 같은 그룹

    # 최소 유사도 임계
    MIN_SIMILARITY = 0.1

    def __init__(self,
                 causal_graph,
                 spreading_activation,
                 hormone_system=None):
        """
        Args:
            causal_graph: 필수 (관계 추출)
            spreading_activation: 필수 (그룹 + Hebbian)
            hormone_system: 옵션 (학습률 변조)
        """
        self.cg = causal_graph
        self.sa = spreading_activation
        self.hs = hormone_system

        self.time = 0.0

        # 통계
        self.find_count = 0
        self.explain_count = 0
        self.similarity_count = 0

    # ============= 1. find_analogy =============
    def find_analogy(self, a: str, b: str, c: str,
                    top_n: int = 1) -> List[Tuple[str, float]]:
        """
        "a:b :: c:?" — D 찾기.

        알고리즘:
        1. a→b 관계 추출 (CG에서)
           - a의 children에 b 있나? → 'a-causes-b'
           - 아니면 a의 parents에 b 있나? → 'b-causes-a'
        2. c의 같은 종류 이웃에서 후보 찾기
        3. 점수: 관계 타입 일치 + weight 유사도 + 그룹 일치

        Returns: [(d, score), ...] top_n
        """
        self.find_count += 1

        if not (a and b and c) or a == b or c == b:
            return []

        # a→b 관계 결정
        relation_type = self._determine_relation(a, b)
        if relation_type is None:
            return []

        # a→b weight
        ab_weight = self._get_relation_weight(a, b, relation_type)

        # c의 같은 종류 이웃들
        if relation_type == 'cause-effect':
            c_neighbors = self.cg.children(c)
        elif relation_type == 'effect-cause':
            c_neighbors = self.cg.parents(c)
        else:
            return []

        if not c_neighbors:
            return []

        # b의 그룹 (있으면)
        b_group = self._get_group(b)

        # 점수 계산
        scores: Dict[str, float] = {}
        for d, d_weight in c_neighbors.items():
            if d == a or d == c:
                continue
            score = 0.0

            # 1) 관계 타입 일치 (이미 같은 종류 이웃이라 +)
            score += self.W_RELATION_TYPE

            # 2) weight 유사도 (1 - |ab - cd| / 1.0)
            weight_diff = abs(ab_weight - d_weight)
            score += self.W_WEIGHT_SIMILARITY * (1.0 - weight_diff)

            # 3) 그룹 일치 (b의 그룹과 d의 그룹 같으면)
            d_group = self._get_group(d)
            if b_group and d_group and b_group == d_group:
                score += self.W_GROUP_MATCH

            scores[d] = score

        if not scores:
            return []

        # 임계 미달 제거
        scores = {d: s for d, s in scores.items() if s >= self.MIN_SIMILARITY}
        if not scores:
            return []

        # 결정론 정렬 (점수 ↓, 알파벳 ↑)
        sorted_scores = sorted(scores.items(),
                              key=lambda kv: (-kv[1], kv[0]))
        return sorted_scores[:top_n]

    def _determine_relation(self, a: str, b: str) -> Optional[str]:
        """a→b 관계 타입"""
        if b in self.cg.children(a):
            return 'cause-effect'
        if b in self.cg.parents(a):
            return 'effect-cause'
        return None

    def _get_relation_weight(self, a: str, b: str,
                            relation: str) -> float:
        """a→b 관계 가중치"""
        if relation == 'cause-effect':
            return self.cg.children(a).get(b, 0.0)
        elif relation == 'effect-cause':
            return self.cg.parents(a).get(b, 0.0)
        return 0.0

    def _get_group(self, category: str) -> Optional[str]:
        """카테고리 그룹 (SA.CATEGORY_GROUPS)"""
        if not hasattr(self.sa, 'CATEGORY_GROUPS'):
            return None
        return self.sa.CATEGORY_GROUPS.get(category)

    # ============= 2. map_relation =============
    def map_relation(self,
                    source_pair: Tuple[str, str],
                    target_anchor: str) -> Optional[Dict[str, Any]]:
        """
        source 관계를 target 도메인으로 매핑.

        Args:
            source_pair: (a, b) — 학습된 관계
            target_anchor: c — 매핑 시작점

        Returns: {
            'source': (a, b),
            'target': (c, d),
            'relation_type': str,
            'similarity': float,
        } or None
        """
        a, b = source_pair
        c = target_anchor

        result = self.find_analogy(a, b, c, top_n=1)
        if not result:
            return None

        d, score = result[0]
        relation_type = self._determine_relation(a, b)

        return {
            'source': (a, b),
            'target': (c, d),
            'relation_type': relation_type,
            'similarity': float(score),
            'time': self.time,
        }

    # ============= 3. structural_similarity =============
    def structural_similarity(self, x: str, y: str) -> float:
        """
        두 카테고리의 인과 구조 유사도.

        측정:
        - parents 그룹 분포
        - children 그룹 분포
        - 직접 weight 패턴

        Returns: 유사도 [0, 1]
        """
        self.similarity_count += 1
        if not x or not y or x == y:
            return 1.0 if x == y else 0.0

        # x와 y의 부모/자식 그룹 분포
        x_parents = self.cg.parents(x)
        y_parents = self.cg.parents(y)
        x_children = self.cg.children(x)
        y_children = self.cg.children(y)

        # 그룹 카운트
        x_p_groups = self._group_distribution(x_parents.keys())
        y_p_groups = self._group_distribution(y_parents.keys())
        x_c_groups = self._group_distribution(x_children.keys())
        y_c_groups = self._group_distribution(y_children.keys())

        # Jaccard 유사도 (parents + children 그룹)
        sim_p = self._dist_similarity(x_p_groups, y_p_groups)
        sim_c = self._dist_similarity(x_c_groups, y_c_groups)

        # 직접 부모/자식 수 비교 (구조적 차이)
        if (len(x_parents) + len(y_parents)) > 0:
            ratio_p = min(len(x_parents), len(y_parents)) / max(
                len(x_parents), len(y_parents), 1)
        else:
            ratio_p = 0.0
        if (len(x_children) + len(y_children)) > 0:
            ratio_c = min(len(x_children), len(y_children)) / max(
                len(x_children), len(y_children), 1)
        else:
            ratio_c = 0.0

        # 종합
        total = (sim_p * 0.3 + sim_c * 0.3
                + ratio_p * 0.2 + ratio_c * 0.2)
        return float(np.clip(total, 0.0, 1.0))

    def _group_distribution(self, categories) -> Dict[str, int]:
        """카테고리들의 그룹 카운트"""
        dist: Dict[str, int] = defaultdict(int)
        for cat in categories:
            g = self._get_group(cat)
            if g:
                dist[g] += 1
            else:
                dist['_none'] += 1
        return dict(dist)

    def _dist_similarity(self, da: Dict[str, int],
                         db: Dict[str, int]) -> float:
        """두 분포의 유사도 (Jaccard 변형)"""
        if not da and not db:
            return 0.0
        all_groups = set(da.keys()) | set(db.keys())
        if not all_groups:
            return 0.0

        common = 0
        total = 0
        for g in all_groups:
            ca = da.get(g, 0)
            cb = db.get(g, 0)
            common += min(ca, cb)
            total += max(ca, cb)

        return common / total if total > 0 else 0.0

    # ============= 4. explain_with_analogy =============
    def explain_with_analogy(self, target: str,
                            top_k: int = 3,
                            min_similarity: float = 0.3
                            ) -> List[Dict[str, Any]]:
        """
        "target과 비슷한 게 뭐지?" — 인과 구조 비슷한 카테고리 찾기.

        후보 풀: CG에 있는 모든 카테고리 (target 외)
        각 후보에 structural_similarity 계산
        top_k 반환.

        Returns: [{'category', 'similarity', 'shared_relations'}, ...]
        """
        self.explain_count += 1

        # 후보 풀: CG에 등장한 모든 카테고리
        all_cats: Set[str] = set()
        if hasattr(self.cg, 'effects'):
            all_cats.update(self.cg.effects.keys())
        if hasattr(self.cg, 'causes'):
            all_cats.update(self.cg.causes.keys())

        candidates = all_cats - {target}
        if not candidates:
            return []

        # 각 후보 유사도
        scored = []
        for cand in candidates:
            sim = self.structural_similarity(target, cand)
            if sim >= min_similarity:
                # 공유 관계 분석 (간단히)
                target_parents = set(self.cg.parents(target).keys())
                cand_parents = set(self.cg.parents(cand).keys())
                target_children = set(self.cg.children(target).keys())
                cand_children = set(self.cg.children(cand).keys())

                shared = {
                    'shared_parents': target_parents & cand_parents,
                    'shared_children': target_children & cand_children,
                }

                scored.append({
                    'category': cand,
                    'similarity': sim,
                    'shared_relations': shared,
                })

        # 결정론 정렬
        scored.sort(key=lambda d: (-d['similarity'], d['category']))
        return scored[:top_k]

    # ============= 5. complete_proportion =============
    def complete_proportion(self, a: str, b: str,
                           c: str) -> Optional[str]:
        """
        "A:B::C:?" 단순 형태 (find_analogy의 별칭, 첫 결과만).
        """
        result = self.find_analogy(a, b, c, top_n=1)
        if result:
            return result[0][0]
        return None

    # ============= 6. tick =============
    def tick(self, dt: float):
        """시간 누적."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'find_count': self.find_count,
            'explain_count': self.explain_count,
            'similarity_count': self.similarity_count,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Analogy(t={s['time']:.1f}s, "
                f"find={s['find_count']}, "
                f"explain={s['explain_count']})")
