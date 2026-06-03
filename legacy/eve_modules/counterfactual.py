"""
EVE v32 - C3: Counterfactual ("만약 X 했다면?")
================================================

Pearl 3단계 인과 추론 中 *3단계 (Counterfactual)*.
EVE의 "만약?" 사고 + 후회/안도 정서.

학계:
- Lewis 1973 — Counterfactuals (closest possible world)
- Pearl 2009 — Counterfactual reasoning (3단계 인과)
- Byrne 2005 — Rational Imagination (반사실적 사고)
- Roese 1997 — Counterfactual Thinking & Emotions (regret, relief)
- Kahneman & Tversky 1982 — Simulation Heuristic

핵심 메커니즘:

A. what_if_not(X):
   "X가 일어나지 않았다면?"
   - CausalGraph에서 X의 descendants 가져옴
   - 각 자식의 *다른 부모들* 확인 (X 외 cause)
   - X만 부모인 자식 → "사라짐"
   - 다른 부모도 있는 자식 → "살아남음 (영향 약해짐)"

B. what_if_instead(X, alternative):
   "X 대신 alternative였다면?"
   - alternative의 descendants 시뮬
   - X의 descendants 중 alternative 영향 받는 건 유지
   - X 고유 descendants는 사라짐

C. regret_or_relief(action_record):
   실제 outcome vs counterfactual outcome 비교.
   - 실제 outcome < counterfactual → regret (후회)
   - 실제 outcome > counterfactual → relief (안도)
   - regret → cortisol ↑ (호르몬 자극, 학습 신호)
   - relief → serotonin/dopamine ↑

D. closest_world(category):
   Lewis 1973: 가장 가깝게 다른 세계.
   - SA 가중치 + CG 인과 거리 종합
   - top N "비슷하면서도 다른" 카테고리

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론 (같은 그래프 + 같은 카테고리 → 같은 시뮬)
- 호르몬 변조 (regret/relief가 호르몬 자극)
- 카테고리 단위
- 모듈 chain (CG + SA + HS + AG + AI 통합)
- read-only on CG/SA (시뮬은 가상, 실제 그래프 변경 X)

핵심 함수 6개:
1. what_if_not(X, depth)           - X 없는 가상 세계
2. what_if_instead(X, alt, depth)  - X 대체 가상 세계
3. counterfactual_action(action, alt) - 다른 행동의 가상 결과
4. regret_or_relief(action_record) - 후회/안도 시그널
5. closest_world(category, top_n)  - Lewis 가장 가까운 다른 세계
6. tick(dt)

Author: 김민석 + Claude v32 (C3)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque


class Counterfactual:
    """EVE v32 C3 Counterfactual — Pearl 3단계 인과"""

    # 시뮬 깊이
    DEFAULT_DEPTH = 3

    # regret/relief 임계 (outcome 차이)
    REGRET_THRESHOLD = 0.3   # outcome 점수 차이가 이 이상이면 신호

    # outcome 라벨 → 점수 (수치 비교용)
    OUTCOME_SCORES = {
        'good':    1.0,
        'neutral': 0.0,
        'bad':    -1.0,
    }

    # 호르몬 자극 강도
    REGRET_CORTISOL_BOOST = 0.10
    RELIEF_SEROTONIN_BOOST = 0.08
    RELIEF_DOPAMINE_BOOST = 0.05

    def __init__(self,
                 causal_graph,
                 spreading_activation,
                 hormone_system=None,
                 agency=None,
                 active_inference=None,
                 self_doubt=None,
                 history_size: int = 50):
        """
        Args:
            causal_graph: 필수 (인과 시뮬)
            spreading_activation: 필수 (대안 카테고리 추출)
            나머지: 옵션 (regret/relief 호르몬, action 평가용)
        """
        self.cg = causal_graph
        self.sa = spreading_activation
        self.hs = hormone_system
        self.ag = agency
        self.ai = active_inference
        self.sd = self_doubt

        self.simulations: deque = deque(maxlen=history_size)
        self.time = 0.0

        self.simulate_count = 0
        self.regret_count = 0
        self.relief_count = 0

    # ============= 1. what_if_not =============
    def what_if_not(self, category: str,
                   depth: int = None) -> Dict[str, Any]:
        """
        "X가 일어나지 않았다면?" — X의 descendants 중 어떤 게 사라지나.

        알고리즘 (BFS depth 순서로 결정론 보장):
        - layer 0: X 자체 (sink)
        - layer 1: X의 직접 자식들 처리
            - 부모가 X뿐이거나, 다른 부모가 모두 removed면 → removed
            - 다른 살아있는 부모 있으면 → weakened
        - layer 2: layer 1에서 살아있는 것들 + 새 자식들 처리
        - ... depth 만큼 반복

        같은 layer 내에서는 알파벳 순으로 처리 (결정론).
        """
        if depth is None:
            depth = self.DEFAULT_DEPTH

        self.simulate_count += 1

        removed: Set[str] = set()
        weakened: Dict[str, float] = {}

        # BFS layer-by-layer
        current_layer = {category}
        all_visited = {category}

        for d in range(depth):
            # 다음 layer = current_layer 자식들 (visited 제외)
            next_layer = set()
            for cur in current_layer:
                for child in self.cg.children(cur).keys():
                    if child not in all_visited and child != category:
                        next_layer.add(child)

            if not next_layer:
                break

            # 결정론적 처리 (알파벳 순)
            for desc in sorted(next_layer):
                parents_dict = self.cg.parents(desc)
                # 살아있는 부모들 (X 본인이 아니고, 이미 removed에 들어간 것도 아님)
                other_parents = {
                    p: w for p, w in parents_dict.items()
                    if p != category and p not in removed
                }

                if not other_parents:
                    removed.add(desc)
                else:
                    # X의 영향만 약화 (X가 부모면 X 가중치만큼)
                    lost_weight = parents_dict.get(category, 0.0)
                    if lost_weight > 0:
                        weakened[desc] = lost_weight
                    # X가 부모 아니지만 모든 부모가 removed라면? 위 not other_parents에서 처리됨

                all_visited.add(desc)

            current_layer = next_layer

        descendants = all_visited - {category}
        unaffected = descendants - removed - set(weakened.keys())

        result = {
            'absent': category,
            'removed': removed,
            'weakened': weakened,
            'unaffected': unaffected,
            'depth': depth,
            'time': self.time,
        }
        self.simulations.append({'type': 'what_if_not', **result})
        return result

    # ============= 2. what_if_instead =============
    def what_if_instead(self, category: str,
                       alternative: str,
                       depth: int = None) -> Dict[str, Any]:
        """
        "X 대신 alternative였다면?"

        - alternative의 descendants를 'gained'
        - X의 descendants 중 alternative와 공통이 아닌 건 'lost'
        - 공통은 'shared' (둘 다 만들어냄)
        """
        if depth is None:
            depth = self.DEFAULT_DEPTH
        if not category or not alternative or category == alternative:
            return {
                'replaced': category,
                'with': alternative,
                'gained': set(),
                'lost': set(),
                'shared': set(),
                'depth': depth,
            }

        self.simulate_count += 1

        x_desc = self.cg.descendants(category, max_depth=depth)
        alt_desc = self.cg.descendants(alternative, max_depth=depth)

        gained = alt_desc - x_desc  # 새로 생기는 결과
        lost = x_desc - alt_desc    # 잃는 결과
        shared = x_desc & alt_desc  # 둘 다 만드는 결과

        result = {
            'replaced': category,
            'with': alternative,
            'gained': gained,
            'lost': lost,
            'shared': shared,
            'depth': depth,
            'time': self.time,
        }
        self.simulations.append({'type': 'what_if_instead', **result})
        return result

    # ============= 3. counterfactual_action =============
    def counterfactual_action(self,
                             actual_action: str,
                             alternative_action: str) -> Dict[str, Any]:
        """
        "다른 행동을 했다면 결과가 어땠을까?"

        Agency 행동 비교 시뮬:
        - actual_action의 결과 카테고리들 (CG.descendants)
        - alternative_action의 결과 카테고리들
        - 차이를 outcome 점수로 (긍정 그룹 vs 부정 그룹)

        Returns: {
            'actual_action', 'alternative_action',
            'actual_outcomes', 'alternative_outcomes',
            'outcome_score': float,      # alternative - actual (양수면 alt 좋음)
            'recommendation': str        # 'try_alternative' / 'stay' / 'similar'
        }
        """
        sim = self.what_if_instead(actual_action, alternative_action)

        # outcome 평가: gained/lost 카테고리들이 reward/threat 그룹인지
        actual_score = self._evaluate_outcome_categories(sim['lost'] | sim['shared'])
        alt_score = self._evaluate_outcome_categories(sim['gained'] | sim['shared'])

        outcome_diff = alt_score - actual_score

        if outcome_diff > 0.3:
            recommendation = 'try_alternative'
        elif outcome_diff < -0.3:
            recommendation = 'stay'
        else:
            recommendation = 'similar'

        return {
            'actual_action': actual_action,
            'alternative_action': alternative_action,
            'actual_outcomes': sim['lost'] | sim['shared'],
            'alternative_outcomes': sim['gained'] | sim['shared'],
            'actual_score': actual_score,
            'alternative_score': alt_score,
            'outcome_score': outcome_diff,
            'recommendation': recommendation,
            'time': self.time,
        }

    def _evaluate_outcome_categories(self, categories: Set[str]) -> float:
        """카테고리 set의 outcome 점수 (-1 ~ +1)"""
        if not categories:
            return 0.0
        if not hasattr(self.sa, 'CATEGORY_GROUPS'):
            return 0.0

        positive_groups = {'reward', 'social', 'rest'}
        negative_groups = {'threat', 'aversion'}

        score = 0.0
        for cat in categories:
            group = self.sa.CATEGORY_GROUPS.get(cat)
            if group in positive_groups:
                score += 1.0
            elif group in negative_groups:
                score -= 1.0
        # 정규화 (카테고리 수로)
        return float(np.clip(score / len(categories), -1.0, 1.0))

    # ============= 4. regret_or_relief =============
    def regret_or_relief(self,
                        action_record: Optional[Dict] = None,
                        alternative_action: Optional[str] = None
                        ) -> Optional[Dict[str, Any]]:
        """
        실제 행동 outcome vs counterfactual outcome 비교 → regret/relief.

        Args:
            action_record: Agency.actions 항목 (None이면 last_action)
            alternative_action: 비교할 대안 카테고리 (None이면 closest_world top1)

        Returns:
            {'signal': 'regret'|'relief'|'neutral',
             'magnitude': float,
             'actual_outcome', 'counterfactual_outcome', ...}
            또는 None (action_record 없거나 outcome 평가 불가)
        """
        # action_record 가져오기
        if action_record is None and self.ag is not None and self.ag.actions:
            action_record = self.ag.actions[-1]
        if not action_record:
            return None

        actual_action = action_record.get('action')
        actual_outcome = action_record.get('outcome')  # 'good'/'bad'/'neutral'/None

        if not actual_action:
            return None

        # 실제 outcome 점수
        actual_score = self.OUTCOME_SCORES.get(actual_outcome, 0.0)
        if actual_outcome is None:
            # outcome 미정 — counterfactual 평가만 가능
            actual_score = self._evaluate_outcome_categories(
                self.cg.descendants(actual_action, max_depth=2))

        # alternative 결정
        if alternative_action is None:
            alts = self.closest_world(actual_action, top_n=1)
            if not alts:
                return None
            alternative_action = alts[0][0]

        # counterfactual 평가
        cf = self.counterfactual_action(actual_action, alternative_action)
        cf_score = cf['alternative_score']

        # signal 결정
        diff = actual_score - cf_score
        magnitude = abs(diff)

        if magnitude < self.REGRET_THRESHOLD:
            signal = 'neutral'
        elif diff < 0:
            # 실제 < counterfactual → 다른 거 했으면 좋았을 텐데 → regret
            signal = 'regret'
            self.regret_count += 1
            self._stimulate_regret(magnitude)
        else:
            # 실제 > counterfactual → 다행히 이거 했네 → relief
            signal = 'relief'
            self.relief_count += 1
            self._stimulate_relief(magnitude)

        result = {
            'signal': signal,
            'magnitude': magnitude,
            'actual_action': actual_action,
            'actual_outcome': actual_outcome,
            'actual_score': actual_score,
            'counterfactual_action': alternative_action,
            'counterfactual_score': cf_score,
            'time': self.time,
        }
        self.simulations.append({'type': 'regret_or_relief', **result})
        return result

    def _stimulate_regret(self, magnitude: float):
        """regret → cortisol 자극 (학습 신호)"""
        if self.hs is None:
            return
        cort = self.hs.hormones.get('cortisol')
        if cort is not None and hasattr(cort, 'stimulate'):
            cort.stimulate(self.REGRET_CORTISOL_BOOST * magnitude)
        elif cort is not None:
            cort.level = float(np.clip(
                cort.level + self.REGRET_CORTISOL_BOOST * magnitude,
                0.0, 1.0))

    def _stimulate_relief(self, magnitude: float):
        """relief → serotonin/dopamine 자극"""
        if self.hs is None:
            return
        for h_name, boost in [('serotonin', self.RELIEF_SEROTONIN_BOOST),
                              ('dopamine', self.RELIEF_DOPAMINE_BOOST)]:
            h = self.hs.hormones.get(h_name)
            if h is not None:
                if hasattr(h, 'stimulate'):
                    h.stimulate(boost * magnitude)
                else:
                    h.level = float(np.clip(
                        h.level + boost * magnitude, 0.0, 1.0))

    # ============= 5. closest_world (Lewis 1973) =============
    def closest_world(self, category: str,
                     top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Lewis "가장 가까운 가능 세계" — category와 가장 비슷한 *다른* 카테고리.

        유사도 = SA Hebbian 가중치 + CG 공유 부모/자식 비율.

        Returns: [(alternative, similarity), ...] 정렬
        """
        # 1) SA에서 직접 이웃들 (Hebbian)
        if not hasattr(self.sa, 'neighbors'):
            return []

        candidates: Dict[str, float] = {}

        # SA 이웃들
        for neighbor in self.sa.neighbors.get(category, set()):
            if neighbor == category:
                continue
            sa_w = self.sa.get_weight(category, neighbor) if hasattr(
                self.sa, 'get_weight') else 0.0
            candidates[neighbor] = sa_w * 0.5

        # 2) CG 공유 부모 (같은 cause)
        my_parents = set(self.cg.parents(category).keys())
        if my_parents:
            for parent in my_parents:
                # 같은 부모를 가진 다른 카테고리 (siblings)
                siblings = set(self.cg.children(parent).keys())
                for sib in siblings:
                    if sib == category:
                        continue
                    candidates[sib] = candidates.get(sib, 0.0) + 0.3

        # 3) 결정론적 정렬 (점수 ↓, 알파벳 ↑)
        items = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:top_n]

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
            'simulate_count': self.simulate_count,
            'regret_count': self.regret_count,
            'relief_count': self.relief_count,
            'simulations_recorded': len(self.simulations),
            'last_simulation_type': (self.simulations[-1].get('type')
                                    if self.simulations else None),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Counterfactual(t={s['time']:.1f}s, "
                f"sims={s['simulate_count']}, "
                f"regret/relief={s['regret_count']}/{s['relief_count']})")
