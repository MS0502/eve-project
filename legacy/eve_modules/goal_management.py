"""
EVE v32 - C6: GoalManagement (목표 추구)
==========================================

EVE의 "원하는 것" 모델. 디지털 거주자의 의지.

학계:
- Schmidhuber 2010 — Curiosity & Goal-Driven AI
- Carver & Scheier 1998 — Self-Regulation Theory (control theory of behavior)
- Botvinick et al. 2009 — Hierarchical Reinforcement Learning
- Murphy 1999 — Goal Pursuit (TOTE: Test-Operate-Test-Exit)
- Frankfurt 1971 — Hierarchical Will (1차 vs 2차 욕구, 이미 B6에서)
- Locke & Latham 1990 — Goal Setting Theory

핵심 메커니즘:

A. 목표 표현:
   Goal {
       id: str (자동),
       category: str (도달하고 싶은 카테고리),
       priority: float [0, 1],
       deadline: float (시뮬 시간, None=무기한),
       status: 'active' | 'completed' | 'abandoned' | 'expired',
       progress: float [0, 1],
       created, last_evaluated, completed_at,
       source: 'manual' | 'need' (DS 자동),
   }

B. 자동 목표 제안 (need → goal):
   DS.dominant_need가 강하면 NEED_TO_GOAL_CAT 매핑으로 자동 생성.
   - fatigue → '편안' / '잠'
   - warmth → '대화' / '함께'
   - energy → '기쁨' / '성취'
   ...

C. Action 추천 (suggest_action):
   목표 X 달성하려면 → CG.parents(X) 중 가장 강한 cause를 try.
   Counterfactual로 다른 선택 대비 outcome 점수 비교.

D. Progress 평가:
   - SA.activations[goal_cat] (현재 활성)
   - EM 최근 일화에서 goal_cat 빈도 (최근 30개)
   - 가중 합

E. 달성 시 보상:
   complete_goal → dopamine + endorphin 자극
   abandon_goal → cortisol 약하게 (실패 학습)

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론
- 호르몬 변조 (priority가 호르몬 영향, 달성/실패가 호르몬 자극)
- 카테고리 단위
- 모듈 chain (DS + SA + CG + CF + AG + EM 통합)

핵심 함수:
1. goal_set(category, priority, deadline) → goal_id
2. propose_goal_from_need() → 자동 목표 또는 None
3. active_goals(top_n) → priority 순 정렬
4. evaluate_progress(goal_id) → progress dict
5. suggest_action(goal_id) → 추천 행동 카테고리
6. complete_goal(goal_id) → 달성 + 보상
7. abandon_goal(goal_id, reason) → 포기
8. tick(dt) → 만료 체크 + 자동 평가

Author: 김민석 + Claude v32 (C6)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Goal:
    """단일 목표"""
    id: str
    category: str
    priority: float = 0.5
    deadline: Optional[float] = None  # 시뮬 시간, None=무기한
    status: str = 'active'  # active/completed/abandoned/expired
    progress: float = 0.0
    created: float = 0.0
    last_evaluated: float = 0.0
    completed_at: Optional[float] = None
    source: str = 'manual'  # manual / need
    abandon_reason: Optional[str] = None


class GoalManagement:
    """EVE v32 C6 GoalManagement"""

    # need → 후보 카테고리 (우선순위 순)
    NEED_TO_GOAL_CATS = {
        'fatigue':              ['편안', '잠', '휴식'],
        'tension':              ['편안', '안전'],
        'mental_load':          ['휴식', '편안'],
        'energy':               ['기쁨', '성취', '재밌다'],
        'warmth':               ['대화', '함께', '민석'],
        'clarity':              ['집중', '주의'],
        'emotional_intensity':  ['편안', '안전'],
        'activity_level':       ['휴식'],
    }

    # 진행도 ≥ 이 값이면 자동 complete
    AUTO_COMPLETE_THRESHOLD = 0.75

    # 진행도 ≥ 이 값이면 자동 abandon (deadline 임박 + 너무 낮은 progress)
    AUTO_ABANDON_THRESHOLD = 0.05

    # 보상/페널티 강도
    COMPLETE_DOPAMINE = 0.15
    COMPLETE_ENDORPHIN = 0.10
    ABANDON_CORTISOL = 0.05  # 약한 실패 학습

    # 자동 평가 간격 (tick 단위)
    EVAL_INTERVAL_TICKS = 10

    # 자동 propose 간격
    PROPOSE_INTERVAL_TICKS = 50

    # 동시 active 목표 max
    MAX_ACTIVE_GOALS = 5

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 digital_somatic=None,
                 causal_graph=None,
                 counterfactual=None,
                 episodic_memory=None,
                 agency=None,
                 history_size: int = 50):
        """
        Args:
            spreading_activation: 필수 (progress 평가)
            나머지: 옵션 (None 안전)
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.ds = digital_somatic
        self.cg = causal_graph
        self.cf = counterfactual
        self.em = episodic_memory
        self.ag = agency

        self.goals: Dict[str, Goal] = {}      # active goals
        self.history: deque = deque(maxlen=history_size)  # completed/abandoned
        self.time = 0.0
        self.tick_count = 0
        self._next_goal_id = 1

        # 통계
        self.set_count = 0
        self.completed_count = 0
        self.abandoned_count = 0
        self.expired_count = 0
        self.suggested_count = 0
        self.proposed_count = 0  # propose_goal_from_need 트리거됨

    # ============= 1. goal_set =============
    def goal_set(self, category: str,
                priority: float = 0.5,
                deadline_offset: Optional[float] = None,
                source: str = 'manual') -> Optional[str]:
        """
        목표 등록.

        Args:
            category: 도달하고 싶은 카테고리
            priority: [0, 1]
            deadline_offset: 현재 시간 기준 N초 후 (None=무기한)
            source: 'manual' / 'need' / 'external'

        Returns: goal_id 또는 None (max 초과)
        """
        if not category:
            return None

        # 같은 카테고리의 active goal이 이미 있으면 priority 강화만
        for g in self.goals.values():
            if g.category == category and g.status == 'active':
                g.priority = float(np.clip(max(g.priority, priority) + 0.05,
                                          0.0, 1.0))
                return g.id

        # max 초과 → 가장 낮은 priority abandon
        active_count = sum(1 for g in self.goals.values()
                          if g.status == 'active')
        if active_count >= self.MAX_ACTIVE_GOALS:
            weakest = min(
                [g for g in self.goals.values() if g.status == 'active'],
                key=lambda g: g.priority
            )
            if weakest.priority < priority:
                self.abandon_goal(weakest.id, reason='outranked')
            else:
                return None  # 새 거가 더 약함, 거부

        deadline = (self.time + deadline_offset
                   if deadline_offset is not None else None)

        gid = f'goal_{self._next_goal_id:04d}'
        self._next_goal_id += 1

        goal = Goal(
            id=gid,
            category=category,
            priority=float(np.clip(priority, 0.0, 1.0)),
            deadline=deadline,
            status='active',
            progress=0.0,
            created=self.time,
            last_evaluated=self.time,
            source=source,
        )
        self.goals[gid] = goal
        self.set_count += 1
        return gid

    # ============= 2. propose_goal_from_need =============
    def propose_goal_from_need(self) -> Optional[str]:
        """
        DS dominant_need 강하면 자동 목표 생성.

        Returns: goal_id 또는 None
        """
        if self.ds is None:
            return None

        dominant = self.ds.get_dominant_need()
        if not dominant:
            return None

        need_name, strength = dominant
        if strength < 0.5:
            return None  # 너무 약하면 skip

        # need → 후보 카테고리들 中 첫 번째 (우선순위 순)
        candidates = self.NEED_TO_GOAL_CATS.get(need_name, [])
        if not candidates:
            return None

        # 그 중 SA에 있는 거 우선 (없으면 첫 번째)
        chosen = None
        for cat in candidates:
            if cat in self.sa.neighbors or cat in self.sa.activations:
                chosen = cat
                break
        if chosen is None:
            chosen = candidates[0]

        # 같은 카테고리 active goal 있으면 priority 강화 (goal_set에서 처리)
        gid = self.goal_set(chosen, priority=strength, source='need')
        if gid:
            self.proposed_count += 1
        return gid

    # ============= 3. active_goals =============
    def active_goals(self, top_n: Optional[int] = None) -> List[Goal]:
        """active 목표들 priority 순 (동점 시 created 빠른 순 — 결정론)"""
        active = [g for g in self.goals.values() if g.status == 'active']
        # 정렬: priority ↓, created ↑, id ↑
        active.sort(key=lambda g: (-g.priority, g.created, g.id))
        if top_n is not None:
            return active[:top_n]
        return active

    # ============= 4. evaluate_progress =============
    def evaluate_progress(self, goal_id: str) -> Dict[str, Any]:
        """
        목표 진행도 평가.

        progress = w_sa × SA 활성도 + w_em × EM 최근 빈도

        Returns: {'goal_id', 'category', 'progress', 'sa_part', 'em_part'}
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return {'goal_id': goal_id, 'progress': 0.0, 'error': 'not_found'}

        # SA 활성도 (현재)
        sa_act = self.sa.activations.get(goal.category, 0.0)
        sa_part = float(np.clip(sa_act, 0.0, 1.0))

        # EM 최근 빈도 (선택)
        em_part = 0.0
        em_available = False
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                episodes = list(self.em.episodes.values())
                if episodes:
                    em_available = True
                    recent = episodes[-30:]
                    count = sum(1 for ep in recent
                               if goal.category in getattr(ep, 'categories', set()))
                    em_part = min(1.0, count / 5.0)  # 5개면 1.0
            except Exception:
                em_part = 0.0
                em_available = False

        # EM 사용 가능하면 가중 합, 아니면 SA만
        # SA 우세 (현재 활성이 progress의 핵심 지표, EM은 빈도 보조)
        if em_available:
            W_SA = 0.8
            W_EM = 0.2
            progress = W_SA * sa_part + W_EM * em_part
        else:
            progress = sa_part
        progress = float(np.clip(progress, 0.0, 1.0))

        goal.progress = progress
        goal.last_evaluated = self.time

        return {
            'goal_id': goal_id,
            'category': goal.category,
            'progress': progress,
            'sa_part': sa_part,
            'em_part': em_part,
            'priority': goal.priority,
        }

    # ============= 5. suggest_action =============
    def suggest_action(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        목표 달성을 위한 행동 추천.

        알고리즘:
        - CG.parents(goal_cat): 이게 일어나면 goal_cat 발생 (학습된 cause)
        - 각 부모를 Counterfactual로 평가 (outcome 점수)
        - 가장 강한 + 좋은 outcome → 추천

        Returns:
            {'goal_id', 'recommended', 'weight', 'rationale'}
            또는 None (CG에 학습된 cause 없음)
        """
        self.suggested_count += 1

        goal = self.goals.get(goal_id)
        if not goal or goal.status != 'active':
            return None

        if self.cg is None:
            return None

        # CG에서 부모들
        parents = self.cg.parents(goal.category)
        if not parents:
            return None

        # 결정론적 정렬: weight ↓, 알파벳 ↑
        sorted_parents = sorted(parents.items(),
                               key=lambda kv: (-kv[1], kv[0]))
        best_cause, best_weight = sorted_parents[0]

        rationale = f"'{best_cause}' is strongest cause of '{goal.category}' (w={best_weight:.2f})"

        # Counterfactual 평가 옵션 (CF 있고 다른 부모 있을 때)
        cf_score = None
        if self.cf is not None and len(sorted_parents) > 1:
            try:
                second_cause = sorted_parents[1][0]
                cf_result = self.cf.counterfactual_action(best_cause, second_cause)
                cf_score = cf_result.get('outcome_score', 0.0)
                if cf_score < -0.2:
                    # second cause가 더 좋으면 그걸 추천
                    rationale = (f"'{second_cause}' preferred (outcome better than "
                                f"'{best_cause}' by {-cf_score:.2f})")
                    best_cause = second_cause
                    best_weight = sorted_parents[1][1]
            except Exception:
                pass

        return {
            'goal_id': goal_id,
            'goal_category': goal.category,
            'recommended': best_cause,
            'weight': best_weight,
            'rationale': rationale,
            'cf_score': cf_score,
            'time': self.time,
        }

    # ============= 6. complete_goal =============
    def complete_goal(self, goal_id: str,
                     auto: bool = False) -> Optional[Dict[str, Any]]:
        """
        목표 달성. dopamine + endorphin 자극.

        Args:
            auto: True면 자동 완료 (tick에서 호출), 통계용 구분

        Returns: {'goal_id', 'category', 'duration', 'auto'}
            또는 None (이미 active 아님)
        """
        goal = self.goals.get(goal_id)
        if not goal or goal.status != 'active':
            return None

        goal.status = 'completed'
        goal.completed_at = self.time
        duration = self.time - goal.created

        # 호르몬 보상
        self._stimulate_reward(goal.priority)

        # history로 이동
        self.history.append({
            'event': 'completed',
            'goal_id': goal_id,
            'category': goal.category,
            'priority': goal.priority,
            'duration': duration,
            'progress': goal.progress,
            'auto': auto,
            'time': self.time,
        })

        # active dict에서 제거 (history에 보존)
        del self.goals[goal_id]
        self.completed_count += 1

        return {
            'goal_id': goal_id,
            'category': goal.category,
            'duration': duration,
            'auto': auto,
        }

    def _stimulate_reward(self, priority: float):
        """달성 보상 — priority 비례"""
        if self.hs is None:
            return
        for h_name, base_amount in [('dopamine', self.COMPLETE_DOPAMINE),
                                    ('endorphin', self.COMPLETE_ENDORPHIN)]:
            h = self.hs.hormones.get(h_name)
            if h is None:
                continue
            amount = base_amount * (0.5 + 0.5 * priority)
            if hasattr(h, 'stimulate'):
                h.stimulate(amount)
            else:
                h.level = float(np.clip(h.level + amount, 0.0, 1.0))

    # ============= 7. abandon_goal =============
    def abandon_goal(self, goal_id: str,
                    reason: str = 'manual') -> Optional[Dict[str, Any]]:
        """포기. 약한 cortisol 자극 (실패 학습)"""
        goal = self.goals.get(goal_id)
        if not goal or goal.status != 'active':
            return None

        goal.status = 'abandoned'
        goal.abandon_reason = reason

        # 약한 페널티 (단, 'outranked' / 'expired'는 제외 — 자기 잘못 X)
        if reason not in ('outranked', 'expired') and self.hs is not None:
            cort = self.hs.hormones.get('cortisol')
            if cort is not None:
                amount = self.ABANDON_CORTISOL * goal.priority
                if hasattr(cort, 'stimulate'):
                    cort.stimulate(amount)
                else:
                    cort.level = float(np.clip(cort.level + amount, 0.0, 1.0))

        self.history.append({
            'event': 'abandoned',
            'goal_id': goal_id,
            'category': goal.category,
            'reason': reason,
            'priority': goal.priority,
            'progress': goal.progress,
            'time': self.time,
        })

        del self.goals[goal_id]
        self.abandoned_count += 1
        return {'goal_id': goal_id, 'reason': reason}

    # ============= 8. tick =============
    def tick(self, dt: float):
        """
        시간 누적 + 주기 평가 + 만료 체크 + 자동 propose.
        """
        if dt is None or dt <= 0:
            return
        self.time += float(dt)
        self.tick_count += 1

        # 만료 체크
        expired_ids = []
        for gid, g in list(self.goals.items()):
            if g.status != 'active':
                continue
            if g.deadline is not None and self.time > g.deadline:
                # 진행도 충분하면 complete, 아니면 expire
                if g.progress >= self.AUTO_COMPLETE_THRESHOLD:
                    self.complete_goal(gid, auto=True)
                else:
                    expired_ids.append(gid)

        for gid in expired_ids:
            g = self.goals.get(gid)
            if g:
                g.status = 'expired'
                self.history.append({
                    'event': 'expired',
                    'goal_id': gid,
                    'category': g.category,
                    'priority': g.priority,
                    'progress': g.progress,
                    'time': self.time,
                })
                del self.goals[gid]
                self.expired_count += 1

        # 주기 평가 + 자동 complete
        if self.tick_count % self.EVAL_INTERVAL_TICKS == 0:
            for gid in list(self.goals.keys()):
                ev = self.evaluate_progress(gid)
                if ev.get('progress', 0.0) >= self.AUTO_COMPLETE_THRESHOLD:
                    self.complete_goal(gid, auto=True)

        # 주기 propose
        if self.tick_count % self.PROPOSE_INTERVAL_TICKS == 0:
            self.propose_goal_from_need()

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        active = self.active_goals()
        return {
            'time': self.time,
            'active_count': len(active),
            'set_count': self.set_count,
            'completed_count': self.completed_count,
            'abandoned_count': self.abandoned_count,
            'expired_count': self.expired_count,
            'suggested_count': self.suggested_count,
            'proposed_count': self.proposed_count,
            'top_goal': (
                {'id': active[0].id, 'category': active[0].category,
                 'priority': active[0].priority, 'progress': active[0].progress}
                if active else None
            ),
            'history_size': len(self.history),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"GoalManagement(t={s['time']:.1f}s, "
                f"active={s['active_count']}, "
                f"completed={s['completed_count']}, "
                f"top={s['top_goal']['category'] if s['top_goal'] else None})")
