"""
EVE v32 - D5 ToolUse (메타 도구 사용)
======================================

EVE가 자기 23 모듈을 의식적으로 도구로 선택/호출.

학계:
- Vygotsky 1978 — Psychological Tools (도구 = 정신 확장)
- Kahneman 2011 — System 2 (의식적 통제, 노력 사용)
- Bratman 1987 — Intention as Plan
- Clark 2008 — Extended Mind Hypothesis
- Norman 1991 — Cognitive Artifacts
- Hutchins 1995 — Distributed Cognition

EVE 원칙 부합:
- 확률 X (matching score, 결정론 정렬)
- 카테고리 = 의미 단위 (도구도 카테고리로 표현)
- 호르몬 = context signature (어떤 도구가 어떤 호르몬 상태에 유리)
- 다른 모듈 호출만 (read 또는 외부 인터페이스), 직접 변경 X

자동 (지금까지) vs 의도적 (D5):
- 자동: 호르몬↑ → 자동 reappraise, DMN → 자동 wandering
- 의도적: "이 상황 어렵다" → 명시적 도구 선택 → 호출 → 반영

핵심 함수 (7개):
1. register_tool(name, fn, signature, description)
2. select_tool(situation, top_n)         — 상황 → 도구 선택 (결정론)
3. invoke_tool(name, *args, **kw)        — 호출 + 기록
4. plan_and_act(situation, args)         — 통합: select → invoke
5. reflect_on_use(tool_name)             — 도구 사용 결과 메타 평가
6. update_effectiveness(name, outcome)   — 효과 학습 (호르몬 신호로)
7. tool_history(last_n)                  — 이력 조회
+ tick(dt)

Author: 김민석 + Claude v32 (D5)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import deque, defaultdict


class ToolUse:
    """
    EVE v32 D5 도구 사용 시스템.

    도구 = (callable, context_signature, description)
    context_signature: 어떤 상황 (호르몬/카테고리) 에 유리한지
        예: {'cortisol_high': 1.0, 'regret_recent': 0.8} → ER.reappraise
    """

    def __init__(self,
                 hormone_system,
                 spreading_activation,
                 working_memory=None,
                 self_doubt=None,
                 metacognition=None,
                 agency=None,
                 history_size: int = 100):
        self.hs = hormone_system
        self.sa = spreading_activation
        self.wm = working_memory
        self.sd = self_doubt
        self.mc = metacognition
        self.ag = agency

        # Tool registry: name → {fn, signature, description, effectiveness}
        self.tools: Dict[str, Dict[str, Any]] = {}

        # 사용 이력: deque of {time, name, args_summary, result_summary,
        #                     pre_horm, post_horm, outcome}
        self.history: deque = deque(maxlen=history_size)

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.invoke_count = 0
        self.select_count = 0
        self.reflect_count = 0
        self.failed_count = 0

    # ============= 1. 도구 등록 =============
    def register_tool(self,
                     name: str,
                     fn: Callable,
                     signature: Dict[str, float],
                     description: str = "",
                     category: str = "general") -> bool:
        """
        도구 등록.

        Args:
            name: 도구 이름 (예: 'reappraise', 'why_caused')
            fn: 실제 호출할 callable
            signature: {context_key: weight} — 매칭 점수 계산용
                예: {'cortisol_high': 1.0, 'doubt_high': 0.5}
            description: 사용 설명
            category: 'reasoning'/'recall'/'simulation'/'regulation'/
                      'creative'/'social'/'analytic'/'meta'

        Returns: 새로 등록 True, 이미 있으면 False
        """
        if name in self.tools:
            return False
        self.tools[name] = {
            'fn': fn,
            'signature': dict(signature),
            'description': description,
            'category': category,
            'effectiveness': 0.5,  # baseline 중립
            'use_count': 0,
            'success_count': 0,
        }
        return True

    # ============= 2. 도구 선택 (의식적 매칭) =============
    def select_tool(self,
                   situation: Optional[Dict[str, float]] = None,
                   category_filter: Optional[str] = None,
                   top_n: int = 3) -> List[Tuple[str, float]]:
        """
        현재 상황 → 도구 매칭 점수.

        situation = None이면 자동 추출 (호르몬 + 카테고리).

        매칭:
            score = sum(situation[k] * signature[k]) for k in shared
                  + effectiveness × 0.3 (학습된 효과)

        결정론 정렬 (점수 ↓, 이름 ↑).

        Returns: [(tool_name, score), ...] top_n
        """
        if situation is None:
            situation = self._auto_situation()

        scored: List[Tuple[str, float]] = []
        for name, info in self.tools.items():
            if category_filter and info['category'] != category_filter:
                continue
            sig = info['signature']
            # 매칭 점수
            match_score = 0.0
            for k, sit_v in situation.items():
                if k in sig:
                    match_score += sit_v * sig[k]

            # 효과 학습 보너스
            effect = info.get('effectiveness', 0.5)
            score = match_score + (effect - 0.5) * 0.3

            if score > 0:
                scored.append((name, float(score)))

        # 정렬
        scored.sort(key=lambda x: (-x[1], x[0]))

        self.select_count += 1
        return scored[:top_n]

    def _auto_situation(self) -> Dict[str, float]:
        """
        현재 EVE 상태 → context signature.
        호르몬 + 활성 카테고리 + SD/MC 시그널.
        Smooth gradient — 작은 변화에도 신호 (모든 상황에서 도구 추천 가능).
        """
        sit: Dict[str, float] = {}

        # 호르몬 → context keys
        cort = self.hs.hormones['cortisol'].level
        da = self.hs.hormones['dopamine'].level
        ot = self.hs.hormones.get(
            'oxytocin', type('h', (), {'level': 0.3})()).level
        ne = self.hs.hormones.get(
            'norepinephrine', type('h', (), {'level': 0.3})()).level
        mel = self.hs.hormones.get(
            'melatonin', type('h', (), {'level': 0.1})()).level

        # Smooth ramp — 0.3 이상부터 작은 신호, 1.0에서 max
        if cort > 0.3:
            sit['cortisol_high'] = (cort - 0.3) / 0.7
        if cort < 0.4:
            sit['cortisol_low'] = (0.4 - cort) / 0.4
        if da > 0.3:
            sit['dopamine_high'] = (da - 0.3) / 0.7
        if da < 0.4:
            sit['dopamine_low'] = (0.4 - da) / 0.4
        if ot > 0.3:
            sit['oxytocin_high'] = (ot - 0.3) / 0.7
        if ne > 0.3:
            sit['ne_high'] = (ne - 0.3) / 0.7
        if mel > 0.3:
            sit['melatonin_high'] = (mel - 0.3) / 0.7

        # SD signal
        if self.sd is not None:
            try:
                doubt_level = getattr(self.sd, 'baseline', 0.2)
                if doubt_level > 0.3:
                    sit['doubt_high'] = min(1.0, doubt_level / 0.5)
            except Exception:
                pass

        # 활성 카테고리 (some signals)
        try:
            active = self.sa.get_active()
            # 슬픔/위협 카테고리 active?
            negative_cats = {'슬픔', '걱정', '불안', '아프다',
                            '두려움', '고통', '외로움'}
            # marker (D2 Suffering '슬픔_타자_X' 등)도 포함
            neg_score = 0.0
            for c in active:
                if c in negative_cats:
                    neg_score += 0.7
                elif any(nc in c for nc in negative_cats):
                    neg_score += 0.5  # marker 카테고리
            if neg_score > 0:
                sit['negative_active'] = min(1.0, neg_score)
            # 의문/궁금
            question_cats = {'왜', '어떻게', '궁금', '모르는'}
            if active & question_cats:
                sit['question_active'] = 0.7
            # 행동 결정
            action_cats = {'할까', '결정', '선택'}
            if active & action_cats:
                sit['decision_pending'] = 0.6
        except Exception:
            pass

        # WM focus
        if self.wm is not None:
            try:
                focus = self.wm.get_focus()
                if focus:
                    sit['has_focus'] = 1.0
            except Exception:
                pass

        return sit

    # ============= 3. 도구 호출 =============
    def invoke_tool(self,
                   name: str,
                   *args,
                   **kwargs) -> Dict[str, Any]:
        """
        도구 호출 + 기록.

        Returns:
            {tool, success, result, error, pre_state, post_state}
        """
        if name not in self.tools:
            self.failed_count += 1
            return {
                'tool': name,
                'success': False,
                'error': f"unknown tool: {name}",
                'result': None,
            }

        info = self.tools[name]

        # pre-state (호르몬)
        pre_horm = {
            'cortisol': self.hs.hormones['cortisol'].level,
            'dopamine': self.hs.hormones['dopamine'].level,
        }
        if 'serotonin' in self.hs.hormones:
            pre_horm['serotonin'] = self.hs.hormones['serotonin'].level

        # 호출
        try:
            result = info['fn'](*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            self.failed_count += 1

        # post-state
        post_horm = {
            'cortisol': self.hs.hormones['cortisol'].level,
            'dopamine': self.hs.hormones['dopamine'].level,
        }
        if 'serotonin' in self.hs.hormones:
            post_horm['serotonin'] = self.hs.hormones['serotonin'].level

        # 기록
        record = {
            'time': self.time,
            'tool': name,
            'category': info['category'],
            'args_summary': str(args)[:80] + ('...' if len(str(args)) > 80 else ''),
            'success': success,
            'error': error,
            'pre_horm': pre_horm,
            'post_horm': post_horm,
            'result_type': type(result).__name__ if result is not None else 'None',
        }
        self.history.append(record)

        info['use_count'] += 1
        if success:
            info['success_count'] += 1
            self.invoke_count += 1

        return {
            'tool': name,
            'success': success,
            'result': result,
            'error': error,
            'pre_state': pre_horm,
            'post_state': post_horm,
        }

    # ============= 4. plan_and_act (통합) =============
    def plan_and_act(self,
                    situation: Optional[Dict[str, float]] = None,
                    args: Optional[tuple] = None,
                    kwargs: Optional[Dict] = None,
                    category_filter: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
        """
        select_tool 1위 자동 호출.

        Bratman intention: 상황 인식 → 계획 → 실행.

        Args:
            situation: None이면 자동
            args/kwargs: 도구 호출 인자
            category_filter: 특정 카테고리 도구만

        Returns: invoke_tool 결과 또는 None (적합 도구 X)
        """
        candidates = self.select_tool(situation,
                                      category_filter=category_filter,
                                      top_n=1)
        if not candidates:
            return None

        tool_name, score = candidates[0]
        args = args or ()
        kwargs = kwargs or {}

        result = self.invoke_tool(tool_name, *args, **kwargs)
        result['planned_score'] = float(score)
        return result

    # ============= 5. reflect_on_use =============
    def reflect_on_use(self,
                      tool_name: Optional[str] = None
                      ) -> Optional[Dict[str, Any]]:
        """
        도구 사용 결과 메타 평가.
        - tool_name=None이면 가장 최근 호출
        - 호르몬 변화 → 효과 추정

        Returns:
            {tool, helpful, cort_change, da_change, mood_change}
        """
        if not self.history:
            return None

        if tool_name is None:
            record = self.history[-1]
        else:
            # 가장 최근 같은 이름
            record = None
            for r in reversed(self.history):
                if r['tool'] == tool_name:
                    record = r
                    break
            if record is None:
                return None

        cort_change = (record['post_horm']['cortisol']
                      - record['pre_horm']['cortisol'])
        da_change = (record['post_horm']['dopamine']
                    - record['pre_horm']['dopamine'])

        # helpful: cort↓ + da↑ → helpful, 반대 → not
        score = -cort_change * 1.5 + da_change * 1.2
        helpful = score > 0.05

        result = {
            'tool': record['tool'],
            'time': record['time'],
            'helpful': helpful,
            'helpful_score': float(score),
            'cort_change': float(cort_change),
            'da_change': float(da_change),
            'mood_better': cort_change < 0 and da_change > 0,
        }

        self.reflect_count += 1
        return result

    # ============= 6. effectiveness 학습 =============
    def update_effectiveness(self,
                            tool_name: str,
                            outcome: str = 'helpful') -> Dict[str, Any]:
        """
        도구 효과 학습.

        Args:
            tool_name: 도구 이름
            outcome: 'helpful' / 'neutral' / 'unhelpful'

        Hebbian-like 업데이트: 효과 baseline 0.5 기준 ±0.05
        Range [0.05, 0.95] (lock 방지)

        Returns: {tool, old, new, outcome}
        """
        if tool_name not in self.tools:
            return {'error': 'unknown tool'}

        old_eff = self.tools[tool_name]['effectiveness']

        delta_map = {
            'helpful': 0.05,
            'neutral': 0.0,
            'unhelpful': -0.05,
        }
        delta = delta_map.get(outcome, 0.0)

        new_eff = float(np.clip(old_eff + delta, 0.05, 0.95))
        self.tools[tool_name]['effectiveness'] = new_eff

        return {
            'tool': tool_name,
            'old_effectiveness': float(old_eff),
            'new_effectiveness': new_eff,
            'outcome': outcome,
        }

    # ============= 7. 이력 조회 =============
    def tool_history(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """최근 도구 사용 이력"""
        return list(self.history)[-last_n:]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """특정 도구 정보 조회"""
        info = self.tools.get(tool_name)
        if info is None:
            return None
        return {
            'name': tool_name,
            'category': info['category'],
            'description': info['description'],
            'effectiveness': info['effectiveness'],
            'use_count': info['use_count'],
            'success_count': info['success_count'],
            'success_rate': (info['success_count'] / info['use_count']
                            if info['use_count'] > 0 else 0.0),
        }

    def list_tools(self,
                  category: Optional[str] = None
                  ) -> List[Dict[str, Any]]:
        """등록된 도구 list"""
        tools = []
        for name in sorted(self.tools.keys()):
            info = self.tools[name]
            if category and info['category'] != category:
                continue
            tools.append({
                'name': name,
                'category': info['category'],
                'description': info['description'],
                'effectiveness': info['effectiveness'],
                'use_count': info['use_count'],
            })
        return tools

    # ============= tick =============
    def tick(self, dt: float):
        self.time += dt
        self.tick_count += 1

    # ============= 상태 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'tools_registered': len(self.tools),
            'invoke_count': self.invoke_count,
            'select_count': self.select_count,
            'reflect_count': self.reflect_count,
            'failed_count': self.failed_count,
            'history_size': len(self.history),
            'recent_tools': [r['tool'] for r in
                            list(self.history)[-3:]],
        }

    def __repr__(self):
        s = self.get_state()
        return (f"ToolUse(t={s['time']:.1f}, "
                f"tools={s['tools_registered']}, "
                f"invoked={s['invoke_count']}, "
                f"failed={s['failed_count']})")
