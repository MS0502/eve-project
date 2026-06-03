"""
EVE v37.13 — IntegratedSelf (분산 자아 통합 View)
==================================================

EVE의 자아는 *통제 타워가 아니라* 분산된 emergent 패턴.
- 호르몬 (체)
- SA 활성 (개념)
- WM focus (주의)
- 보호 카테고리 (정체성)
- 일주기 시간 (체내 시계)
- 환경 위치 (공간 자아)
- 사용자 관계 (사회 자아)
- 최근 thought stream (시간 자아)
- 활성 goal (의도 자아)

이 클래스는 read-only — 어느 모듈도 수정 X.
대시보드/응답 컨텍스트/메타 인지에 사용.

학계:
- Damasio 1999 — Protoself (Body + Memory + Emotion)
- Strawson 2000 — Self as 'narrative center'
- Friston 2010 — Self as model of own states
- Dennett 1991 — Self = Center of Narrative Gravity (분산 emergent)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SelfSnapshot:
    """한 시점의 EVE 자아 통합 view"""
    time: float = 0.0
    sim_hour: float = 0.0
    
    # 체 (Body)
    feeling: str = ""
    primary_hormone: Optional[Tuple[str, float]] = None
    mood: Dict[str, float] = field(default_factory=dict)  # valence/arousal/dominance
    
    # 정신 (Mind)
    focus: Optional[str] = None
    active_categories: List[Tuple[str, float]] = field(default_factory=list)
    inner_voice: Optional[str] = None
    dmn_mode: Optional[str] = None
    
    # 정체성 (Identity)
    protected_categories: Set[str] = field(default_factory=set)
    user_intimacy: float = 0.0
    user_visit_count: int = 0
    
    # 공간 (Space)
    location: Optional[str] = None
    is_in_simulation: bool = False
    
    # 시간 (Time)
    recent_thoughts: List[Dict[str, Any]] = field(default_factory=list)
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 의도 (Intention)
    current_goals: List[Dict[str, Any]] = field(default_factory=list)
    last_decision: Optional[Tuple[str, str]] = None
    
    # 메타
    tick_count: int = 0
    spontaneous_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'sim_hour': self.sim_hour,
            'feeling': self.feeling,
            'primary_hormone': self.primary_hormone,
            'mood': self.mood,
            'focus': self.focus,
            'active_categories': self.active_categories,
            'inner_voice': self.inner_voice,
            'dmn_mode': self.dmn_mode,
            'protected_categories': sorted(self.protected_categories),
            'user_intimacy': self.user_intimacy,
            'user_visit_count': self.user_visit_count,
            'location': self.location,
            'is_in_simulation': self.is_in_simulation,
            'recent_thoughts': self.recent_thoughts,
            'recent_actions': self.recent_actions,
            'current_goals': self.current_goals,
            'last_decision': self.last_decision,
            'tick_count': self.tick_count,
            'spontaneous_count': self.spontaneous_count,
        }

    def summary(self) -> str:
        """자연어 요약 — 한 줄 정도"""
        bits = []
        if self.location:
            sim_marker = '(sim)' if self.is_in_simulation else ''
            bits.append(f"{self.location}{sim_marker}")
        if self.focus:
            bits.append(f"focus={self.focus}")
        if self.feeling:
            bits.append(self.feeling)
        if self.current_goals:
            bits.append(f"goals={len(self.current_goals)}")
        return ' / '.join(bits)


class IntegratedSelf:
    """
    EVE의 분산 자아를 통합 view로 노출.
    
    원칙:
    - read-only (어떤 모듈도 수정 X)
    - lazy 평가 (snapshot 호출 시에만 수집)
    - 모듈 누락 안전 (없으면 default 값)
    """

    def __init__(self, eve):
        self.eve = eve
        self.history: List[SelfSnapshot] = []
        self.max_history = 100

    def snapshot(self) -> SelfSnapshot:
        """현재 EVE 상태 → 통합 snapshot"""
        eve = self.eve
        snap = SelfSnapshot()

        # 시간
        snap.time = getattr(eve, 'time', 0.0)
        snap.sim_hour = getattr(eve.hs, 'sim_hour', 0.0) if hasattr(eve, 'hs') else 0.0
        snap.tick_count = getattr(eve, 'tick_count', 0)

        # 체 — 호르몬 + feeling
        if hasattr(eve, 'nl'):
            try:
                snap.feeling = eve.nl.feeling_to_text()
            except Exception:
                pass
        if hasattr(eve, 'hs'):
            try:
                snap.primary_hormone = eve.hs.primary_hormone()
                snap.mood = eve.hs.compute_mood()
            except Exception:
                pass

        # 정신 — focus + 활성 + 속마음
        if hasattr(eve, 'wm'):
            snap.focus = eve.wm.get_focus()
        if hasattr(eve, 'sa'):
            items = list(eve.sa.activations.items())
            items.sort(key=lambda x: -x[1])
            snap.active_categories = [(c, float(v)) for c, v in items[:10]]
        if hasattr(eve, 'dmn'):
            snap.dmn_mode = getattr(eve.dmn, 'current_mode', None)
            snap.spontaneous_count = getattr(eve.dmn, 'spontaneous_count', 0)
            if hasattr(eve, 'nl'):
                try:
                    snap.inner_voice = eve.nl.inner_voice(eve.dmn)
                except Exception:
                    pass

        # 정체성
        snap.protected_categories = set(getattr(eve, 'protected_categories', set()))
        if hasattr(eve, 'user_presence'):
            snap.user_intimacy = eve.user_presence.get_intimacy()
            snap.user_visit_count = eve.user_presence.visit_count

        # 공간
        adapter = getattr(eve, '_env_adapter', None)
        if adapter is not None and adapter.current_env is not None:
            snap.location = adapter.current_env.name
            snap.is_in_simulation = getattr(adapter.current_env, 'is_simulation', False)

        # 시간 자아 (recent thoughts)
        snap.recent_thoughts = self._collect_thoughts()
        snap.recent_actions = self._collect_recent_actions()

        # 의도 (goals)
        snap.current_goals = self._collect_goals()

        # 마지막 결정
        if adapter and hasattr(adapter, '_decider') and adapter._decider:
            last = adapter._decider.last_decision
            if last:
                snap.last_decision = (last.action, last.target)

        # history 누적
        self.history.append(snap)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return snap

    def _collect_thoughts(self) -> List[Dict[str, Any]]:
        """다양한 모듈의 thought를 시간순으로 통합"""
        eve = self.eve
        thoughts = []

        # 1) DMN wandering — 자발 활성
        if hasattr(eve, 'dmn'):
            for time_, cat, mode in list(eve.dmn.wandering_history)[-8:]:
                thoughts.append({
                    'time': float(time_),
                    'source': 'dmn',
                    'mode': mode,
                    'content': cat,
                })

        # 2) MetaCognition recent_thoughts (있으면)
        if hasattr(eve, 'mc') and eve.mc is not None:
            recent_mc = getattr(eve.mc, 'recent_thoughts', None)
            if recent_mc:
                for t in list(recent_mc)[-5:]:
                    if isinstance(t, dict):
                        thoughts.append({
                            'time': t.get('time', 0.0),
                            'source': 'metacog',
                            'mode': t.get('cause', 'unknown'),
                            'content': t.get('content', str(t)),
                        })

        # 3) EM 최근 일화 (있으면)
        if hasattr(eve, 'em') and eve.em is not None:
            try:
                eps = eve.em.episodes
                if isinstance(eps, dict):
                    items = list(eps.values())
                else:
                    items = list(eps)
                items.sort(key=lambda e: getattr(e, 'time', 0), reverse=True)
                for ep in items[:3]:
                    cats = getattr(ep, 'categories', set())
                    if cats:
                        thoughts.append({
                            'time': float(getattr(ep, 'time', 0)),
                            'source': 'em',
                            'mode': 'recall',
                            'content': sorted(cats)[0],
                        })
            except Exception:
                pass

        # 시간순 정렬
        thoughts.sort(key=lambda t: -t['time'])
        return thoughts[:12]

    def _collect_recent_actions(self) -> List[Dict[str, Any]]:
        """최근 환경 행동"""
        eve = self.eve
        if (not hasattr(eve, '_env_adapter') or eve._env_adapter is None
                or eve._env_adapter.current_env is None):
            return []
        env = eve._env_adapter.current_env
        history = getattr(env, 'action_history', [])
        out = []
        for ac in list(history)[-5:]:
            if isinstance(ac, dict):
                out.append({
                    'action': ac.get('action'),
                    'target': ac.get('target'),
                    'success': ac.get('success'),
                })
        return out

    def _collect_goals(self) -> List[Dict[str, Any]]:
        """현재 active goals"""
        eve = self.eve
        if not hasattr(eve, 'gm') or eve.gm is None:
            return []
        try:
            active = eve.gm.active_goals(top_n=5)
        except Exception:
            return []
        out = []
        for g in active:
            out.append({
                'id': getattr(g, 'id', None),
                'category': getattr(g, 'category', None),
                'priority': float(getattr(g, 'priority', 0)),
                'progress': float(getattr(g, 'progress', 0)),
                'source': getattr(g, 'source', None),
            })
        return out

    # ============= 누적 view =============
    def thought_stream(self, n: int = 20) -> List[Dict[str, Any]]:
        """최근 N tick의 thought 시간순 흐름"""
        return [thought for snap in self.history[-n:]
                for thought in snap.recent_thoughts]

    def goal_alignment_score(self, action: str, target: Optional[str] = None) -> float:
        """
        후보 행동이 현재 active goals와 얼마나 정렬되는지 점수.
        decide_action에서 활용 가능.
        
        Returns:
            0.0~1.0 — 1.0이면 강한 goal과 정렬
        """
        snap = self.history[-1] if self.history else self.snapshot()
        goals = snap.current_goals
        if not goals:
            return 0.0

        score = 0.0
        for g in goals:
            cat = g.get('category', '')
            priority = g.get('priority', 0)
            # action이 goal 카테고리와 매칭되거나
            # target이 goal 카테고리와 매칭되면 정렬
            if cat and (cat in action or (target and cat in target) or action in cat):
                score += priority * 0.5
            # SA 그래프에서 action ↔ goal_category 가중치 확인
            if hasattr(self.eve, 'sa') and cat:
                w = self.eve.sa.get_weight(action, cat)
                if w > 0.2:
                    score += priority * w * 0.3
        return min(1.0, score)
