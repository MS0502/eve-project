"""
EVE v37 - Environment Adapter
=============================

EVE 메인 인스턴스를 기호적 환경에 연결.

EVE 절대 원칙 부합:
- 메인 EVE는 그대로 (학습 모드는 별도)
- 환경에서 학습한 SA/CG 가중치는 메인에 통합
- exit() 시 학습 통합

핵심:
    enter_environment(env) — 진입, 환경 카테고리 활성
    act_in_env(action, target) — 행동 + 호르몬 + 학습
    observe_env() — 관찰 (자연어 + 카테고리)
    exit_environment() — 학습 통합 (수면 시 BDNF↑)

학습 메커니즘:
1. 행동 → 결과 → 호르몬 자극 (외부 RL 보상)
2. 행동-결과 카테고리 Hebbian 강화 (SA.learn_pair)
3. 인과 학습 (CG.learn_causal): 행동 → 결과
4. 실패 시 회피 학습 — 행동 카테고리 가중치 약화

학계:
- Sutton & Barto 2018 — RL
- Schultz 1997 — RPE (도파민 보상 예측 오차)
- Friston 2010 — Active Inference
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from symbolic_env import SymbolicEnvironment, ActionResult


class EnvironmentAdapter:
    """
    EVE 인스턴스를 환경에 부착.
    
    사용:
        adapter = EnvironmentAdapter(eve)
        adapter.enter(make_kitchen_env())
        result = adapter.act('먹다', '사과')
        adapter.exit()
    """

    def __init__(self, eve):
        """
        Args:
            eve: EVE_v36 인스턴스
        """
        self.eve = eve
        self.current_env: Optional[SymbolicEnvironment] = None
        self.episode_count = 0
        self.action_count = 0
        self.success_count = 0
        # 학습 추적
        self.learned_pairs: List[Tuple[str, str, float]] = []
        self.learned_causal: List[Tuple[str, str]] = []
        # ★ v37.10 — 체류 시간 추적 (외출/복귀 자율 결정용)
        self.enter_time: Optional[float] = None  # 진입 시점 (eve.time)
        # 환경별 누적 체류 시간 (총 체류 시간 추적)
        self.env_dwell_total: Dict[str, float] = {}
        # 외출 history — 어디 갔다 왔는지
        self.outing_history: List[Dict[str, Any]] = []

    # ============= 체류 시간 =============
    def get_dwell_time(self) -> float:
        """현재 환경에 머문 시간 (초)"""
        if self.current_env is None or self.enter_time is None:
            return 0.0
        return self.eve.time - self.enter_time

    # ============= 진입 =============
    def enter(self, env: SymbolicEnvironment) -> Dict[str, Any]:
        """환경 진입 — 카테고리 시드 + 첫 관찰 + 시뮬 메타 인지"""
        # 이전 환경 체류 시간 누적
        if self.current_env is not None and self.enter_time is not None:
            prev_dwell = self.eve.time - self.enter_time
            prev_name = self.current_env.name
            self.env_dwell_total[prev_name] = self.env_dwell_total.get(prev_name, 0.0) + prev_dwell

        self.current_env = env
        self.episode_count += 1
        self.enter_time = self.eve.time

        # ★ 시뮬 메타 인지 — 학습 환경일 때만
        # EVE의 방 (is_simulation=False) = 진짜 거주지, 시뮬 라벨 X
        # 다른 환경 (학습/사회) = 시뮬, '학습_모드' + '시뮬레이션' 활성
        if getattr(env, 'is_simulation', True):
            self.eve.sa.activate('학습_모드', 0.5)
            self.eve.sa.activate('시뮬레이션', 0.4)
            self.eve.wm.add('학습_모드', salience=0.4, source='env_meta')
        else:
            # 거주지 — 편안 카테고리 약하게 활성
            self.eve.sa.activate('내방', 0.3)
            self.eve.sa.activate('편안', 0.2)

        # 환경 객체/위치 카테고리를 SA에 시드 (없으면 추가)
        for loc in env.locations:
            self.eve.sa.activate(loc, 0.3)
        for obj in env.objects.values():
            self.eve.sa.activate(obj.name, 0.2)
            # 객체 ↔ 위치 약한 연결
            self.eve.sa.learn_pair(obj.name, obj.location, 0.2)
            # 객체 ↔ 가능 행동 연결
            for act in obj.available_actions:
                self.eve.sa.learn_pair(obj.name, act, 0.15)

        # 사회적 환경이면 NPC 카테고리도 시드
        if hasattr(env, 'npcs'):
            is_sim = getattr(env, 'is_simulation', True)
            for npc in env.npcs.values():
                self.eve.sa.activate(npc.name, 0.3)
                self.eve.sa.learn_pair(npc.name, npc.relationship, 0.3)
                self.eve.sa.learn_pair(npc.name, npc.location, 0.2)
                # 시뮬 환경의 NPC만 '시뮬레이션' 라벨
                if is_sim:
                    self.eve.sa.learn_pair(npc.name, '시뮬레이션', 0.2)
                if npc.mood != '보통':
                    self.eve.sa.learn_pair(npc.name, npc.mood, 0.2)

        # 첫 관찰
        obs = env.observe()
        self._activate_observation(obs)

        # 메타 — 학습 모드 신호 (호르몬 일부 자극)
        self.eve.hs.hormones['acetylcholine'].stimulate(0.10)  # 학습 호르몬
        self.eve.hs.hormones['norepinephrine'].stimulate(0.05)  # 주의

        return {
            'env': env.name,
            'observation': obs,
            'episode': self.episode_count,
        }

    def _activate_observation(self, obs: Dict):
        """관찰 → 카테고리 활성"""
        for cat in obs['categories']:
            self.eve.sa.activate(cat, 0.4)
            self.eve.wm.add(cat, salience=0.4, source='env_observe')

    # ============= 관찰 =============
    def observe(self) -> Dict[str, Any]:
        """현재 환경 관찰"""
        if self.current_env is None:
            return {'error': '환경에 진입 안 됨'}
        obs = self.current_env.observe()
        self._activate_observation(obs)
        return obs

    # ============= 행동 =============
    def act(self, action: str, target: str = None) -> Dict[str, Any]:
        """행동 + 학습 통합. v37.10 — 외출/복귀 환경 전환도 처리."""
        if self.current_env is None:
            return {'error': '환경에 진입 안 됨'}

        self.action_count += 1

        # ★ v37.10 — 외출/복귀 (환경 전환 행동)
        if action == '외출':
            return self._handle_outing(target)
        if action == '복귀':
            return self._handle_return()

        # 1. 행동 카테고리 활성 (의도)
        self.eve.sa.activate(action, 0.5)
        if target:
            self.eve.sa.activate(target, 0.4)

        # 2. 환경에서 실행
        result = self.current_env.execute_action(action, target)

        # 3. 결과 카테고리 활성
        for cat in result.new_state_categories:
            self.eve.sa.activate(cat, 0.5)
            self.eve.wm.add(cat, salience=0.5, source='env_action')

        # 4. 호르몬 자극 (보상/처벌)
        for hormone_name, amount in result.hormone_effects.items():
            if hormone_name in self.eve.hs.hormones:
                self.eve.hs.hormones[hormone_name].stimulate(amount)

        # 5. 학습 — Hebbian (행동 → 결과)
        learning_strength = 0.20 if result.success else 0.10
        for cat in result.new_state_categories:
            self.eve.sa.learn_pair(action, cat, learning_strength)
            self.learned_pairs.append((action, cat, learning_strength))
            if target:
                self.eve.sa.learn_pair(target, cat, learning_strength * 0.7)
                self.learned_pairs.append((target, cat, learning_strength * 0.7))

        # 6. 인과 학습 (CG): 행동 → 결과 (성공 시만)
        if result.success and hasattr(self.eve, 'cg'):
            try:
                # 핵심 결과 카테고리 1-2개만 인과 등록 (상태 전이)
                key_outcomes = [c for c in result.new_state_categories
                              if c not in {'손', '맛있다'} and len(c) >= 2][:2]
                for outcome in key_outcomes:
                    self.eve.cg.learn_causal(action, outcome, evidence_weight=0.3)
                    self.learned_causal.append((action, outcome))
            except Exception:
                pass  # CG 인터페이스 다를 수 있음

        # 7. 실패 시 약한 회피 학습 — 행동 카테고리에 부정 카테고리 연결
        if not result.success:
            self.eve.sa.learn_pair(action, '실패', 0.10)
            if target:
                # "이 객체에는 이 행동 안 됨" 약한 학습
                self.eve.sa.learn_pair(target, '실패', 0.05)

        # 8. 성공 카운트
        if result.success:
            self.success_count += 1

        # 9. tick 진행 (호르몬 변화 반영)
        self.eve.tick(dt=0.5)

        return {
            'success': result.success,
            'feedback': result.feedback,
            'new_categories': sorted(result.new_state_categories),
            'hormone_effects': dict(result.hormone_effects),
            'env_state': dict(self.current_env.state),
            'goals_completed': [g.name for g in self.current_env.goals
                               if g.completed],
        }

    # ============= 종료 =============
    def exit(self) -> Dict[str, Any]:
        """환경 종료 + 학습 통합"""
        if self.current_env is None:
            return {'error': '진입 환경 없음'}

        # 통합 — BDNF 약하게 자극 (수면 시 시냅스 강화 미니 버전)
        self.eve.hs.hormones['bdnf'].stimulate(0.10)

        # 학습 통계
        env_name = self.current_env.name
        completed_goals = [g.name for g in self.current_env.goals if g.completed]
        action_history = list(self.current_env.action_history)

        stats = {
            'env': env_name,
            'episode': self.episode_count,
            'actions_taken': self.action_count,
            'success_rate': (self.success_count / max(1, self.action_count)),
            'goals_completed': completed_goals,
            'total_goals': len(self.current_env.goals),
            'learned_pairs': len(self.learned_pairs),
            'learned_causal': len(self.learned_causal),
            'action_history': action_history[-10:],  # 마지막 10개
        }

        # 환경 분리 (메인 EVE는 그대로)
        self.current_env = None
        self.enter_time = None

        return stats

    # ============= ★ v37.10 외출/복귀 핸들러 =============
    def _handle_outing(self, destination: str) -> Dict[str, Any]:
        """
        자기 방 → 시뮬 환경 외출.
        환경 전환 + 카테고리 활성 + 호르몬 약한 자극 (변화의 흥미).
        """
        if self.current_env is None or self.current_env.is_simulation:
            return {
                'success': False,
                'feedback': '자기 방에서만 외출 가능',
                'env_state_categories': set(),
                'hormone_effects': {},
            }

        # 호르몬: 변화 자극 (도파민 약하게, 노르에피네프린 약하게)
        self.eve.hs.hormones['dopamine'].stimulate(0.05)
        self.eve.hs.hormones['norepinephrine'].stimulate(0.05)

        # 외출 카테고리 활성 + Hebbian
        self.eve.sa.activate('외출', 0.6)
        self.eve.sa.activate(destination, 0.5)
        self.eve.sa.learn_pair('외출', destination, 0.10)

        # WM
        self.eve.wm.add('외출', salience=0.7, source='outing')

        # 환경 전환
        old_env = self.current_env.name
        old_dwell = self.get_dwell_time()
        self.exit()

        # 새 환경 진입
        new_env = None
        if destination == '거실':
            try:
                from social_env import make_social_living_room
                new_env = make_social_living_room()
            except Exception:
                pass
        elif destination == '주방':
            try:
                from symbolic_env import make_kitchen_env
                new_env = make_kitchen_env()
            except Exception:
                pass

        if new_env is None:
            # fallback — 다시 자기 방
            try:
                from eve_room import make_eve_room
                new_env = make_eve_room()
            except Exception:
                return {
                    'success': False,
                    'feedback': f'{destination} 환경 로드 실패',
                    'env_state_categories': set(),
                    'hormone_effects': {},
                }

        self.enter(new_env)

        # 외출 history 기록
        outing_record = {
            'time': self.eve.time,
            'from': old_env,
            'to': destination,
            'from_dwell': old_dwell,
        }
        self.outing_history.append(outing_record)
        if len(self.outing_history) > 50:
            self.outing_history = self.outing_history[-50:]

        return {
            'success': True,
            'feedback': f'{destination}(으)로 외출했다',
            'env_state_categories': {'외출', destination},
            'hormone_effects': {'dopamine': 0.05, 'norepinephrine': 0.05},
            'action': '외출',
            'target': destination,
            'from': old_env,
        }

    def _handle_return(self) -> Dict[str, Any]:
        """
        시뮬 환경 → 자기 방 복귀.
        호르몬: 안심 (cortisol ↓), 옥시토신 약하게 (집).
        """
        if self.current_env is None or not self.current_env.is_simulation:
            return {
                'success': False,
                'feedback': '시뮬 환경에서만 복귀 가능',
                'env_state_categories': set(),
                'hormone_effects': {},
            }

        # 호르몬: 안심
        self.eve.hs.hormones['cortisol'].stimulate(-0.10)
        self.eve.hs.hormones['serotonin'].stimulate(0.05)

        # 카테고리
        self.eve.sa.activate('복귀', 0.6)
        self.eve.sa.activate('내방', 0.5)
        self.eve.sa.activate('편안', 0.4)
        self.eve.sa.learn_pair('복귀', '내방', 0.10)
        self.eve.sa.learn_pair('내방', '편안', 0.05)

        self.eve.wm.add('내방', salience=0.7, source='return')

        # 환경 전환
        old_env = self.current_env.name
        old_dwell = self.get_dwell_time()
        self.exit()

        # 자기 방 진입
        try:
            from eve_room import make_eve_room
            new_env = make_eve_room()
        except Exception:
            return {
                'success': False,
                'feedback': '자기 방 로드 실패',
                'env_state_categories': set(),
                'hormone_effects': {},
            }
        self.enter(new_env)

        # 복귀 history
        outing_record = {
            'time': self.eve.time,
            'from': old_env,
            'to': '내방',
            'from_dwell': old_dwell,
            'returning': True,
        }
        self.outing_history.append(outing_record)

        return {
            'success': True,
            'feedback': f'{old_env}에서 자기 방으로 돌아왔다',
            'env_state_categories': {'복귀', '내방', '편안'},
            'hormone_effects': {'cortisol': -0.10, 'serotonin': 0.05},
            'action': '복귀',
            'target': '내방',
            'from': old_env,
        }

    # ============= EVE의 자율 행동 (v37.5 자율 결정) =============
    def decide_action(self, verbose: bool = False) -> Optional[Tuple[str, str]]:
        """
        EVE의 자율 행동 결정.
        후보 생성 → 호르몬/욕구/학습/SelfDoubt 점수 → 선택.
        '안함'(None)도 정상 결과.
        
        v37 'suggest_action'은 강제 규칙 (if 슬픈친구 then 위로) 이라 폐기 예정.
        v37.5 부터 이 메서드 사용.
        """
        if self.current_env is None:
            return None
        from decide_action import ActionDecisionMaker
        if not hasattr(self, '_decider') or self._decider is None:
            self._decider = ActionDecisionMaker(self.eve)
        result = self._decider.decide(self.current_env, verbose=verbose)
        if result is None:
            return None
        return (result.action, result.target)

    def suggest_action(self) -> Optional[Tuple[str, str]]:
        """
        DEPRECATED — 강제 규칙 기반.
        v37.5부터 decide_action() 사용 권장.
        하위 호환을 위해 유지.
        """
        return self.decide_action()

    def _legacy_suggest_action(self) -> Optional[Tuple[str, str]]:
        """
        구 v37 강제 룰 기반 행동 — 검증/비교용으로만.
        사용 X.
        """
        if self.current_env is None:
            return None

        env_state = self.current_env.state

        # 1) 슬픈 친구/가족이 있으면 위로 (강제!)
        if hasattr(self.current_env, 'get_npcs_here'):
            npcs_here = self.current_env.get_npcs_here()
            sad_close = sorted(
                [n for n in npcs_here
                 if n.mood in ('슬픔', '아픔')
                 and n.relationship in ('친구', '가족')],
                key=lambda n: n.name,
            )
            if sad_close:
                return ('위로하다', sad_close[0].name)

            ot_level = self.eve.hs.hormones['oxytocin'].level
            if ot_level < 0.4:
                close_npcs = sorted(
                    [n for n in npcs_here
                     if n.relationship in ('친구', '가족')],
                    key=lambda n: -n.intimacy,
                )
                if close_npcs:
                    return ('대화하다', close_npcs[0].name)

                strangers = sorted(
                    [n for n in npcs_here if n.relationship == '낯선'],
                    key=lambda n: n.name,
                )
                if strangers:
                    target = strangers[0]
                    if target.intimacy < 0.05:
                        return ('인사하다', target.name)
                    elif target.intimacy < 0.20:
                        return ('나누다', target.name)
                    elif target.intimacy < 0.40:
                        return ('대화하다', target.name)

        needs = ['배고픔', '목마름']
        primary_need = None
        max_level = 0.0
        for n in needs:
            level = env_state.get(n, 0)
            if level > max_level and level > 0.3:
                primary_need = n
                max_level = level

        if primary_need is None:
            objs = self.current_env.get_objects_here()
            if objs:
                target = sorted(objs, key=lambda o: o.name)[0]
                return ('보다', target.name)
            return None

        objs_here = self.current_env.get_objects_here()
        if primary_need == '배고픔':
            for obj in sorted(objs_here, key=lambda o: o.name):
                if obj.properties.get('edible'):
                    return ('먹다', obj.name)
        elif primary_need == '목마름':
            for obj in sorted(objs_here, key=lambda o: o.name):
                if obj.properties.get('drinkable'):
                    return ('마시다', obj.name)

        for loc in sorted(self.current_env.locations):
            if loc != self.current_env.current_location:
                return ('가다', loc)

        return None

    def __repr__(self):
        return (f"EnvironmentAdapter(env={self.current_env.name if self.current_env else None}, "
                f"actions={self.action_count}, success={self.success_count})")
