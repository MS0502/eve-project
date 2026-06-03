"""
EVE v36 통합 모듈
================

v35 EVE_TierAB에 4개 보강 기능 부착:
1. Persistence — save/load (체크포인트)
2. CommonSense seed — 상식 그래프 350+ 트리플
3. ResponseEnhancer — NL 응답 결정론적 강화
4. Proactive — 능동 인터랙션 (먼저 말 검)

기존 회귀 안전:
- v35 모듈은 그대로
- 부착 방식 (monkey-patch + 새 메서드)
- 옵션: 기존 say()는 그대로, say_v3()로 강화 응답
- 또는 enhance=True 모드로 say()도 강화
"""

import sys
import os

# v35 baseline 경로
V35_PATH = '/home/claude/eve_v36/eve_v35_release'
V36_PATH = '/home/claude/eve_v36/v36_modules'
sys.path.insert(0, V35_PATH)
sys.path.insert(0, V36_PATH)

from eve_main_abc import EVE_TierAB as _EVE_v35
from persistence import attach_persistence, Persistence
from commonsense_seed import seed_commonsense, get_categories as cs_get_categories
from response_enhancer import ResponseEnhancer
from proactive import Proactive


class EVE_v36(_EVE_v35):
    """
    EVE v35 + v36 강화.
    
    추가 인스턴스:
        self.enhancer: ResponseEnhancer
        self.proactive: Proactive
    
    추가 메서드:
        save / load / from_checkpoint
        say_v3(text)  - 강화된 응답
        proactive_message()  - 먼저 말 검
        seed_common_sense()  - 상식 시드
    """

    def __init__(self, *args, seed_common_sense_on_init: bool = True,
                 enhance_say: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        # FrameSemantics가 v35에 통합됐는지 확인
        # 안 됐으면 별도 인스턴스화
        if not hasattr(self, 'fs') or self.fs is None:
            try:
                from frame_semantics import FrameSemantics
                self.fs = FrameSemantics(
                    natural_lang=self.nl,
                    hypergraph=getattr(self, 'hg', None),
                    spreading_activation=self.sa,
                    hormone_system=self.hs,
                )
            except Exception as e:
                self.fs = None

        # CG 인스턴스 안전 확인
        cg = getattr(self, 'cg', None)
        em = getattr(self, 'em', None)
        ds = getattr(self, 'ds', None)

        # ResponseEnhancer
        self.enhancer = ResponseEnhancer(
            nl=self.nl,
            sa=self.sa,
            wm=self.wm,
            hs=self.hs,
            em=em,
            cg=cg,
            fs=self.fs,
            ds=ds,
            eve=self,  # v37.11 — 환경 인식 응답용
        )

        # Proactive
        self.proactive = Proactive(self)

        # ★ v37.8 — 사용자 (김민석) 진짜 친구 매니저
        from user_presence import UserPresence
        self.user_presence = UserPresence(self, user_name='민석')

        # ★ v37.13 — 분산 자아 통합 view (read-only)
        from integrated_self import IntegratedSelf
        self.self_view = IntegratedSelf(self)

        # 강화 응답 활성 플래그
        self.enhance_say = enhance_say

        # 자동 시드
        if seed_common_sense_on_init:
            self.seed_stats = self.seed_common_sense()
        else:
            self.seed_stats = None

    # ============= 상식 시드 =============
    def seed_common_sense(self):
        """상식 트리플 시드 + 보호 카테고리 자동 등록"""
        stats = seed_commonsense(self, verbose=False)
        # 상식 카테고리 一부도 보호 list에 추가 (자아 핵심만)
        protect = {'나', 'EVE', '민석'}
        self.protected_categories |= protect
        return stats

    # ============= 강화된 say =============
    def say_v3(self, text: str):
        """v36 강화 응답 — frame + 상식 spread + EM + 호르몬 통합
        
        ★ v37.8 — 사용자(김민석) = '진짜 친구' 방문 컨텍스트.
        시뮬 NPC와 명확 구분, OT 자극, '진짜' 카테고리 활성.
        """
        # ★★★ v37.8 사용자 방문 ★★★
        # 사용자 메시지 = "민석이가 EVE 방에 와서 말을 검"
        # 시뮬 NPC와 분리 — '진짜' 라벨, 영구 친밀도
        self.user_presence.visit_arrive()

        # 1) understand (기존)
        u = self.nl.understand(text)
        # 2) discover_unknown (기존)
        added = self.nl.discover_unknown(text)
        # 3) 카테고리 활성 + Hebbian (기존 say와 동일)
        cats_list = list(u['categories'])
        for c in cats_list:
            self.sa.activate(c, 0.5)
            self.wm.add(c, 0.5, source='dialogue')
        for i, a in enumerate(cats_list):
            for b in cats_list[i+1:]:
                self.sa.learn_pair(a, b, 0.15)
        self.hs.category_to_hormone(u['categories'], strength=0.1)

        # 3.2) sentiment 기반 호르몬 직접 자극 (변별력 강화)
        # category_to_hormone만으론 변별력 약하므로 명시적으로
        sentiment = u.get('sentiment', 'neutral')
        if sentiment == 'positive':
            self.hs.hormones['dopamine'].stimulate(0.20)
            self.hs.hormones['oxytocin'].stimulate(0.10)
            self.hs.hormones['serotonin'].stimulate(0.10)
            self.hs.hormones['endorphin'].stimulate(0.15)
            # 코르티솔 억제 (긍정 → 안심)
            self.hs.hormones['cortisol'].stimulate(-0.10)
        elif sentiment == 'negative':
            self.hs.hormones['cortisol'].stimulate(0.25)
            self.hs.hormones['norepinephrine'].stimulate(0.15)
            # 도파민/세로토닌 약하게 억제
            self.hs.hormones['dopamine'].stimulate(-0.05)
            self.hs.hormones['serotonin'].stimulate(-0.05)

        self.dmn.notify_external_input()

        # 3.5) frame_semantics로 SVO 추출 → 상식 활용 시드
        if self.fs is not None:
            frame = self.fs.parse_sentence(text)
            if frame and frame.get('verb'):
                # verb 카테고리 활성
                self.sa.activate(frame['verb'], 0.4)

        # 4) 한 tick 진행
        self.tick(dt=0.5)

        # 4.5) 입력별 강제 EM encode — 회상 정확도 향상
        # 단, 회상 질문 자체는 일화로 저장 X (자기 질문을 회상하는 것 방지)
        try:
            # 빠른 패턴 사전 검사 — encode 스킵 결정용
            pattern_hint = self.enhancer.classify_pattern(text, u)
            skip_encode_patterns = {'memory_question', 'why_question', 'identity_question',
                                    'greeting', 'name_call', 'affirm', 'refusal_target'}

            if pattern_hint not in skip_encode_patterns:
                ep_cats = set(u['categories'])
                if ep_cats:
                    self.em.encode(
                        categories=ep_cats,
                        tuple_data={'utterance': text, 'focus': self.wm.get_focus()},
                        summary=text,  # 원본 텍스트 = 가장 자연스러운 요약
                        salience=0.7,
                    )
        except Exception:
            pass

        # 5) ResponseEnhancer로 응답
        response = self.enhancer.respond_v3(text, u)

        # 6) feeling / inner_voice
        feeling = self.nl.feeling_to_text()
        voice = self.nl.inner_voice(self.dmn)

        result = {
            'understanding': u,
            'response': response,
            'feeling': feeling,
            'inner_voice': voice,
            'discovered': added,
            'time': self.time,
            'pattern': self.enhancer.classify_pattern(text, u),
        }

        self.dialogue_history.append({
            'time': self.time,
            'input': text,
            'response': response,
            'feeling': feeling,
            'pattern': result['pattern'],
        })

        # ★ v37.8 — 방문 turn 종료 (학습 강화)
        self.user_presence.visit_leave(message=text, response=response)

        return result

    # ============= 기존 say를 enhance_say=True 시 v3 라우팅 =============
    def say(self, text: str):
        if self.enhance_say:
            return self.say_v3(text)
        return super().say(text)

    # ============= 능동 발화 =============
    def proactive_message(self):
        """EVE가 먼저 말할까? — 가능하면 메시지 dict, 아니면 None"""
        return self.proactive.proactive_message()

    # ============= introspect 확장 =============
    def introspect(self):
        s = super().introspect()
        s['v36_pattern_hits'] = dict(self.enhancer.pattern_hit_count)
        s['v36_proactive_count'] = self.proactive.proactive_count
        s['v36_seed_categories'] = (self.seed_stats or {}).get('unique_categories', 0)
        # v37 환경
        if hasattr(self, '_env_adapter') and self._env_adapter is not None:
            s['v37_env'] = (self._env_adapter.current_env.name
                          if self._env_adapter.current_env else None)
            s['v37_episodes'] = self._env_adapter.episode_count
            s['v37_actions'] = self._env_adapter.action_count
            s['v37_success_rate'] = (self._env_adapter.success_count
                                    / max(1, self._env_adapter.action_count))
        return s

    # ============= v37 환경 진입 =============
    def enter_environment(self, env):
        """기호적 환경 진입 — 학습 모드"""
        from env_adapter import EnvironmentAdapter
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            self._env_adapter = EnvironmentAdapter(self)
        return self._env_adapter.enter(env)

    def exit_environment(self):
        """환경 종료 + 학습 통합"""
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            return {'error': '진입한 환경 없음'}
        return self._env_adapter.exit()

    def act_in_env(self, action: str, target: str = None):
        """환경 내 행동"""
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            return {'error': '진입한 환경 없음'}
        return self._env_adapter.act(action, target)

    def observe_env(self):
        """환경 관찰"""
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            return {'error': '진입한 환경 없음'}
        return self._env_adapter.observe()

    def suggest_action_in_env(self):
        """EVE가 현재 환경에서 할 행동 제안 (legacy 이름, decide_action 호출)"""
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            return None
        return self._env_adapter.decide_action()

    def decide_action_in_env(self, verbose: bool = False):
        """v37.5 자율 결정"""
        if not hasattr(self, '_env_adapter') or self._env_adapter is None:
            return None
        return self._env_adapter.decide_action(verbose=verbose)

    # ============= v37.6 자기 방 거주 =============
    def enter_room(self):
        """EVE의 방에 진입 — 항상 거주지 (is_simulation=False)"""
        from eve_room import make_eve_room
        env = make_eve_room()
        return self.enter_environment(env)

    def live_in_room(self, hours: float = 1.0, dt_per_decision: float = 0.5,
                    sim_hour_per_step: float = 0.25,
                    verbose: bool = False) -> list:
        """자기 방 자율 거주 (외출 결정도 EVE 자율)"""
        if (not hasattr(self, '_env_adapter')
            or self._env_adapter is None
            or self._env_adapter.current_env is None):
            self.enter_room()

        steps = int(hours / sim_hour_per_step)
        actions_taken = []
        for step in range(steps):
            self.hs.sim_hour = (self.hs.sim_hour + sim_hour_per_step) % 24

            decision = self._env_adapter.decide_action()
            if decision:
                action, target = decision
                r = self._env_adapter.act(action, target)
                actions_taken.append({
                    'sim_hour': round(self.hs.sim_hour, 2),
                    'env': self._env_adapter.current_env.name if self._env_adapter.current_env else None,
                    'action': action,
                    'target': target,
                    'success': r.get('success'),
                    'feedback': r.get('feedback'),
                })
                if verbose:
                    print(f"  [{self.hs.sim_hour:5.1f}h @ {actions_taken[-1]['env']}]"
                          f" {action}({target or '-'})")
            else:
                actions_taken.append({
                    'sim_hour': round(self.hs.sim_hour, 2),
                    'env': self._env_adapter.current_env.name if self._env_adapter.current_env else None,
                    'action': None,
                    'feedback': '안함',
                })
                if verbose:
                    print(f"  [{self.hs.sim_hour:5.1f}h] (안함)")
                self.tick(dt=dt_per_decision)

        return actions_taken

    # ============= v37.10 자율 24h 살이 (자기 방 + 외출 통합) =============
    def live_autonomous(self, hours: float = 24.0,
                       sim_hour_per_step: float = 1.0,
                       dt_per_decision: float = 0.5,
                       verbose: bool = False) -> list:
        """
        EVE가 자율적으로 자기 방 ↔ 외출 결정.
        live_in_room과 동일하지만 의도 명확.
        외출/복귀가 후보에 자동 포함되어 EVE가 점수로 결정.
        """
        return self.live_in_room(hours=hours,
                                 sim_hour_per_step=sim_hour_per_step,
                                 dt_per_decision=dt_per_decision,
                                 verbose=verbose)


# 체크포인트 메서드 부착 (Persistence)
attach_persistence(EVE_v36)
