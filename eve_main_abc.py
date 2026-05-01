"""
EVE v32 - Tier A + B 통합본 (B3 SelfDoubt 통합)
================================================

8 모듈 통합:
- HormoneSystem (26 호르몬)
- SpreadingActivation (카테고리 활성 + 자기 진화)
- WorkingMemory + GNW broadcast
- DMN (자발 활성 + self_intent)
- DigitalSomatic (신체 감각)
- NaturalLanguage (자연어 + 신념 + 거절 창발)
- SelfDoubt (5종 신호 통합 의심) ← B3 신규 통합

EVE 메인 루프 (tick 1회):
1. HormoneSystem.update(dt)
2. 호르몬 → SA 임계 변조 (Top-down)
3. SA.spread()
4. SA.decay(dt)
5. SA → 카테고리 → 호르몬 자극 (Bottom-up)
6. WM.update_from_activation(SA)
7. WM.apply_hormone_state(HS)
8. WM.broadcast()                         - GNW
9. WM.decay(dt)
10. DMN.tick(dt)
11. DS.update(dt)
12. SD.update(dt)                          ← NEW (baseline 시간 회귀)
13. (주기적) SA.forget(dt, hormone_modifier)

신규 외부 인터페이스 (B3 통합):
- say(text)에서 SelfDoubt가 명령/주장 평가 → action 카테고리 SA broadcast
  → NL.respond가 의심/거절 활성 상태에서 자연스러운 어조로 응답 (창발)
- result['doubt']에 SD 평가 결과 포함
- introspect()에 doubt baseline/reject_count 추가
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eve_modules'))

from typing import Dict, List, Set, Tuple, Optional, Any
from hormone_system import HormoneSystem
from spreading_activation import SpreadingActivation
from working_memory import WorkingMemory
from dmn import DefaultModeNetwork
from digital_somatic import DigitalSomatic
from natural_lang import NaturalLanguage
from self_doubt import SelfDoubt  # ← NEW B3


class EVE_TierAB:
    """
    EVE v32 티어 A + B (B1 + B3) 통합.

    살아있음(A) + 자연어 대화(B1) + 자기 의심(B3).
    """

    # SelfDoubt action → SA에 broadcast할 카테고리 + 강도
    # 창발 어조를 위해 NL.respond 직전에 활성
    DOUBT_ACTION_BROADCAST = {
        'accept':   [],  # 활성 X (NL 자연 응답 그대로)
        'question': [('의심', 0.5), ('궁금', 0.4)],
        'defer':    [('의심', 0.6), ('회피', 0.4)],
        'reject':   [('거절', 0.7), ('싫다', 0.5), ('의심', 0.5)],
    }

    def __init__(self,
                 phase: int = 4,
                 stage: str = "adult",
                 wm_capacity: int = 30,
                 dmn_idle_threshold: float = 2.0,
                 forget_interval_ticks: int = 100,
                 forget_base_decay: float = 0.0005,
                 verbose: bool = False):
        # 7 + 1 모듈 인스턴스화
        self.hs = HormoneSystem(phase=phase, developmental_stage=stage)
        self.sa = SpreadingActivation(base_threshold=0.3)
        self.wm = WorkingMemory(base_capacity=wm_capacity)
        self.dmn = DefaultModeNetwork(self.sa, self.wm, self.hs,
                                     digital_somatic=None,
                                     idle_threshold=dmn_idle_threshold)
        self.ds = DigitalSomatic(self.hs, self.wm, self.sa)
        self.dmn.ds = self.ds
        self.nl = NaturalLanguage(self.sa, self.wm, self.hs,
                                 digital_somatic=self.ds)

        # ★ B3 신규: SelfDoubt
        self.sd = SelfDoubt(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            natural_lang=self.nl,
            episodic_memory=None,  # B2 통합 시 self.em 으로 교체
            digital_somatic=self.ds,
        )

        self.verbose = verbose
        self.time = 0.0
        self.tick_count = 0

        self.forget_interval_ticks = forget_interval_ticks
        self.forget_base_decay = forget_base_decay

        self.protected_categories: Set[str] = set()

        self.event_log: List[Dict[str, Any]] = []
        self.broadcast_count = 0
        self.dialogue_history: List[Dict[str, Any]] = []
        self.forget_runs = 0

        self._setup_gnw_listeners()

    def _setup_gnw_listeners(self):
        def gnw_to_hormone(focus, salience, ctx):
            self.hs.category_to_hormone({focus}, strength=0.05 * salience)
            self.broadcast_count += 1
        self.wm.add_broadcast_listener(gnw_to_hormone)

    # ============= 외부 입력 =============
    def perceive(self, category: str, strength: float = 0.7,
                 source: str = 'external'):
        self.sa.activate(category, strength)
        self.wm.add(category, strength, source=source)
        self.hs.category_to_hormone({category}, strength=0.1)
        self.dmn.notify_external_input()

    def learn(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_pair(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_chain(self, chain: List[str], strength: float = 0.4):
        self.sa.learn_chain(chain, strength)

    # ============= 자연어 대화 (B3 통합) =============
    def say(self, text: str) -> Dict[str, Any]:
        """
        EVE에게 자연어 입력 → 이해 → 의심 평가 → 응답.

        B3 통합:
        - intent에 따라 evaluate_command / evaluate_claim 호출
        - action(accept/question/defer/reject) → 의심/거절 카테고리 SA 활성
        - NL.respond가 그 활성 상태에서 자연스러운 어조로 응답 (창발)

        Returns:
            기존 + {'doubt': SD 평가 결과}
        """
        # 1) 이해
        u = self.nl.understand(text)

        # 2) 모르는 단어 자기 발견
        added = self.nl.discover_unknown(text)

        # 3) 카테고리 활성 + Hebbian
        cats_list = list(u['categories'])
        for c in cats_list:
            self.sa.activate(c, 0.5)
            self.wm.add(c, 0.5, source='dialogue')
        for i, a in enumerate(cats_list):
            for b in cats_list[i+1:]:
                self.sa.learn_pair(a, b, 0.15)

        self.hs.category_to_hormone(u['categories'], strength=0.1)
        self.dmn.notify_external_input()

        # 4) 한 tick 진행 (응답 전 EVE가 생각하게)
        self.tick(dt=0.5)

        # ★ B3: 의심 평가 (intent에 따라 분기)
        intent = u.get('intent', '')
        is_command = intent in ('command', 'request', 'ask')
        if is_command:
            doubt_result = self.sd.evaluate_command(text, parsed=u)
        else:
            doubt_result = self.sd.evaluate_claim(text, parsed=u)

        # ★ B3: action → 카테고리 broadcast (창발 어조)
        action = doubt_result['recommended_action']
        for cat, strength in self.DOUBT_ACTION_BROADCAST.get(action, []):
            self.sa.activate(cat, strength)
            self.wm.add(cat, strength * 0.6, source='self_doubt')

        # 5) 응답 창발 (의심/거절 활성 상태에서 NL이 응답 → 어조 자연스럽게 변함)
        response = self.nl.respond(u)

        # 6) feeling
        feeling = self.nl.feeling_to_text()

        # 7) inner_voice
        voice = self.nl.inner_voice(self.dmn)

        result = {
            'understanding': u,
            'response': response,
            'feeling': feeling,
            'inner_voice': voice,
            'discovered': added,
            'doubt': doubt_result,  # ★ NEW
            'time': self.time,
        }

        self.dialogue_history.append({
            'time': self.time,
            'input': text,
            'response': response,
            'feeling': feeling,
            'doubt_action': action,
            'doubt_level': doubt_result['doubt_level'],
        })

        if self.verbose:
            print(f"  [{self.time:.1f}s] 사용자: {text}")
            print(f"  [{self.time:.1f}s] EVE: {response} (feel: {feeling})")
            print(f"  [{self.time:.1f}s] (의심: {doubt_result['doubt_level']:.2f} "
                  f"→ {action}, sources={doubt_result['doubt_sources']})")
            if voice:
                print(f"  [{self.time:.1f}s] (속마음: {voice})")

        return result

    def learn_beliefs(self, path: str = None,
                     beliefs_dict: Dict = None,
                     max_beliefs: Optional[int] = None) -> Dict[str, int]:
        result = self.nl.load_beliefs(path=path, beliefs_dict=beliefs_dict,
                                     max_beliefs=max_beliefs)
        # 보호 카테고리 자동 등록
        for belief in self.nl.beliefs.values():
            if belief.is_innate:
                triple = belief.triple
                if triple:
                    if triple.subject:
                        self.protected_categories.add(triple.subject)
                    if triple.object_:
                        self.protected_categories.add(triple.object_)

        # 결과에 보호 카테고리 수 추가
        result['protected_categories'] = len(self.protected_categories)
        result['connections_made'] = len(self.sa.weights)
        return result

    def learn_text(self, text: str, link_strength: float = 0.4):
        cats = self.nl.parse(text)
        if not cats:
            return
        # 카테고리 학습 (페어로 연결)
        cats_list = list(cats)
        for i, a in enumerate(cats_list):
            for b in cats_list[i+1:]:
                self.sa.learn_pair(a, b, link_strength)
        # 활성도 약하게
        for c in cats_list:
            self.sa.activate(c, 0.3)

    def inner_voice(self) -> Optional[str]:
        return self.nl.inner_voice(self.dmn)

    def feeling(self) -> str:
        return self.nl.feeling_to_text()

    # ============= 메인 루프 =============
    def tick(self, dt: float = 0.5) -> Dict[str, Any]:
        """EVE 한 tick = 모든 모듈 1 사이클"""
        self.time += dt
        self.tick_count += 1

        result = {'time': self.time, 'tick': self.tick_count}

        # 1. 호르몬
        self.hs.update(dt=dt)

        # 2. 호르몬 → SA 임계 변조
        mods = self.hs.get_category_threshold_modulation()
        self.sa.apply_hormone_modulation(mods)

        # 3-4. spread + decay
        self.sa.spread(steps=1)
        self.sa.decay(dt=dt)

        # 5. SA → 호르몬
        active_cats = self.sa.get_active()
        if active_cats:
            self.hs.category_to_hormone(active_cats, strength=0.05)

        # 6-9. WM + GNW
        self.wm.update_from_activation(self.sa)
        self.wm.apply_hormone_state(self.hs.get_state())
        broadcast_result = self.wm.broadcast()
        if broadcast_result:
            result['broadcast'] = broadcast_result
        self.wm.decay(dt=dt)

        # 10. DMN
        dmn_result = self.dmn.tick(dt=dt)
        if dmn_result:
            result['spontaneous'] = dmn_result
            if self.verbose:
                voice = self.nl.inner_voice(self.dmn)
                print(f"  [{self.time:5.1f}s] [{dmn_result['mode']:18s}] "
                      f"{dmn_result['category']:15s} → '{voice}'")

        # 11. DigitalSomatic
        self.ds.update(dt=dt)
        result['feeling'] = self.ds.get_feeling()

        # 12. ★ B3 신규: SelfDoubt baseline 시간 회귀
        self.sd.update(dt=dt)

        # 13. (주기적) 망각
        if self.tick_count % self.forget_interval_ticks == 0:
            cort = self.hs.hormones['cortisol'].level
            modifier = 1.0
            if cort > 0.6:
                modifier = 1.0 + (cort - 0.6) * 2.0
            elif self.hs.hormones.get('bdnf') and self.hs.hormones['bdnf'].level > 0.6:
                modifier = 0.5

            forget_result = self.sa.forget(
                dt=dt * self.forget_interval_ticks,
                base_decay=self.forget_base_decay,
                hormone_modifier=modifier,
                protected_categories=self.protected_categories,
            )
            self.forget_runs += 1
            if forget_result.get('pairs_removed', 0) or forget_result.get('categories_removed', 0):
                result['forget'] = forget_result
                if self.verbose:
                    print(f"  [{self.time:5.1f}s] [forget] "
                          f"pairs -{forget_result['pairs_removed']}, "
                          f"cats -{forget_result['categories_removed']}")

        return result

    def live(self, duration: float, dt: float = 0.5,
            log_every: int = 0) -> List[Dict]:
        results = []
        steps = int(duration / dt)
        for step in range(steps):
            r = self.tick(dt=dt)
            results.append(r)
            if log_every > 0 and step % log_every == 0:
                self._print_status_brief()
        return results

    def _print_status_brief(self):
        primary = self.hs.primary_hormone()
        focus = self.wm.get_focus()
        feel = self.nl.feeling_to_text()
        print(f"    t={self.time:5.1f}s | focus={focus} | feel={feel} | "
              f"primary={primary[0]}({primary[1]:.2f})")

    # ============= 자기 보고 =============
    def introspect(self) -> Dict[str, Any]:
        mood = self.hs.compute_mood()
        primary_h = self.hs.primary_hormone()
        focus = self.wm.get_focus()
        active_cats = self.sa.get_active()
        recent_wandering = [c for _, c, _ in self.dmn.wandering_history[-5:]]
        dominant_need = self.ds.get_dominant_need()

        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'feeling': self.nl.feeling_to_text(),
            'inner_voice': self.nl.inner_voice(self.dmn),
            'mood': mood,
            'primary_hormone': primary_h,
            'focus': focus,
            'active_categories': sorted(active_cats),
            'wm_slots_used': f"{len(self.wm.slots)}/{self.wm.current_capacity}",
            'recent_wandering': recent_wandering,
            'dmn_mode': self.dmn.current_mode,
            'spontaneous_count': self.dmn.spontaneous_count,
            'self_intent_count': self.dmn.self_intent_count,
            'broadcast_count': self.broadcast_count,
            'dominant_need': dominant_need,
            'sim_hour': self.hs.sim_hour,
            'beliefs_loaded': len(self.nl.beliefs),
            'discovered_count': self.nl.discovered_count,
            'refusal_count': self.nl.refusal_count,
            'forget_runs': self.forget_runs,
            'protected_categories': len(self.protected_categories),
            'graph_size': len(self.sa.neighbors),
            'graph_connections': len(self.sa.weights),
            # ★ B3 신규
            'doubt_baseline': self.sd.baseline,
            'doubt_baseline_with_hormone': self.sd.get_baseline_doubt(),
            'doubt_eval_count': self.sd.eval_count,
            'doubt_reject_count': self.sd.reject_count,
            'last_doubt_action': (self.sd.last_doubt['recommended_action']
                                  if self.sd.last_doubt else None),
        }

    def report(self):
        intro = self.introspect()
        print(f"\n{'=' * 60}")
        print(f"EVE 상태 보고 @ t={intro['time']:.1f}s")
        print(f"{'=' * 60}")
        print(f"feeling      : {intro['feeling']}")
        print(f"inner_voice  : {intro['inner_voice']}")
        print(f"mood         : v={intro['mood']['valence']:+.2f} "
              f"a={intro['mood']['arousal']:.2f} d={intro['mood']['dominance']:.2f}")
        print(f"primary horm : {intro['primary_hormone'][0]} "
              f"({intro['primary_hormone'][1]:.2f})")
        print(f"focus        : {intro['focus']}")
        print(f"active cats  : {len(intro['active_categories'])} "
              f"{intro['active_categories'][:8]}")
        print(f"WM           : {intro['wm_slots_used']}")
        print(f"DMN          : {intro['dmn_mode']} "
              f"(spont={intro['spontaneous_count']}, "
              f"self_intent={intro['self_intent_count']})")
        print(f"recent wander: {intro['recent_wandering']}")
        print(f"dominant_need: {intro['dominant_need']}")
        print(f"sim_hour     : {intro['sim_hour']:.1f}h")
        print(f"beliefs      : {intro['beliefs_loaded']} "
              f"(discovered={intro['discovered_count']}, "
              f"refusals={intro['refusal_count']})")
        print(f"graph        : {intro['graph_size']} cats, "
              f"{intro['graph_connections']} pairs "
              f"(forget runs={intro['forget_runs']})")
        # ★ B3
        print(f"doubt        : baseline={intro['doubt_baseline']:.2f}, "
              f"with_horm={intro['doubt_baseline_with_hormone']:.2f}, "
              f"evals={intro['doubt_eval_count']}, "
              f"rejects={intro['doubt_reject_count']}, "
              f"last={intro['last_doubt_action']}")
        print(f"{'=' * 60}\n")

    def __repr__(self):
        return (f"EVE_TierAB(t={self.time:.1f}s, "
                f"horm={self.hs.primary_hormone()[0]}, "
                f"focus={self.wm.get_focus()}, "
                f"doubt={self.sd.baseline:.2f})")


# 하위 호환 alias
EVE_TierA = EVE_TierAB
