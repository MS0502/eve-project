"""
EVE v32 - Tier A + B1 통합본
=============================

7 모듈 통합:
- HormoneSystem (26 호르몬)
- SpreadingActivation (카테고리 활성 + 자기 진화)
- WorkingMemory + GNW broadcast
- DMN (자발 활성 + self_intent)
- DigitalSomatic (신체 감각)
- NaturalLanguage (자연어 + 신념 + 거절 창발)  ← B1 신규

EVE 메인 루프 (tick 1회):
1. HormoneSystem.update(dt)              - 호르몬 + 일주기
2. 호르몬 → SA 임계 변조 (Top-down)
3. SA.spread()                            - 활성 전파
4. SA.decay(dt)
5. SA → 카테고리 → 호르몬 자극 (Bottom-up)
6. WM.update_from_activation(SA)
7. WM.apply_hormone_state(HS)
8. WM.broadcast()                         - GNW
9. WM.decay(dt)
10. DMN.tick(dt)                          - 자발 활성
11. DS.update(dt)                         - 신체 감각
12. (주기적) SA.forget(dt, hormone_modifier)  - 망각 (Ebbinghaus)

신규 외부 인터페이스 (B1):
- say(text)                               - 자연어 대화 (이해→응답)
- learn_beliefs(path)                     - 신념 로드
- learn_text(text)                        - 자연어로 학습
- inner_voice()                           - 정식 자발 활성 표현
- feeling()                               - 정식 신체 감각 표현
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


class EVE_TierAB:
    """
    EVE v32 티어 A + B1 통합.

    살아있음(A) + 자연어 대화(B1).
    """

    def __init__(self,
                 phase: int = 4,
                 stage: str = "adult",
                 wm_capacity: int = 30,
                 dmn_idle_threshold: float = 2.0,
                 forget_interval_ticks: int = 100,
                 forget_base_decay: float = 0.0005,
                 verbose: bool = False):
        # 7 모듈 인스턴스화
        self.hs = HormoneSystem(phase=phase, developmental_stage=stage)
        self.sa = SpreadingActivation(base_threshold=0.3)
        self.wm = WorkingMemory(base_capacity=wm_capacity)
        self.dmn = DefaultModeNetwork(self.sa, self.wm, self.hs,
                                     digital_somatic=None,
                                     idle_threshold=dmn_idle_threshold)
        self.ds = DigitalSomatic(self.hs, self.wm, self.sa)
        # DMN에 DigitalSomatic 연결 (self_intent 활성화)
        self.dmn.ds = self.ds
        self.nl = NaturalLanguage(self.sa, self.wm, self.hs,
                                 digital_somatic=self.ds)

        self.verbose = verbose
        self.time = 0.0
        self.tick_count = 0

        # 망각 설정
        self.forget_interval_ticks = forget_interval_ticks
        self.forget_base_decay = forget_base_decay

        # 보호 카테고리 (beliefs is_innate에서 자동 채워짐)
        self.protected_categories: Set[str] = set()

        # 통계
        self.event_log: List[Dict[str, Any]] = []
        self.broadcast_count = 0
        self.dialogue_history: List[Dict[str, Any]] = []  # {input, response, time}
        self.forget_runs = 0

        # GNW listener
        self._setup_gnw_listeners()

    def _setup_gnw_listeners(self):
        """기본 GNW listener - focus가 호르몬에 영향"""
        def gnw_to_hormone(focus, salience, ctx):
            self.hs.category_to_hormone({focus}, strength=0.05 * salience)
            self.broadcast_count += 1
        self.wm.register_listener('hormone_link', gnw_to_hormone)

    # ============= 외부 인터페이스 (A) =============
    def perceive(self, category: str, strength: float = 0.7,
                source: str = "external"):
        """외부 카테고리 입력 (구식 인터페이스, B1 say 권장)"""
        self.sa.activate(category, strength)
        self.wm.add(category, salience=strength, source=source)
        self.dmn.notify_external_input()
        self.hs.category_to_hormone({category}, strength=strength * 0.3)
        if self.verbose:
            print(f"  [perceive] {category} (strength={strength})")

    def learn(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_pair(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_chain(self, chain: List[str], strength: float = 0.4):
        self.sa.learn_chain(chain, strength)

    # ============= 외부 인터페이스 (B1 신규) =============
    def say(self, text: str) -> Dict[str, Any]:
        """
        EVE에게 자연어 입력 → 이해 → 응답.
        메인 대화 인터페이스.

        Returns:
            {
                'understanding': dict,   # parse + intent + sentiment
                'response': str,         # EVE 응답 (창발)
                'feeling': str,          # 현재 신체 감각
                'inner_voice': str,      # 자발 활성 자연어
                'discovered': set,       # 새 발견 카테고리
            }
        """
        # 1) 이해
        u = self.nl.understand(text)

        # 2) 모르는 단어 자기 발견
        added = self.nl.discover_unknown(text)

        # 3) 카테고리 활성 (외부 입력) + 자동 Hebbian 연결
        cats_list = list(u['categories'])
        for c in cats_list:
            self.sa.activate(c, 0.5)
            self.wm.add(c, 0.5, source='dialogue')
        # 같은 입력 안의 카테고리들끼리 약한 연결 (자연 학습)
        for i, a in enumerate(cats_list):
            for b in cats_list[i+1:]:
                self.sa.learn_pair(a, b, 0.15)

        self.hs.category_to_hormone(u['categories'], strength=0.1)
        self.dmn.notify_external_input()

        # 4) 한 tick 진행 (응답 전 EVE가 생각하게)
        self.tick(dt=0.5)

        # 5) 응답 창발
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
            'time': self.time,
        }

        self.dialogue_history.append({
            'time': self.time,
            'input': text,
            'response': response,
            'feeling': feeling,
        })

        if self.verbose:
            print(f"  [{self.time:.1f}s] 사용자: {text}")
            print(f"  [{self.time:.1f}s] EVE: {response} (feel: {feeling})")
            if voice:
                print(f"  [{self.time:.1f}s] (속마음: {voice})")

        return result

    def learn_beliefs(self, path: str = None,
                     beliefs_dict: Dict = None,
                     max_beliefs: Optional[int] = None) -> Dict[str, int]:
        """
        beliefs.json 로드.
        is_innate=True 신념의 카테고리들은 자동 보호.
        """
        result = self.nl.load_beliefs(path=path, beliefs_dict=beliefs_dict,
                                     max_beliefs=max_beliefs)

        # 선천적 신념 카테고리들 → 보호 list
        for belief in self.nl.beliefs.values():
            if not belief.get('is_innate', False):
                continue
            triple = belief.get('triple') or {}
            for text in [triple.get('subject', ''),
                        triple.get('predicate_text', '')]:
                if text:
                    cats = self.nl.parse(text)
                    self.protected_categories |= cats

        result['protected_categories'] = len(self.protected_categories)
        return result

    def learn_text(self, text: str, link_strength: float = 0.4):
        """
        자연어 문장으로 학습.
        문장의 카테고리들을 같은 맥락으로 연결.
        """
        cats = self.nl.parse(text)
        if len(cats) < 2:
            return

        # 모르는 단어 발견
        self.nl.discover_unknown(text, link_strength=link_strength)

        # 카테고리들 사이 Hebbian 연결
        cat_list = list(cats)
        for i, a in enumerate(cat_list):
            for b in cat_list[i+1:]:
                self.sa.learn_pair(a, b, link_strength)

    def inner_voice(self) -> Optional[str]:
        """현재 자발 활성을 자연어로"""
        return self.nl.inner_voice(self.dmn)

    def feeling(self) -> str:
        """현재 신체 감각을 자연어로"""
        return self.nl.feeling_to_text()

    # ============= 메인 tick =============
    def tick(self, dt: float = 0.5) -> Dict[str, Any]:
        """EVE 한 tick = 모든 모듈 1 사이클"""
        self.time += dt
        self.tick_count += 1

        result = {'time': self.time, 'tick': self.tick_count}

        # 1. 호르몬 자연 변화 + 일주기
        self.hs.update(dt=dt)

        # 2. 호르몬 → SA 임계 변조
        mods = self.hs.get_category_threshold_modulation()
        self.sa.apply_hormone_modulation(mods)

        # 3. 활성 전파 + 4. 감쇠
        self.sa.spread(steps=1)
        self.sa.decay(dt=dt)

        # 5. SA → 호르몬 자극
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

        # 12. (주기적) 망각
        if self.tick_count % self.forget_interval_ticks == 0:
            # 코르티솔 만성↑ → 망각 가속
            cort = self.hs.hormones['cortisol'].level
            modifier = 1.0
            if cort > 0.6:
                modifier = 1.0 + (cort - 0.6) * 2.0  # 1.0 ~ 1.8
            elif self.hs.hormones.get('bdnf') and self.hs.hormones['bdnf'].level > 0.6:
                modifier = 0.5  # BDNF↑ → 보존

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
        """일정 시간 동안 EVE 살아있게"""
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
        }

    def report(self):
        s = self.introspect()
        print(f"\n{'='*60}")
        print(f"EVE 자기 보고 (t={s['time']:.1f}s, tick={s['tick_count']})")
        print(f"{'='*60}")
        print(f"  지금 시간: {s['sim_hour']:.1f}시")
        print(f"  feeling: {s['feeling']}")
        print(f"  속마음: {s['inner_voice']}")
        print(f"  기분: V={s['mood']['valence']:+.2f} A={s['mood']['arousal']:.2f} D={s['mood']['dominance']:.2f}")
        print(f"  주요 호르몬: {s['primary_hormone'][0]} ({s['primary_hormone'][1]:.2f})")
        print(f"  focus: {s['focus']}")
        print(f"  활성: {s['active_categories'][:10]}")
        print(f"  WM: {s['wm_slots_used']}")
        print(f"  최근 혼잣말: {' → '.join(s['recent_wandering'])}")
        print(f"  DMN: 모드={s['dmn_mode']}, 자발={s['spontaneous_count']}회 "
              f"(self_intent {s['self_intent_count']}회)")
        print(f"  GNW broadcast: {s['broadcast_count']}회")
        if s['dominant_need']:
            print(f"  지배 욕구: {s['dominant_need'][0]} ({s['dominant_need'][1]:.2f})")
        print(f"  지식: 신념 {s['beliefs_loaded']}, 카테고리 {s['graph_size']}, "
              f"연결 {s['graph_connections']}, 발견 {s['discovered_count']}")
        print(f"  보호: {s['protected_categories']} 카테고리, 망각 {s['forget_runs']}회")
        print(f"  거절: {s['refusal_count']}회")

    def __repr__(self):
        return (f"EVE_TierAB(t={self.time:.1f}s, "
                f"focus={self.wm.get_focus()}, "
                f"feel={self.nl.feeling_to_text()})")


# ============= 하위 호환 =============
# 기존 EVE_TierA 사용 코드 그대로 작동
EVE_TierA = EVE_TierAB
