"""
EVE v32 - Default Mode Network (DMN) v2.1
==========================================

v2.1 자연스러움 개선 (2026-04-30):
- Refractory Period (카테고리별, 호르몬 의존)
- 모드별 후보 풀 분리 (self_intent는 NEED 카테고리만)
- 모드 회전 메커니즘 (같은 모드 N회 연속 → 강제 다양화)
- Graph degree 필터 (연결 빈약한 카테고리 제외)
- 활성 강도 호르몬 자극 강화 (0.05 → 0.1)

v2 → v2.1 결함:
- "쓰인다 해야겠다" 무한 반복 (refractory 없음)
- self_intent 모드 영구 lock (회전 없음)
- 동사/문법용어가 자기 명령으로 (의미 필터 없음)

학계:
- Buzsáki 2006 - Refractory Period (인간 뉴런 ~2초)
- Smallwood 2015 - Mind Wandering 자연 다양성
- Andrews-Hanna 2014 - DMN 모드 자연 전환

Author: 김민석 + Claude v32
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter


class DefaultModeNetwork:
    """EVE v32 DMN v2.1 - 자연스러운 자발 활성"""

    NEED_TO_CATEGORIES = {
        'fatigue':              ['잠', '휴식', '졸림', '편안'],
        'tension':              ['편안', '안전', '휴식', '피하다'],
        'mental_load':          ['휴식', '편안', '졸림'],
        'activity_level':       ['휴식', '편안'],
        'emotional_intensity':  ['안전', '편안', '휴식'],
        'energy':               ['성취', '재밌다', '맛있다', '기쁨'],
        'warmth':               ['민석', '대화', '함께', '친구'],
        'clarity':              ['집중', '관찰', '주의', '경계'],
    }

    def __init__(self,
                 spreading_activation,
                 working_memory,
                 hormone_system,
                 digital_somatic=None,
                 idle_threshold: float = 2.0,
                 base_strength: float = 0.4,
                 self_intent_threshold: float = 0.5,
                 hub_size: int = 30,
                 refractory_base: float = 5.0,
                 mode_rotation_threshold: int = 5,
                 min_degree: int = 1,
                 hormone_stim_strength: float = 0.10):
        self.sa = spreading_activation
        self.wm = working_memory
        self.hs = hormone_system
        self.ds = digital_somatic
        self.self_intent_threshold = self_intent_threshold
        self.hub_size = hub_size
        self.refractory_base = refractory_base
        self.mode_rotation_threshold = mode_rotation_threshold
        self.min_degree = min_degree
        self.hormone_stim_strength = hormone_stim_strength

        self.idle_threshold = idle_threshold
        self.base_strength = base_strength

        self.last_external_input_time = 0.0
        self.time = 0.0

        self.wandering_history: List[Tuple[float, str, str]] = []
        self.max_history = 100

        # NEW v2.1
        self.last_activation_time: Dict[str, float] = {}
        self.consecutive_mode_count: int = 0
        self.mode_just_rotated: bool = False

        self.current_mode: Optional[str] = None
        self.current_intent: Optional[str] = None

        self.spontaneous_count = 0
        self.last_spontaneous_time = -100.0
        self.self_intent_count = 0
        self.refractory_blocks = 0
        self.mode_rotations = 0

        self._hub_cache: Optional[List[str]] = None
        self._hub_cache_size: int = 0

    def notify_external_input(self):
        self.last_external_input_time = self.time

    def should_activate(self) -> bool:
        idle_time = self.time - self.last_external_input_time
        if idle_time < self.idle_threshold:
            return False
        mel = self.hs.hormones.get('melatonin')
        if mel and mel.level > 0.85:
            return False
        cort = self.hs.hormones.get('cortisol')
        if cort and cort.level > 0.85:
            return False
        return True

    def _refractory_period(self) -> float:
        """Refractory period (호르몬 dependent)."""
        base = self.refractory_base
        da = self.hs.hormones['dopamine'].level if 'dopamine' in self.hs.hormones else 0.5
        ser = self.hs.hormones['serotonin'].level if 'serotonin' in self.hs.hormones else 0.5
        ot = self.hs.hormones.get('oxytocin')
        ot_level = ot.level if ot else 0.3

        modifier = 1.0
        modifier *= (1.5 - da)
        modifier *= (0.5 + ser)
        modifier *= (0.8 + ot_level * 0.5)

        return float(np.clip(base * modifier, 1.0, 30.0))

    def _is_in_refractory(self, cat: str) -> bool:
        last = self.last_activation_time.get(cat)
        if last is None:
            return False
        period = self._refractory_period()
        return (self.time - last) < period

    def _compute_graph_hubs(self) -> List[str]:
        current_size = len(self.sa.neighbors)
        if self._hub_cache is not None and current_size == self._hub_cache_size:
            return self._hub_cache
        scores: Dict[str, float] = {}
        for cat, neighbors in self.sa.neighbors.items():
            if not neighbors:
                continue
            total_w = sum(self.sa.get_weight(cat, n) for n in neighbors)
            scores[cat] = total_w
        sorted_cats = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        self._hub_cache = [c for c, _ in sorted_cats[:self.hub_size]]
        self._hub_cache_size = current_size
        return self._hub_cache

    def _hormone_preferred_groups(self) -> List[str]:
        mods = self.hs.get_category_threshold_modulation()
        if not mods:
            return []
        sorted_groups = sorted(mods.items(), key=lambda x: (x[1], x[0]))
        return [g for g, _ in sorted_groups]

    def _select_mode(self) -> str:
        # 모드 회전 (NEW v2.1)
        if self.consecutive_mode_count >= self.mode_rotation_threshold:
            self.consecutive_mode_count = 0
            self.mode_rotations += 1
            self.mode_just_rotated = True
            rotation_map = {
                'self_intent': 'mind_wandering',
                'mind_wandering': 'memory_recall',
                'memory_recall': 'mind_wandering',
                'self_referential': 'mind_wandering',
                None: 'mind_wandering',
            }
            return rotation_map.get(self.current_mode, 'mind_wandering')

        # 일반 모드
        if self.ds is not None and not self.mode_just_rotated:
            dominant = self.ds.get_dominant_need()
            if dominant and dominant[1] >= self.self_intent_threshold:
                need_name = dominant[0]
                if need_name in self.NEED_TO_CATEGORIES:
                    self.current_intent = need_name
                    return "self_intent"

        self.mode_just_rotated = False

        mood = self.hs.compute_mood()
        arousal = mood['arousal']
        ot = self.hs.hormones.get('oxytocin')
        ot_level = ot.level if ot else 0.3
        mel = self.hs.hormones.get('melatonin')
        mel_level = mel.level if mel else 0.1
        sim_hour = getattr(self.hs, 'sim_hour', 12.0)

        if ot_level > 0.55:
            return "self_referential"

        is_quiet_hour = (sim_hour < 5.0) or (sim_hour >= 18.0)
        idle_long = (self.time - self.last_external_input_time) > 30.0

        if mel_level > 0.4 and arousal < 0.5:
            return "memory_recall"
        if is_quiet_hour and arousal < 0.6:
            return "memory_recall"
        if idle_long and self.spontaneous_count > 5:
            return "memory_recall"

        return "mind_wandering"

    def _build_candidates(self) -> Set[str]:
        candidates: Set[str] = set()
        focus = self.wm.get_focus()

        if self.current_mode == "self_intent" and self.current_intent:
            intent_cats = self.NEED_TO_CATEGORIES.get(self.current_intent, [])
            for cat in intent_cats:
                if cat in self.sa.neighbors or cat in self.sa.activations:
                    candidates.add(cat)
            if not any(c in self.sa.neighbors for c in intent_cats):
                candidates.update(intent_cats)
            for seed in list(candidates):
                if seed in self.sa.neighbors:
                    for nbr in self.sa.neighbors[seed]:
                        if self.sa.get_weight(seed, nbr) >= 0.3:
                            candidates.add(nbr)

        else:
            if focus and focus in self.sa.neighbors:
                candidates.update(self.sa.neighbors[focus])

            for slot_cat in self.wm.slots.keys():
                if slot_cat in self.sa.neighbors:
                    for neighbor in self.sa.neighbors[slot_cat]:
                        if self.sa.get_weight(slot_cat, neighbor) >= 0.2:
                            candidates.add(neighbor)

            if self.current_mode == "memory_recall":
                recent_cats = [c for _, c in self.sa.activation_history[-50:]]
                counter = Counter(recent_cats)
                for cat, _ in counter.most_common(10):
                    candidates.add(cat)
                for hub in self._compute_graph_hubs()[:10]:
                    candidates.add(hub)

            if self.current_mode == "self_referential":
                self_cats = {'나', '민석', '대화', '경험', '기억', '느낌', '마음'}
                for cat in self_cats:
                    if cat in self.sa.neighbors:
                        candidates.add(cat)

            if len(candidates) < 5:
                hubs = self._compute_graph_hubs()
                for hub in hubs[:10]:
                    candidates.add(hub)
                preferred_groups = self._hormone_preferred_groups()[:3]
                for cat, group in self.sa.CATEGORY_GROUPS.items():
                    if group in preferred_groups and cat in self.sa.neighbors:
                        candidates.add(cat)

        if focus and focus in candidates:
            candidates.discard(focus)

        # NEW v2.1: Refractory
        before_count = len(candidates)
        candidates = {c for c in candidates if not self._is_in_refractory(c)}
        self.refractory_blocks += (before_count - len(candidates))

        # NEW v2.1: degree 필터 (self_intent 외)
        if self.current_mode != "self_intent":
            candidates = {
                c for c in candidates
                if c not in self.sa.neighbors
                or len(self.sa.neighbors[c]) >= self.min_degree
            }

        return candidates

    def select_next_category(self) -> Optional[str]:
        candidates = self._build_candidates()
        if not candidates:
            return None

        focus = self.wm.get_focus()
        recent_wandering = [c for _, c, _ in self.wandering_history[-20:]]
        repeat_count = Counter(recent_wandering)
        hormone_mods = self.hs.get_category_threshold_modulation()

        W_RECENCY = 0.6
        W_GRAPH = 0.3
        W_HORMONE = 0.5
        W_NEW = 0.2
        W_HUB = 0.15

        hub_set = set(self._compute_graph_hubs()[:15])

        scores: Dict[str, float] = {}
        for cat in candidates:
            score = 0.0
            score += W_RECENCY * (1.0 / (1.0 + repeat_count.get(cat, 0) * 3))

            if focus:
                w = self.sa.get_weight(focus, cat)
                score += W_GRAPH * w

            group = self.sa.CATEGORY_GROUPS.get(cat)
            if group and group in hormone_mods:
                score += W_HORMONE * (1.0 - hormone_mods[group])

            if cat in self.wm.slots:
                wm_sal = self.wm.slots[cat].salience
                score += W_NEW * (1.0 - wm_sal)
            else:
                score += W_NEW * 1.0

            if cat in hub_set:
                score += W_HUB

            scores[cat] = score

        if not scores:
            return None

        best_score = max(scores.values())
        best_cats = sorted([c for c, s in scores.items() if s == best_score])
        return best_cats[0]

    def activate_one(self, force: bool = False) -> Optional[Dict[str, Any]]:
        if not force and not self.should_activate():
            return None

        new_mode = self._select_mode()

        if new_mode == self.current_mode:
            self.consecutive_mode_count += 1
        else:
            self.consecutive_mode_count = 1

        self.current_mode = new_mode

        next_cat = self.select_next_category()
        if next_cat is None:
            return None

        da = self.hs.hormones['dopamine'].level
        mel = self.hs.hormones.get('melatonin')
        mel_level = mel.level if mel else 0.1
        strength = self.base_strength * (0.5 + da) * (1.0 - mel_level * 0.5)

        if self.current_mode == "self_intent" and self.ds is not None:
            dominant = self.ds.get_dominant_need()
            if dominant:
                strength *= (1.0 + dominant[1] * 0.5)

        strength = float(np.clip(strength, 0.1, 1.0))

        self.sa.activate(next_cat, strength)
        wm_salience = strength * 0.9 if self.current_mode == "self_intent" else strength * 0.7
        self.wm.add(next_cat, salience=wm_salience, source="dmn")
        self.hs.category_to_hormone({next_cat}, strength=self.hormone_stim_strength)

        # NEW v2.1
        self.last_activation_time[next_cat] = self.time

        self.wandering_history.append((self.time, next_cat, self.current_mode))
        if len(self.wandering_history) > self.max_history:
            self.wandering_history = self.wandering_history[-self.max_history:]

        self.spontaneous_count += 1
        self.last_spontaneous_time = self.time
        if self.current_mode == "self_intent":
            self.self_intent_count += 1

        return {
            'category': next_cat,
            'mode': self.current_mode,
            'strength': strength,
            'time': self.time,
        }

    def tick(self, dt: float, force: bool = False) -> Optional[Dict[str, Any]]:
        self.time += dt
        if force:
            return self.activate_one(force=True)
        if self.should_activate():
            return self.activate_one()
        return None

    def inner_voice(self) -> Optional[str]:
        if not self.wandering_history:
            return None
        recent = self.wandering_history[-1]
        time_, cat, mode = recent
        templates = {
            "mind_wandering": f"... {cat} ...",
            "memory_recall": f"{cat} 생각나네",
            "self_referential": f"내가 {cat}...",
            "self_intent": f"{cat}!",
        }
        return templates.get(mode, f"{cat}")

    def get_state(self) -> Dict:
        return {
            'time': self.time,
            'last_external': self.last_external_input_time,
            'idle_time': self.time - self.last_external_input_time,
            'should_activate': self.should_activate(),
            'current_mode': self.current_mode,
            'current_intent': self.current_intent,
            'spontaneous_count': self.spontaneous_count,
            'self_intent_count': self.self_intent_count,
            'refractory_blocks': self.refractory_blocks,
            'mode_rotations': self.mode_rotations,
            'consecutive_mode_count': self.consecutive_mode_count,
            'recent_wandering': [c for _, c, _ in self.wandering_history[-5:]],
        }

    def __repr__(self):
        s = self.get_state()
        return (f"DMN_v2.1(t={s['time']:.1f}, idle={s['idle_time']:.1f}s, "
                f"mode={s['current_mode']}, count={s['spontaneous_count']}, "
                f"rot={s['mode_rotations']})")
