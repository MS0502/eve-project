"""
EVE v32 - D4 MultiStream (평행 사고)
=====================================

5+ 개의 인지 stream이 평행하게 돌고, GNW가 한 번에 하나만 focus로 broadcast.
다른 stream은 background (의식 X) — 가끔 cross-talk으로 영감.

학계:
- William James 1890 — Stream of consciousness
- Baars 1988 — Global Workspace Theory (GWT)
- Dehaene & Naccache 2001 — GNW (의식의 신경 작업공간)
- Kahneman 2011 — System 1/2 (자동 vs 통제)
- Bargh & Chartrand 1999 — Automaticity
- Tononi 2008 — IIT (Integrated Information)

EVE 원칙 부합:
- 확률 X (bid 점수 결정론, 정렬 tie-break)
- 카테고리 = 의미 단위 (각 stream의 큐)
- 자발 활성 = inner stream (DMN 위임)
- 호르몬 변조 = stream 별 affinity (bid 가중치)
- 다른 모듈 read-only 우선 (stream은 자기 큐만 변경)

5 Streams:
- EXTERNAL: say() 입력 → 카테고리 큐
- INNER:    DMN 자발 활성 결과
- EMOTION:  호르몬 따라 활성 카테고리
- PLANNING: 미래 인과 시뮬 (TM/GM)
- RECALL:   EM 자동 회상

핵심 함수 (7개):
1. push_to_stream(name, cat, salience) — 외부 push
2. emotion_stream_tick()                — 호르몬 → 감정 카테고리 큐 채움
3. planning_stream_tick()               — 미래 시뮬 큐 채움
4. recall_stream_tick()                 — 회상 큐 채움
5. compete_for_focus()                  — bid 비교 → focus 결정 (결정론)
6. cross_talk(focus_cat)                — 다른 stream 약한 활성
7. integrate(dt)                        — 한 사이클 통합

Author: 김민석 + Claude v32 (D4)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque, defaultdict


class Stream:
    """단일 stream — 카테고리 큐 + bid + affinity"""

    def __init__(self, name: str,
                 base_priority: float = 0.5,
                 max_size: int = 10,
                 hormone_affinity: Optional[Dict[str, float]] = None):
        self.name = name
        self.base_priority = base_priority
        self.queue: deque = deque(maxlen=max_size)  # [(time, cat, salience), ...]
        # 호르몬 → bid 가중치 (없으면 1.0)
        self.hormone_affinity = hormone_affinity or {}
        self.active_count = 0  # focus 됐던 횟수
        self.push_count = 0

    def push(self, time: float, category: str, salience: float = 0.5):
        """카테고리 큐에 추가"""
        if not category:
            return
        self.queue.append((time, category, salience))
        self.push_count += 1

    def top_bid(self, hormone_state: Dict[str, float]) -> Tuple[float, Optional[str]]:
        """
        가장 강한 카테고리 + bid 점수 반환.
        bid = base_priority × max_salience × hormone_modifier

        Returns: (bid_score, category) — 큐 빈 경우 (0.0, None)
        """
        if not self.queue:
            return (0.0, None)

        # 가장 강한 카테고리 (큐에서)
        best_cat = None
        best_sal = 0.0
        for _, cat, sal in self.queue:
            if sal > best_sal:
                best_sal = sal
                best_cat = cat

        if best_cat is None:
            return (0.0, None)

        # 호르몬 modifier
        hormone_mod = 1.0
        if self.hormone_affinity:
            mod_sum = 0.0
            mod_n = 0
            for hname, weight in self.hormone_affinity.items():
                level = hormone_state.get(hname, 0.3)
                mod_sum += level * weight
                mod_n += 1
            if mod_n > 0:
                hormone_mod = 0.5 + mod_sum / mod_n  # range [0.5, 1.5]

        bid = self.base_priority * best_sal * hormone_mod
        return (float(bid), best_cat)

    def consume(self, category: str):
        """focus 된 카테고리 큐에서 제거 (한 번만 trigger)"""
        new_queue = deque(maxlen=self.queue.maxlen)
        consumed = False
        for entry in self.queue:
            if entry[1] == category and not consumed:
                consumed = True
                continue  # skip
            new_queue.append(entry)
        self.queue = new_queue

    def decay(self, dt: float, decay_rate: float = 0.05):
        """큐 항목 시간 따라 약화 (오래된 거)"""
        new_queue = deque(maxlen=self.queue.maxlen)
        for t, cat, sal in self.queue:
            new_sal = sal * (1.0 - decay_rate * dt)
            if new_sal > 0.1:  # 임계 미만 제거
                new_queue.append((t, cat, new_sal))
        self.queue = new_queue


class MultiStream:
    """
    EVE v32 D4 평행 사고 시스템.

    의존 (sa/hs/wm 필수, 나머지 옵션):
        sa: SpreadingActivation (필수)
        hs: HormoneSystem (필수)
        wm: WorkingMemory (필수, focus broadcast)
        dmn: DMN (옵션, INNER stream source)
        em: EpisodicMemory (옵션, RECALL stream)
        tm: Temporal (옵션, PLANNING stream — imagine_future)
        gm: GoalManagement (옵션, PLANNING stream — active_goals)
        ds: DigitalSomatic (옵션, EMOTION stream — dominant_need)
        nl: NaturalLanguage (옵션, EXTERNAL — feeling)
    """

    # 5 stream 이름
    STREAM_NAMES = ['external', 'inner', 'emotion', 'planning', 'recall']

    # Stream별 설정
    STREAM_CONFIGS = {
        # external: 외부 자극 우선순위 ★ 가장 높음
        'external': {
            'base_priority': 0.9,
            'max_size': 15,
            'hormone_affinity': {'norepinephrine': 1.5},  # NE↑ 시 더 주목
        },
        # inner: 자발 사고 (DMN)
        'inner': {
            'base_priority': 0.5,
            'max_size': 10,
            'hormone_affinity': {'dopamine': 1.0, 'cortisol': -0.8},
        },
        # emotion: 감정 흐름
        'emotion': {
            'base_priority': 0.6,
            'max_size': 8,
            'hormone_affinity': {'cortisol': 1.2, 'oxytocin': 0.8},
        },
        # planning: 미래 시뮬
        'planning': {
            'base_priority': 0.55,
            'max_size': 8,
            'hormone_affinity': {'dopamine': 0.8, 'serotonin': 0.5},
        },
        # recall: 과거 회상
        'recall': {
            'base_priority': 0.4,
            'max_size': 8,
            'hormone_affinity': {'melatonin': 1.2, 'acetylcholine': 1.0},
        },
    }

    def __init__(self,
                 spreading_activation,
                 hormone_system,
                 working_memory,
                 dmn=None,
                 episodic_memory=None,
                 temporal=None,
                 goal_management=None,
                 digital_somatic=None,
                 natural_lang=None,
                 cross_talk_strength: float = 0.15):
        self.sa = spreading_activation
        self.hs = hormone_system
        self.wm = working_memory
        self.dmn = dmn
        self.em = episodic_memory
        self.tm = temporal
        self.gm = goal_management
        self.ds = digital_somatic
        self.nl = natural_lang

        # 5 streams
        self.streams: Dict[str, Stream] = {}
        for name in self.STREAM_NAMES:
            cfg = self.STREAM_CONFIGS[name]
            self.streams[name] = Stream(
                name=name,
                base_priority=cfg['base_priority'],
                max_size=cfg['max_size'],
                hormone_affinity=cfg['hormone_affinity'],
            )

        self.cross_talk_strength = cross_talk_strength

        # focus history: [(time, stream_name, category), ...]
        self.focus_history: List[Tuple[float, str, str]] = []

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.compete_count = 0
        self.cross_talk_count = 0
        self.integration_count = 0

    # ============= 1. push_to_stream =============
    def push_to_stream(self, stream_name: str, category: str,
                      salience: float = 0.5) -> bool:
        """
        외부에서 stream에 카테고리 push.
        예: say()가 'external' stream에 push.
        """
        if stream_name not in self.streams:
            return False
        self.streams[stream_name].push(self.time, category, salience)
        return True

    # ============= 2. emotion stream tick =============
    def emotion_stream_tick(self):
        """
        호르몬 상태 → 감정 카테고리 큐 채움.

        - cortisol 높음 → '걱정', '불안'
        - dopamine 높음 → '기쁨', '성취'
        - oxytocin 높음 → '사랑', '함께'
        - DS dominant_need 있으면 그것도
        """
        cort = self.hs.hormones['cortisol'].level
        da = self.hs.hormones['dopamine'].level
        ot_h = self.hs.hormones.get('oxytocin')
        ot = ot_h.level if ot_h else 0.3

        # 강한 호르몬 → 카테고리 push (임계 0.6)
        if cort > 0.6:
            self.streams['emotion'].push(self.time, '걱정', salience=cort)
        if da > 0.6:
            self.streams['emotion'].push(self.time, '기쁨', salience=da)
        if ot > 0.6:
            self.streams['emotion'].push(self.time, '함께', salience=ot)

        # DS dominant_need
        if self.ds is not None:
            try:
                dom = self.ds.get_dominant_need()
                if dom and dom[1] >= 0.5:
                    need_cat_map = {
                        'fatigue': '피곤',
                        'warmth': '함께',
                        'mental_load': '쉬다',
                        'energy': '기쁨',
                    }
                    cat = need_cat_map.get(dom[0])
                    if cat:
                        self.streams['emotion'].push(
                            self.time, cat, salience=dom[1])
            except Exception:
                pass

    # ============= 3. planning stream tick =============
    def planning_stream_tick(self):
        """
        TM.imagine_future + GM.active_goals → planning 큐.
        """
        # 활성 목표 카테고리 push
        if self.gm is not None:
            try:
                if hasattr(self.gm, 'active_goals'):
                    goals = self.gm.active_goals(top_n=3)
                    for g in goals:
                        cat = getattr(g, 'category', None)
                        prio = getattr(g, 'priority', 0.5)
                        if cat:
                            self.streams['planning'].push(
                                self.time, cat, salience=prio)
            except Exception:
                pass

        # TM imagine_future from current focus
        if self.tm is not None and self.wm is not None:
            try:
                focus = self.wm.get_focus()
                if focus and hasattr(self.tm, 'imagine_future'):
                    fut = self.tm.imagine_future(focus, depth=2,
                                                top_n_per_step=2)
                    if isinstance(fut, dict) and 'timeline' in fut:
                        for entry in fut['timeline'][:3]:
                            cat = entry.get('category')
                            w = entry.get('weight', 0.3)
                            if cat:
                                self.streams['planning'].push(
                                    self.time, cat, salience=w * 0.7)
            except Exception:
                pass

    # ============= 4. recall stream tick =============
    def recall_stream_tick(self):
        """
        EM에서 자동 회상 — focus 관련 일화의 다른 카테고리 push.
        멜라토닌/ACh 영향 받음 (수면 시 강화).
        """
        if self.em is None or self.wm is None:
            return

        focus = self.wm.get_focus()
        if not focus:
            return

        # 멜라토닌/ACh 활성도 (회상 가속)
        mel_h = self.hs.hormones.get('melatonin')
        ach_h = self.hs.hormones.get('acetylcholine')
        mel = mel_h.level if mel_h else 0.1
        ach = ach_h.level if ach_h else 0.3
        recall_drive = (mel + ach) / 2.0

        # 임계 — 너무 자주 회상 X
        if recall_drive < 0.2:
            return

        try:
            if hasattr(self.em, 'recall'):
                episodes = self.em.recall(focus, limit=2)
                for ep in episodes:
                    # episode.categories에서 focus 외의 것 push
                    eps_cats = getattr(ep, 'categories', None)
                    if not eps_cats:
                        continue
                    for cat in eps_cats:
                        if cat == focus:
                            continue
                        self.streams['recall'].push(
                            self.time, cat,
                            salience=recall_drive * 0.5)
                        break  # 일화당 하나만 (스팸 방지)
        except Exception:
            pass

    # ============= 5. compete_for_focus =============
    def compete_for_focus(self) -> Optional[Dict[str, Any]]:
        """
        모든 stream의 top_bid 비교 → 가장 높은 bid가 focus.
        결정론 정렬 (bid ↓, stream_name ↑).

        Returns: {'stream', 'category', 'bid'} or None (모두 빈 큐)
        """
        # 호르몬 state
        h_state = {name: h.level for name, h in self.hs.hormones.items()}

        # 모든 stream의 bid
        bids = []
        for name in self.STREAM_NAMES:
            stream = self.streams[name]
            bid_score, cat = stream.top_bid(h_state)
            if cat is not None and bid_score > 0:
                bids.append((bid_score, name, cat))

        if not bids:
            return None

        # 결정론 정렬 (bid ↓, name ↑)
        bids.sort(key=lambda x: (-x[0], x[1]))

        winner = bids[0]
        bid_score, name, cat = winner

        # consume
        self.streams[name].consume(cat)
        self.streams[name].active_count += 1

        # focus history
        self.focus_history.append((self.time, name, cat))
        if len(self.focus_history) > 200:
            self.focus_history = self.focus_history[-100:]

        self.compete_count += 1

        return {
            'stream': name,
            'category': cat,
            'bid': float(bid_score),
            'all_bids': [(b, n, c) for b, n, c in bids],
        }

    # ============= 6. cross_talk =============
    def cross_talk(self, focus_category: str):
        """
        Baars GWT 핵심: focus가 모든 다른 stream에 broadcast.
        다른 stream의 큐에 약한 관련 카테고리 추가 (창의적 연결).

        focus의 SA 이웃 中 약한 가중치로 각 stream에 push.
        """
        if not focus_category:
            return

        # focus의 그래프 이웃
        neighbors = self.sa.neighbors.get(focus_category, set())
        if not neighbors:
            return

        # 결정론적 선택: top 3 weight (이웃 中)
        scored = []
        for n in neighbors:
            w = self.sa.get_weight(focus_category, n)
            scored.append((w, n))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top_neighbors = [n for _, n in scored[:3]]

        # 모든 stream에 약하게 push (focus는 제외)
        for name, stream in self.streams.items():
            for n in top_neighbors:
                stream.push(self.time, n,
                          salience=self.cross_talk_strength)

        self.cross_talk_count += 1

    # ============= 7. integrate =============
    def integrate(self, dt: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        한 통합 사이클:
        1. 각 stream tick (자체 tick — emotion/planning/recall)
        2. compete_for_focus
        3. cross_talk
        4. focus를 SA + WM에 broadcast (GNW)

        Returns: focus 정보 또는 None
        """
        # 1. stream별 자체 tick
        self.emotion_stream_tick()
        self.planning_stream_tick()
        self.recall_stream_tick()
        # external/inner는 외부에서 push (say + DMN)

        # 큐 decay
        for stream in self.streams.values():
            stream.decay(dt)

        # 2. compete
        winner = self.compete_for_focus()
        if winner is None:
            return None

        focus_cat = winner['category']

        # 3. cross-talk
        self.cross_talk(focus_cat)

        # 4. broadcast — SA + WM
        self.sa.activate(focus_cat, 0.5)
        if hasattr(self.wm, 'add'):
            self.wm.add(focus_cat, salience=0.6,
                       source=f"stream_{winner['stream']}")

        self.integration_count += 1

        return winner

    # ============= tick =============
    def tick(self, dt: float = 0.5) -> Optional[Dict[str, Any]]:
        """시간 진행 + integrate."""
        self.time += dt
        self.tick_count += 1

        # DMN 결과를 inner stream에 자동 push (있으면)
        # — DMN이 외부에서 dmn.tick() 부르면 wandering_history에 기록됨
        # — 그것을 우리가 가져옴
        if self.dmn is not None and hasattr(self.dmn, 'wandering_history'):
            try:
                # 가장 최근 wandering 가져오기 (last 1 only)
                if self.dmn.wandering_history:
                    last = self.dmn.wandering_history[-1]
                    if isinstance(last, tuple) and len(last) >= 2:
                        t, cat = last[0], last[1]
                        # 같은 시점에 안 가져오게 — time 차이로 dedupe
                        # 이미 push된 거 안 들어가게 — last push time 추적 추가 가능하지만 단순 처리
                        if (self.streams['inner'].queue
                                and self.streams['inner'].queue[-1][1] != cat):
                            self.streams['inner'].push(
                                self.time, cat, salience=0.5)
                        elif not self.streams['inner'].queue:
                            self.streams['inner'].push(
                                self.time, cat, salience=0.5)
            except Exception:
                pass

        return self.integrate(dt)

    # ============= 디버그/조회 =============
    def dump_streams(self) -> Dict[str, List[str]]:
        """각 stream의 현재 큐 (디버그)"""
        return {
            name: [cat for _, cat, _ in stream.queue]
            for name, stream in self.streams.items()
        }

    def get_focus_history(self, last_n: int = 10
                         ) -> List[Tuple[float, str, str]]:
        """최근 focus history"""
        return self.focus_history[-last_n:]

    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'compete_count': self.compete_count,
            'cross_talk_count': self.cross_talk_count,
            'integration_count': self.integration_count,
            'streams': {
                name: {
                    'queue_size': len(s.queue),
                    'push_count': s.push_count,
                    'active_count': s.active_count,
                }
                for name, s in self.streams.items()
            },
            'recent_focus': self.focus_history[-3:],
        }

    def __repr__(self):
        s = self.get_state()
        sizes = {n: info['queue_size'] for n, info in s['streams'].items()}
        return (f"MultiStream(t={s['time']:.1f}, "
                f"queues={sizes}, integrations={s['integration_count']})")
