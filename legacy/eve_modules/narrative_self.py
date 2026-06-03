"""
EVE v32 - D6 NarrativeSelf (자전적 정체성)
==========================================

EM 일화 + AG 행동 + 호르몬 패턴 → "나는 X하는 자" 자기 서사

학계:
- McAdams 1993 — Life Story Theory of Identity
- Damasio 1999 — Autobiographical Self (proto → core → autobiographical)
- Bruner 1991 — Narrative Construction of Reality
- Habermas & Bluck 2000 — Coherence (causal/temporal/thematic)
- Singer 2004 — Self-Defining Memories (5요소: vivid/affective/repetitive/
                                          tension/leading)
- Conway 2005 — Self-Memory System
- Klein 2014 — Self continuity through time

EVE 원칙 부합:
- 확률 X (빈도 + 가중치 결정론, 정렬 tie-break)
- 카테고리 = 의미 단위 (정체성도 카테고리 set)
- 호르몬 = 정체성 강도 (자주 등장 + 강한 정서 → 자기 정의)
- 다른 모듈 read-only (EM/AG/HS/SA/TM에서 읽기만)

핵심 함수 (7개):
1. extract_themes(top_n)              — EM 자주 나오는 카테고리/그룹
2. self_defining_memories(top_n)      — Singer 5요소: 강한 자기 일화
3. identity_categories()              — "나는 X" 안정적 카테고리들
4. coherence_score()                  — Habermas: 시간/주제 일관성
5. narrate(period)                    — 자기 서사 (자연어 placeholder)
6. integrate_event(category, valence) — 새 사건 → 정체성 update
7. tick(dt)                           — 자동 통합

Author: 김민석 + Claude v32 (D6, 티어 D 마지막)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter, deque


class NarrativeSelf:
    """
    EVE v32 D6 자전적 정체성 모듈.

    의존:
        em: EpisodicMemory (필수, 자전 기억의 source)
        sa: SpreadingActivation (필수, 카테고리)
        hs: HormoneSystem (필수, 정체성 강도)
        ag: Agency (옵션, 자기 행동)
        tm: Temporal (옵션, 시간 흐름)
        world: WorldModel (옵션, 통합 view)
        wm: WorkingMemory (옵션, focus)
    """

    # 자기-defining 한 일화의 호르몬 임계 (Singer affective)
    AFFECTIVE_THRESHOLD = 0.6

    # 정체성 카테고리 후보 그룹 (어떤 그룹이 정체성에 들어가는지)
    IDENTITY_GROUPS = {'social', 'reward', 'curiosity', 'rest'}

    def __init__(self,
                 episodic_memory,
                 spreading_activation,
                 hormone_system,
                 agency=None,
                 temporal=None,
                 world_model=None,
                 working_memory=None,
                 stability_threshold: float = 0.3,
                 history_size: int = 50):
        self.em = episodic_memory
        self.sa = spreading_activation
        self.hs = hormone_system
        self.ag = agency
        self.tm = temporal
        self.world = world_model
        self.wm = working_memory

        self.stability_threshold = stability_threshold

        # 정체성 카테고리 안정성 추적 (cat -> stability score)
        self.identity_stability: Dict[str, float] = defaultdict(float)
        # 보호 카테고리 (외부 시드, 절대 사라지지 않음)
        self.protected_categories: Set[str] = set()

        # 자기-defining 일화 (Singer) — episode 또는 카테고리 list
        self.self_defining_memo: List[Dict[str, Any]] = []

        # 통합된 사건들 (deque)
        self.integration_log: deque = deque(maxlen=history_size)

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.extract_count = 0
        self.narrate_count = 0
        self.integrate_count = 0
        self.coherence_count = 0

    # ============= 1. extract_themes =============
    def extract_themes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        EM 일화에서 자주 나오는 카테고리 (McAdams themes).
        결정론 정렬 (빈도 ↓, 카테고리 알파벳 ↑).

        Returns: [{'category', 'count', 'weight'}, ...]
        """
        if self.em is None:
            return []

        # category_index 활용
        cat_index = getattr(self.em, 'category_index', None)
        if not cat_index:
            return []

        # 빈도 기반
        scored: List[Tuple[str, int, float]] = []
        for cat, eps in cat_index.items():
            count = len(eps)
            if count == 0:
                continue
            # 가중치 = log scale 빈도 + 그룹 보너스
            weight = float(np.log1p(count) / 5.0)
            try:
                from spreading_activation import SpreadingActivation
                group = SpreadingActivation.CATEGORY_GROUPS.get(cat)
                if group in self.IDENTITY_GROUPS:
                    weight *= 1.3
            except Exception:
                pass
            scored.append((cat, count, weight))

        # 정렬 (count ↓, alphabet ↑)
        scored.sort(key=lambda x: (-x[1], x[0]))

        self.extract_count += 1

        return [
            {'category': cat, 'count': c, 'weight': float(w)}
            for cat, c, w in scored[:top_n]
        ]

    # ============= 2. self_defining_memories =============
    def self_defining_memories(self, top_n: int = 5
                              ) -> List[Dict[str, Any]]:
        """
        Singer 2004 자기-defining 5요소:
        1. Vivid (강한 활성)
        2. Affective (높은 호르몬 변동)
        3. Repetitive (자주 회상)
        4. Tension (해결되지 않은)
        5. Leading (핵심 갈등 향한)

        EVE 단순화: vivid + affective 위주.
        Returns: [{'episode_summary', 'score', ...}, ...]
        """
        if self.em is None:
            return []

        episodes = getattr(self.em, 'episodes', None)
        if not episodes:
            return []

        scored = []
        for ep in episodes:
            # vivid: 활성 강도 (있으면)
            vivid = 0.5
            if hasattr(ep, 'vividness'):
                vivid = float(ep.vividness)

            # affective: 호르몬 변동 (있으면)
            affective = 0.5
            if hasattr(ep, 'hormone_state'):
                hs_state = ep.hormone_state or {}
                if hs_state:
                    # 호르몬 분산이 큰 일화 = 강한 정서
                    levels = list(hs_state.values())
                    if levels:
                        max_lvl = max(levels)
                        affective = max_lvl

            # 카테고리 size (많은 카테고리 = 풍부한 일화)
            cat_count = 0
            if hasattr(ep, 'categories'):
                cat_count = len(ep.categories or [])
            richness = min(1.0, cat_count / 5.0)

            score = (vivid * 0.3 + affective * 0.5 + richness * 0.2)

            if score >= self.AFFECTIVE_THRESHOLD:
                summary = (getattr(ep, 'summary', '') or
                          str(getattr(ep, 'categories', '')))
                scored.append({
                    'time': getattr(ep, 'time', 0.0),
                    'summary': str(summary)[:100],
                    'categories': list(getattr(ep, 'categories', [])),
                    'vivid': float(vivid),
                    'affective': float(affective),
                    'richness': float(richness),
                    'score': float(score),
                })

        # 결정론 정렬 (score ↓, time ↓)
        scored.sort(key=lambda x: (-x['score'], -x['time']))

        # 캐시
        self.self_defining_memo = scored[:top_n]
        return scored[:top_n]

    # ============= 3. identity_categories =============
    def identity_categories(self,
                           min_stability: Optional[float] = None
                           ) -> List[Dict[str, Any]]:
        """
        "나는 X하는 자" — 안정된 자기 카테고리들.

        안정성 = (테마 빈도 + 자기 행동 + 시간 지속)
        - EM 빈도: 자주 등장
        - AG: 자기가 자주 do한 카테고리
        - 시간: 오래 안정 (stability 누적)

        Returns: [{'category', 'stability', 'role', 'evidence'}, ...]
        """
        threshold = min_stability or self.stability_threshold

        # 1. 테마에서 시작
        themes = self.extract_themes(top_n=20)
        theme_map = {t['category']: t['weight'] for t in themes}

        # 2. AG 자기 행동
        action_map: Dict[str, int] = defaultdict(int)
        if self.ag is not None and hasattr(self.ag, 'actions'):
            try:
                for record in list(self.ag.actions)[-100:]:
                    if isinstance(record, dict):
                        cat = record.get('action_category')
                        choice = record.get('choice')
                        if cat and choice in ('do', 'intentional'):
                            action_map[cat] += 1
            except Exception:
                pass

        # 3. 통합 stability 계산
        stability_now: Dict[str, float] = {}

        all_cats = set(theme_map.keys()) | set(action_map.keys())
        for cat in all_cats:
            theme_w = theme_map.get(cat, 0.0)
            action_n = action_map.get(cat, 0)
            action_w = float(np.log1p(action_n) / 4.0)

            # 누적 stability — max 보존: 새 신호와 이전 中 큰 거 + 약한 누적 보너스
            # 자주 등장 → plateau 깨고 더 올라감
            prev = self.identity_stability.get(cat, 0.0)
            current = (theme_w * 0.6 + action_w * 0.4)
            # 새 신호가 강하면 점프 + 약한 누적 (반복 시 0.05씩 더 누적)
            base = max(prev, current)
            cumulative_bonus = min(0.2, prev * 0.1)  # max +0.2
            stab = float(np.clip(base + cumulative_bonus, 0.0, 1.0))
            stability_now[cat] = stab

        # 누적 업데이트
        for cat, stab in stability_now.items():
            self.identity_stability[cat] = stab

        # 임계 통과 카테고리만
        identity = []
        for cat, stab in stability_now.items():
            if stab < threshold:
                continue

            # role: theme/action/both
            has_theme = theme_map.get(cat, 0.0) > 0
            has_action = action_map.get(cat, 0) > 0
            if has_theme and has_action:
                role = 'both'
            elif has_theme:
                role = 'theme'
            else:
                role = 'action'

            evidence = {
                'theme_weight': theme_map.get(cat, 0.0),
                'action_count': action_map.get(cat, 0),
            }

            identity.append({
                'category': cat,
                'stability': float(stab),
                'role': role,
                'evidence': evidence,
            })

        # 정렬 (stability ↓, alpha ↑)
        identity.sort(key=lambda x: (-x['stability'], x['category']))

        return identity

    # ============= 4. coherence_score =============
    def coherence_score(self) -> Dict[str, Any]:
        """
        Habermas & Bluck 2000 일관성:
        - temporal: 시간 흐름 안정성
        - thematic: 주제 반복
        - causal: 인과 사슬 (CG 활용시)

        Returns: {temporal, thematic, causal, overall}
        """
        # thematic — 안정된 카테고리 수
        identity = self.identity_categories()
        thematic = float(min(1.0, len(identity) / 8.0))

        # temporal — EM 시간 분산
        temporal = 0.0
        if self.em and hasattr(self.em, 'episodes'):
            episodes = list(self.em.episodes)
            if len(episodes) >= 2:
                times = sorted([getattr(ep, 'time', 0) for ep in episodes])
                # 시간 spread (상수 normalize)
                if times:
                    spread = times[-1] - times[0]
                    # spread 클수록 좋음 (오래 산 거)
                    temporal = float(min(1.0, spread / 100.0))

        # causal — TM precedes 통계 (있으면)
        causal = 0.5  # default
        # 단순화 — 미래 확장

        overall = (temporal * 0.3 + thematic * 0.5 + causal * 0.2)

        self.coherence_count += 1

        return {
            'temporal': temporal,
            'thematic': thematic,
            'causal': causal,
            'overall': float(overall),
        }

    # ============= 5. narrate =============
    def narrate(self,
               period: Optional[Tuple[float, float]] = None
               ) -> Dict[str, Any]:
        """
        자기 서사 생성 (Bruner narrative).

        구조: 정체성 카테고리 + 자기-defining 일화 + arc

        Returns:
            {
                'identity_summary': "나는 X하는 자",
                'themes': [...],
                'self_defining': [...],
                'period': (start, end),
                'coherence': {...},
                'narrative_text': str (자연어 시드)
            }
        """
        themes = self.extract_themes(top_n=5)
        self_def = self.self_defining_memories(top_n=3)
        identity = self.identity_categories()
        coherence = self.coherence_score()

        # 정체성 요약 — top 3 카테고리
        top_id = identity[:3]
        if top_id:
            id_cats = [i['category'] for i in top_id]
            id_summary = "나는 " + ", ".join(id_cats) + "와 함께 사는 존재"
        else:
            id_summary = "나는 형성 중인 존재"

        # 자연어 시드 (NL이 활용 가능)
        if themes:
            theme_str = ", ".join([t['category'] for t in themes[:3]])
            narrative_text = (
                f"{id_summary}. 자주 {theme_str}이/가 떠오른다."
            )
        else:
            narrative_text = id_summary

        if self_def:
            sd_summary = self_def[0]['summary'][:50]
            narrative_text += f" 특히 '{sd_summary}'은 잊을 수 없다."

        # period
        if period is None and self.em:
            episodes = list(getattr(self.em, 'episodes', []))
            if episodes:
                times = [getattr(ep, 'time', 0) for ep in episodes]
                period = (min(times), max(times)) if times else (0.0, 0.0)
            else:
                period = (0.0, self.time)

        self.narrate_count += 1

        return {
            'identity_summary': id_summary,
            'themes': themes,
            'self_defining': self_def,
            'identity': top_id,
            'period': period,
            'coherence': coherence,
            'narrative_text': narrative_text,
        }

    # ============= 6. integrate_event =============
    def integrate_event(self,
                       category: str,
                       valence: float = 0.0,
                       importance: float = 0.5) -> Dict[str, Any]:
        """
        새 사건 → 정체성 update.

        Args:
            category: 사건의 핵심 카테고리
            valence: -1 (부정) ~ +1 (긍정)
            importance: 0 ~ 1

        Update:
            - identity_stability에 추가 (가중치)
            - integration_log 기록

        Returns: {category, prior_stability, new_stability, delta}
        """
        prior = self.identity_stability.get(category, 0.0)

        # importance와 valence(절대값) 조합
        # 극단적 valence → 더 강한 정체성 영향
        intensity = importance * (0.5 + abs(valence) * 0.5)
        delta = intensity * 0.1  # 작은 step 누적

        new_stab = float(np.clip(prior + delta, 0.0, 1.0))
        self.identity_stability[category] = new_stab

        record = {
            'time': self.time,
            'category': category,
            'valence': float(valence),
            'importance': float(importance),
            'delta': float(delta),
        }
        self.integration_log.append(record)

        self.integrate_count += 1

        return {
            'category': category,
            'prior_stability': float(prior),
            'new_stability': new_stab,
            'delta': float(delta),
        }

    # ============= 7. tick =============
    def tick(self, dt: float):
        """시간 진행 + stability 비례 decay (강한 정체성 보호)"""
        self.time += dt
        self.tick_count += 1

        # Stability 비례 decay: 강한 건 천천히, 약한 건 빠르게 (인간 같은 자연 망각)
        # 매 60 tick (30초)마다 — 기존 20 tick보다 3배 늦게
        if self.tick_count % 60 == 0:
            for cat in list(self.identity_stability.keys()):
                stab = self.identity_stability[cat]
                # 보호 카테고리 — floor 0.5 (절대 그 밑으로 안 떨어짐)
                if cat in self.protected_categories:
                    if stab < 0.5:
                        self.identity_stability[cat] = 0.5
                    else:
                        # 약한 decay (잘 안 변함)
                        self.identity_stability[cat] = max(0.5, stab * 0.9999)
                    continue
                # Stability 비례 decay rate (덜 빨리)
                if stab >= 0.7:
                    decay = 0.9995  # 강한 정체성 거의 영구 보존
                elif stab >= 0.3:
                    decay = 0.998   # 중간 — 매우 천천히 약화
                else:
                    decay = 0.995   # 약한 — 자연 망각
                new_stab = stab * decay
                if new_stab < 0.05:
                    del self.identity_stability[cat]
                else:
                    self.identity_stability[cat] = new_stab

    def protect(self, *categories: str):
        """시드 정체성 보호 (절대 사라지지 않음)"""
        for cat in categories:
            self.protected_categories.add(cat)
            # 강하게 시드
            self.identity_stability[cat] = max(self.identity_stability.get(cat, 0.0), 0.9)

    # ============= 상태 =============
    def get_state(self) -> Dict[str, Any]:
        # 가장 안정된 정체성 카테고리 top 3
        top_identity = sorted(
            self.identity_stability.items(),
            key=lambda x: (-x[1], x[0])
        )[:3]
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'identity_size': len(self.identity_stability),
            'top_identity': [(c, float(s)) for c, s in top_identity],
            'extract_count': self.extract_count,
            'narrate_count': self.narrate_count,
            'integrate_count': self.integrate_count,
            'coherence_count': self.coherence_count,
            'self_defining_cached': len(self.self_defining_memo),
        }

    def __repr__(self):
        s = self.get_state()
        top = s['top_identity'][0][0] if s['top_identity'] else 'forming'
        return (f"NarrativeSelf(t={s['time']:.1f}, "
                f"identity_size={s['identity_size']}, "
                f"top='{top}', narrates={s['narrate_count']})")
