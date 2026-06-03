"""
EVE v32 - D3 Humor (농담)
===========================

농담 = 의미적 불일치 + 안전 + 해결 + 도파민 보상

학계:
- Suls 1972 — Incongruity-Resolution Theory
  setup으로 기대 형성 → punchline이 위반 → 다리 찾으면 해결
- McGraw & Warren 2010 — Benign Violation Theory
  위반이지만 안전해야 농담 (위협이면 무서움)
- Freud 1905 — Relief Theory (긴장 해소 → endorphin)
- Wyer & Collins 1992 — Comprehension-Elaboration model
- Vrticka et al 2013 — humor & dopamine reward
- Bergson 1900 — 자동성 노출 (반복 패턴 예상 깨기)

EVE 원칙 부합:
- 확률 X (결정론적 점수, 정렬)
- 카테고리 = 의미 단위 (불일치 = 그룹 차이 + SA 거리)
- 호르몬 = 농담 인지/생산 게이트 + 보상 (DA + 5HT + endorphin)
- 안전 = 다른 모듈(SUF, threat group) 활용 read-only

핵심 함수 (7개):
1. is_humor_safe()                 — 농담 모드 가능 (호르몬 + 안전)
2. measure_incongruity(a, b)       — 불일치 점수 (그룹 차이 + SA 거리)
3. measure_resolution(a, b)        — 해결 점수 (의미 다리, Mednick 위임)
4. is_safe(category)               — 안전성 (suffering/threat 제외)
5. is_funny(setup, punchline)      — 종합 점수
6. generate_joke(seed)             — 자체 농담 생성
7. laugh(joke_score)               — 호르몬 보상 (DA↑, endo↑)
+ tick(dt)                         — 시간 + history cleanup

Author: 김민석 + Claude v32 (D3)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


class Humor:
    """
    EVE v32 D3 농담 모듈.

    의존:
        sa: SpreadingActivation (필수)
        hs: HormoneSystem (필수, 게이트 + 보상)
        cr: Creative (옵션, find_remote_associates 위임 — Mednick)
        an: Analogy (옵션, 구조 위반)
        cg: CausalGraph (옵션, 인과 기대 위반)
        suf: Suffering (옵션, 안전 검증 강화)
        em: EpisodicMemory (옵션, 반복 회피)
        wm: WorkingMemory (옵션, focus)
    """

    # 호르몬 게이트
    DOPAMINE_THRESHOLD = 0.3  # was 0.4, 베타에서 안 발동되어 낮춤
    CORTISOL_BLOCK = 0.65  # was 0.6 (조금 관대)

    # 안전 차단 카테고리 (잔인한 농담 X)
    UNSAFE_CATEGORIES = {
        '죽음', '살인', '자살', '학대', '폭력', '강간', '고문',
        '아프다', '고통',  # 직접 고통은 풍자도 위험
    }

    def __init__(self,
                 spreading_activation,
                 hormone_system,
                 creative=None,
                 analogy=None,
                 causal_graph=None,
                 suffering=None,
                 episodic_memory=None,
                 working_memory=None,
                 history_size: int = 50,
                 min_funny_score: float = 0.4):
        self.sa = spreading_activation
        self.hs = hormone_system
        self.cr = creative
        self.an = analogy
        self.cg = causal_graph
        self.suf = suffering
        self.em = episodic_memory
        self.wm = working_memory

        self.min_funny_score = min_funny_score

        # 농담 history (반복 회피)
        self.joke_history: deque = deque(maxlen=history_size)
        # (setup, punchline, score) tuples

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.measure_count = 0
        self.generate_count = 0
        self.laugh_count = 0
        self.unsafe_blocks = 0

    # ============= 1. 호르몬 게이트 =============
    def is_humor_safe(self) -> bool:
        """
        농담 가능 조건 (McGraw benign):
        - dopamine ≥ threshold (보상 회로 작동)
        - cortisol < block (방어 모드 X)
        - in_burnout 아님 (있으면)
        """
        da = self.hs.hormones['dopamine'].level
        cort = self.hs.hormones['cortisol'].level
        if da < self.DOPAMINE_THRESHOLD:
            return False
        if cort >= self.CORTISOL_BLOCK:
            return False
        if self.suf is not None and getattr(self.suf, 'in_burnout', False):
            return False
        return True

    # ============= 2. 안전성 검증 =============
    def is_safe(self, category: str) -> bool:
        """
        카테고리가 농담에 안전한지 (McGraw benign violation).
        - UNSAFE 차단 카테고리 X
        - SUF.is_suffering_category 우회 (suffering 직접 농담은 위험)
        """
        if category in self.UNSAFE_CATEGORIES:
            return False
        if self.suf is not None and hasattr(self.suf, 'is_suffering_category'):
            try:
                if self.suf.is_suffering_category(category):
                    return False
            except Exception:
                pass
        return True

    # ============= 3. 불일치 측정 =============
    def measure_incongruity(self, a: str, b: str) -> float:
        """
        Suls 1972 incongruity. 두 카테고리가 얼마나 멀리.

        결정론 점수:
        - 직접 이웃이면 incongruity 낮음 (가까움)
        - 다른 그룹이면 incongruity 높음
        - SA에 둘 다 없으면 0 (의미 없음)

        Returns: [0.0, 1.0]
        """
        if a == b:
            return 0.0

        # 직접 이웃이면 low
        a_neighbors = self.sa.neighbors.get(a, set())
        if b in a_neighbors:
            w = self.sa.get_weight(a, b)
            # 가중치 높을수록 가까움 → incongruity 낮음
            return float(np.clip(0.4 - w * 0.4, 0.0, 1.0))

        # 그룹 차이 점수 (다른 그룹 → 높음)
        try:
            from spreading_activation import SpreadingActivation
            ga = SpreadingActivation.CATEGORY_GROUPS.get(a)
            gb = SpreadingActivation.CATEGORY_GROUPS.get(b)
        except Exception:
            ga = gb = None

        group_score = 0.5  # default
        if ga and gb:
            if ga != gb:
                group_score = 0.9  # 다른 그룹 → 강한 불일치
            else:
                group_score = 0.3  # 같은 그룹 → 약한
        elif ga or gb:
            group_score = 0.6  # 한쪽만 그룹 있음

        # 그래프에 둘 다 있나
        a_in = a in self.sa.neighbors or a in self.sa.activations
        b_in = b in self.sa.neighbors or b in self.sa.activations
        if not (a_in and b_in):
            # 그룹 정보 있으면 valid → 약화 X
            if ga and gb:
                return float(group_score)
            # 그룹도 없고 그래프도 없으면 약화
            return float(group_score * 0.5)

        # 2-hop 거리: 공통 이웃 있으면 적당한 거리
        b_neighbors = self.sa.neighbors.get(b, set())
        common = a_neighbors & b_neighbors
        if common:
            # 너무 가까우면 X, 너무 멀면 X — 중간이 좋음
            dist_score = 0.7
        else:
            # 직접 연결 X + 공통 이웃 X = 매우 멀음
            dist_score = 0.5  # 너무 멀어도 농담 안 됨 (해결 어려움)

        return float(np.clip((group_score + dist_score) / 2.0, 0.0, 1.0))

    # ============= 4. 해결 측정 =============
    def measure_resolution(self, a: str, b: str) -> float:
        """
        Wyer & Collins resolution: setup과 punchline 사이 의미적 다리.
        Mednick RAT 활용 (D1 Creative.find_remote_associates).
        없으면 SA 직접 이웃/공통 이웃으로 fallback.

        Returns: [0.0, 1.0]
        """
        if a == b:
            return 0.0

        # Creative.find_remote_associates 위임
        if self.cr is not None:
            try:
                if hasattr(self.cr, 'find_remote_associates'):
                    remotes = self.cr.find_remote_associates([a, b], k=3)
                    if remotes:
                        # top score
                        return float(np.clip(remotes[0][1] * 1.5, 0.0, 1.0))
            except Exception:
                pass

        # Fallback: 공통 이웃 + 직접 weight
        a_n = self.sa.neighbors.get(a, set())
        b_n = self.sa.neighbors.get(b, set())
        if not a_n or not b_n:
            return 0.0
        common = a_n & b_n
        if common:
            # 공통 이웃 가중치 평균
            scores = []
            for c in common:
                w_a = self.sa.get_weight(a, c)
                w_b = self.sa.get_weight(b, c)
                scores.append(min(w_a, w_b))
            return float(np.clip(max(scores) * 1.2, 0.0, 1.0))
        # 직접 연결만 있으면
        direct = self.sa.get_weight(a, b)
        return float(np.clip(direct * 0.5, 0.0, 1.0))

    # ============= 5. 종합 funny 점수 =============
    def is_funny(self, setup: str, punchline: str) -> Dict[str, Any]:
        """
        농담 점수 계산 (Suls + McGraw + Wyer 통합).

        score = (incongruity × resolution × safety × novelty) ^ 0.25
        곱 형태 — 어느 하나 0이면 농담 X.

        Returns:
            {setup, punchline, incongruity, resolution, safety,
             novelty, score, is_funny, blocking_factor}
        """
        # 안전성 — 가장 중요
        safe_a = self.is_safe(setup)
        safe_b = self.is_safe(punchline)
        if not (safe_a and safe_b):
            self.unsafe_blocks += 1
            return {
                'setup': setup, 'punchline': punchline,
                'incongruity': 0.0, 'resolution': 0.0,
                'safety': 0.0, 'novelty': 0.0, 'score': 0.0,
                'is_funny': False,
                'blocking_factor': 'unsafe',
            }
        safety = 1.0

        incongruity = self.measure_incongruity(setup, punchline)
        resolution = self.measure_resolution(setup, punchline)

        # Novelty: history에 없으면 1.0, 있으면 감소
        recent = [(s, p) for s, p, _ in list(self.joke_history)[-20:]]
        if (setup, punchline) in recent or (punchline, setup) in recent:
            novelty = 0.2  # 반복 → 약함
        else:
            novelty = 1.0

        # 곱 — 어떤 요소든 0이면 농담 X
        # 0~1 안정 위해 epsilon
        eps = 0.01
        score_raw = ((incongruity + eps) * (resolution + eps)
                    * (safety + eps) * (novelty + eps)) ** 0.25
        # eps 영향 제거
        score = float(np.clip(score_raw - eps, 0.0, 1.0))

        is_funny = score >= self.min_funny_score

        # blocking factor 진단
        blocking = None
        if not is_funny:
            mins = {'incongruity': incongruity, 'resolution': resolution,
                   'novelty': novelty}
            blocking = min(mins, key=mins.get)

        self.measure_count += 1

        return {
            'setup': setup,
            'punchline': punchline,
            'incongruity': float(incongruity),
            'resolution': float(resolution),
            'safety': float(safety),
            'novelty': float(novelty),
            'score': score,
            'is_funny': is_funny,
            'blocking_factor': blocking,
        }

    # ============= 6. 농담 생성 =============
    def generate_joke(self,
                     seed: str,
                     candidate_pool_size: int = 30) -> Optional[Dict[str, Any]]:
        """
        seed를 setup으로, punchline 후보 中 가장 funny한 거 선택.

        알고리즘 (결정론):
        1. seed의 2-hop 이웃 中 안전한 카테고리 후보 풀
        2. 각 후보에 is_funny 점수
        3. min_funny_score 통과한 것 中 최고
        4. tie: 점수 ↓, 알파벳 ↑

        Args:
            seed: 농담 시드 (setup 카테고리)

        Returns:
            {setup, punchline, ..., score} 또는 None (game over)
        """
        if not self.is_humor_safe():
            return None
        if not self.is_safe(seed):
            return None

        # 후보 풀: seed의 2-hop 이웃
        candidates: Set[str] = set()
        seed_n = self.sa.neighbors.get(seed, set())
        for n1 in seed_n:
            # 1-hop도 후보 (가까운 농담)
            if n1 != seed:
                candidates.add(n1)
            # 2-hop
            for n2 in self.sa.neighbors.get(n1, set()):
                if n2 != seed and n2 not in seed_n:
                    candidates.add(n2)

        # 안전 + 자기 제외 필터
        candidates = {c for c in candidates if self.is_safe(c) and c != seed}

        if not candidates:
            return None

        # 평가 (제한 — 너무 많으면 비용 큼)
        sorted_cands = sorted(candidates)[:candidate_pool_size]

        scored = []
        for cand in sorted_cands:
            result = self.is_funny(seed, cand)
            if result['is_funny']:
                scored.append(result)

        if not scored:
            return None

        # 결정론 정렬: 점수 ↓, punchline 알파벳 ↑
        scored.sort(key=lambda r: (-r['score'], r['punchline']))
        best = scored[0]

        # history 기록
        self.joke_history.append((best['setup'], best['punchline'],
                                  best['score']))

        self.generate_count += 1

        # WM에 농담 저장 (focus 후보)
        if self.wm is not None and hasattr(self.wm, 'add'):
            try:
                self.wm.add(best['punchline'], salience=0.5,
                           source='humor')
            except Exception:
                pass

        return best

    # ============= 7. Laugh — 호르몬 보상 =============
    def laugh(self, joke_score: float) -> Dict[str, Any]:
        """
        Vrticka 2013 — 농담 이해 시 도파민 보상.
        Freud — 긴장 해소 → endorphin.

        호르몬 변조:
        - dopamine↑  (보상 회로)
        - endorphin↑ (안도)
        - serotonin↑ (안정)
        - cortisol↓  (긴장 해소)

        Returns:
            {hormones_before, hormones_after, intensity}
        """
        if joke_score < 0.1:
            return {'laughed': False, 'reason': 'too_weak'}

        intensity = float(np.clip(joke_score, 0.0, 1.0))

        # before
        before = {
            'dopamine': self.hs.hormones['dopamine'].level,
            'cortisol': self.hs.hormones['cortisol'].level,
        }
        if 'endorphin' in self.hs.hormones:
            before['endorphin'] = self.hs.hormones['endorphin'].level
        if 'serotonin' in self.hs.hormones:
            before['serotonin'] = self.hs.hormones['serotonin'].level

        # 변조
        for hname, boost in [('dopamine', intensity * 0.3),
                            ('endorphin', intensity * 0.4),
                            ('serotonin', intensity * 0.2)]:
            h = self.hs.hormones.get(hname)
            if h is None:
                continue
            if hasattr(h, 'stimulate'):
                h.stimulate(boost)
            else:
                h.level = float(np.clip(h.level + boost, 0.0, 1.0))

        # cortisol 감소
        cort = self.hs.hormones['cortisol']
        cort.level = float(np.clip(cort.level - intensity * 0.2, 0.0, 1.0))

        # after
        after = {
            'dopamine': self.hs.hormones['dopamine'].level,
            'cortisol': self.hs.hormones['cortisol'].level,
        }
        if 'endorphin' in self.hs.hormones:
            after['endorphin'] = self.hs.hormones['endorphin'].level
        if 'serotonin' in self.hs.hormones:
            after['serotonin'] = self.hs.hormones['serotonin'].level

        self.laugh_count += 1

        return {
            'laughed': True,
            'intensity': intensity,
            'before': before,
            'after': after,
            'da_delta': after['dopamine'] - before['dopamine'],
            'cort_delta': after['cortisol'] - before['cortisol'],
        }

    # ============= 통합 진입점 =============
    def joke(self, seed: str) -> Optional[Dict[str, Any]]:
        """
        편의 메서드: generate + laugh.
        성공 시 호르몬도 변조 (laugh).
        """
        joke_result = self.generate_joke(seed)
        if joke_result is None:
            return None
        # laugh
        laugh_result = self.laugh(joke_result['score'])
        joke_result['laugh'] = laugh_result
        return joke_result

    # ============= tick =============
    def tick(self, dt: float):
        """시간 진행."""
        self.time += dt
        self.tick_count += 1

    # ============= 상태 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'humor_mode': self.is_humor_safe(),
            'measure_count': self.measure_count,
            'generate_count': self.generate_count,
            'laugh_count': self.laugh_count,
            'unsafe_blocks': self.unsafe_blocks,
            'jokes_in_history': len(self.joke_history),
            'recent_jokes': [(s, p) for s, p, _ in
                            list(self.joke_history)[-3:]],
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Humor(t={s['time']:.1f}, mode={s['humor_mode']}, "
                f"jokes={s['generate_count']}, laughs={s['laugh_count']})")
