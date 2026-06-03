"""
EVE v32 - D1 Creative
======================

창의성 = 먼 카테고리들 사이 의미적 다리 + 새로운 조합 + 인큐베이션 통찰

학계:
- Mednick 1962 — Remote Associates Test (RAT)
  먼 카테고리 사이 "공통 매개"가 통찰의 본질
- Boden 1990 — Combinational/Exploratory creativity
  EVE = 1+2 (먼 조합 + 영역 내 탐색)
- Wallas 1926 — 4단계: preparation → incubation → illumination → verification
- Csíkszentmihályi 1996 — Flow + 도파민
- Beaty et al. 2018 — DMN + executive 협력 (Big-C 창의성)
- Sawyer 2012 — Group genius

EVE 원칙 부합:
- 확률 X (결정론적 점수 + 알파벳 정렬)
- 호르몬 게이트: 도파민↑ + cortisol 낮음 → 창의 모드 (Csíkszentmihályi)
- 카테고리 = 의미 단위 (먼 카테고리 매개)
- DMN 협력 (incubation = mind_wandering 활용)
- 다른 모듈 read-only (수정 X)

핵심 함수 (7개):
1. is_creative_mode()              — 호르몬 게이트
2. find_remote_associates(seeds)   — Mednick RAT
3. blend_concepts(a, b)            — Boden 조합
4. analogy_blend(target, source)   — AN 활용 구조 차용
5. incubate(seed)                  — Wallas 시드
6. harvest_incubation()            — Wallas illumination 수확
7. verify_idea(idea)               — Agency/SD 통합 평가
+ tick(dt)                         — 자동 수확

Author: 김민석 + Claude v32 (D1)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class Creative:
    """
    EVE v32 D1 창의성 모듈.

    의존 (모두 옵션 — None 안전):
        sa: SpreadingActivation (필수)
        hs: HormoneSystem (필수)
        an: Analogy (옵션, analogy_blend 풍부화)
        cg: CausalGraph (옵션, 미래 인과 시뮬)
        em: EpisodicMemory (옵션, 일화 영감)
        dmn: DefaultModeNetwork (옵션, incubation)
        wm: WorkingMemory (옵션, seed 활성)
        ag: Agency (옵션, verify)
        sd: SelfDoubt (옵션, verify)
    """

    def __init__(self,
                 spreading_activation,
                 hormone_system,
                 analogy=None,
                 causal_graph=None,
                 episodic_memory=None,
                 dmn=None,
                 working_memory=None,
                 agency=None,
                 self_doubt=None,
                 dopamine_threshold: float = 0.5,
                 cortisol_block: float = 0.7,
                 incubation_max: int = 5,
                 incubation_window: float = 30.0):
        self.sa = spreading_activation
        self.hs = hormone_system
        self.an = analogy
        self.cg = causal_graph
        self.em = episodic_memory
        self.dmn = dmn
        self.wm = working_memory
        self.ag = agency
        self.sd = self_doubt

        self.dopamine_threshold = dopamine_threshold
        self.cortisol_block = cortisol_block
        self.incubation_max = incubation_max
        self.incubation_window = incubation_window

        # 인큐베이션 상태: seed -> {time, baseline, harvested}
        self.incubating: Dict[str, Dict[str, Any]] = {}
        # 통찰 이력 (50개 cap)
        self.illuminations: List[Dict[str, Any]] = []

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.remote_count = 0
        self.blend_count = 0
        self.incubate_count = 0
        self.illuminate_count = 0
        self.verify_count = 0

    # ============= 1. 호르몬 게이트 =============
    def is_creative_mode(self) -> bool:
        """
        창의 모드 활성 조건:
        - dopamine ≥ threshold (Csíkszentmihályi flow)
        - cortisol < block (위협 시 수렴 사고만)
        """
        da = self.hs.hormones['dopamine'].level
        cort = self.hs.hormones['cortisol'].level
        return da >= self.dopamine_threshold and cort < self.cortisol_block

    # ============= 2. Remote Associates (Mednick) =============
    def find_remote_associates(self,
                               seeds: List[str],
                               k: int = 3) -> List[Tuple[str, float]]:
        """
        먼 seed 카테고리들을 잇는 공통 매개 카테고리 찾기.
        Mednick 1962 RAT.

        예: seeds=['민석', '비'] → '슬픔' (둘 다 슬픔과 연결)

        알고리즘 (결정론):
        1. 각 seed의 BFS depth=2 이웃 + path strength
        2. 모든 seed의 공통 이웃 = 매개 후보
        3. 점수 = (avg_path + min_path) / 2
        4. 정렬 (점수↓, 알파벳↑)

        Returns: [(category, score), ...]
        """
        if not seeds or len(seeds) < 2:
            return []
        seeds_set = set(seeds)

        # 각 seed의 BFS 이웃 + path 강도
        seed_paths: List[Dict[str, float]] = []
        for s in seeds:
            if s not in self.sa.neighbors:
                continue
            paths: Dict[str, float] = {}
            # depth 1
            for n1 in self.sa.neighbors.get(s, set()):
                if n1 in seeds_set:
                    continue
                w1 = self.sa.get_weight(s, n1)
                if w1 > paths.get(n1, 0):
                    paths[n1] = w1
                # depth 2
                for n2 in self.sa.neighbors.get(n1, set()):
                    if n2 == s or n2 in seeds_set or n2 == n1:
                        continue
                    w2 = self.sa.get_weight(n1, n2)
                    score = w1 * w2
                    if score > paths.get(n2, 0):
                        paths[n2] = score
            seed_paths.append(paths)

        if len(seed_paths) < 2:
            return []

        # 공통 매개
        common = set(seed_paths[0].keys())
        for sp in seed_paths[1:]:
            common &= set(sp.keys())

        # 점수
        scored = []
        for cand in common:
            paths = [sp[cand] for sp in seed_paths]
            min_path = min(paths)
            avg_path = sum(paths) / len(paths)
            score = (avg_path + min_path) / 2.0
            scored.append((cand, float(score)))

        # 결정론 정렬
        scored.sort(key=lambda x: (-x[1], x[0]))

        self.remote_count += 1
        return scored[:k]

    # ============= 3. Concept Blend (Boden) =============
    def blend_concepts(self, a: str, b: str) -> Dict[str, Any]:
        """
        두 카테고리의 의미적 blend.
        - shared_neighbors: 공통 이웃 (조합 핵심)
        - novel_a/b: 한쪽에만 있는 이웃 (새 가능성)
        - coherence: shared / total
        - novelty: only / total
        - score: 둘의 평균 (Boden 균형)

        Returns: dict
        """
        nb_a = self.sa.neighbors.get(a, set())
        nb_b = self.sa.neighbors.get(b, set())

        shared = nb_a & nb_b
        only_a = (nb_a - nb_b) - {b}
        only_b = (nb_b - nb_a) - {a}

        total = len(nb_a | nb_b)
        coherence = len(shared) / total if total else 0.0
        novelty = (len(only_a) + len(only_b)) / total if total else 0.0
        # 균형이 좋아야 창의 (둘 다 0 아님)
        score = (coherence + novelty) / 2.0

        result = {
            'a': a,
            'b': b,
            'shared_neighbors': sorted(shared),
            'novel_a': sorted(only_a)[:5],
            'novel_b': sorted(only_b)[:5],
            'blend_name': f"{a}+{b}",
            'coherence': float(coherence),
            'novelty': float(novelty),
            'score': float(score),
        }
        self.blend_count += 1
        return result

    # ============= 4. Analogy Blend =============
    def analogy_blend(self, target: str, source: str) -> Dict[str, Any]:
        """
        Source의 인과/관계 구조를 target에 차용.
        AN 모듈이 있으면 활용, 없으면 단순 blend로 fallback.

        예: target='민석의_슬픔', source='비' → '눈물'(비가 떨어지듯 슬픔이 흐름)
        """
        result: Dict[str, Any] = {
            'target': target,
            'source': source,
            'mappings': [],
            'an_used': False,
        }

        if self.an is not None:
            try:
                if hasattr(self.an, 'find_analogy'):
                    src_neighbors = sorted(self.sa.neighbors.get(source, set()))[:5]
                    for src_nb in src_neighbors:
                        try:
                            found = self.an.find_analogy(source, src_nb, target)
                            if found:
                                # found = [(cand, score), ...] 또는 (cand, score) 또는 str
                                if isinstance(found, list) and found:
                                    cand = found[0]
                                    if isinstance(cand, tuple):
                                        cand = cand[0]
                                elif isinstance(found, tuple):
                                    cand = found[0]
                                else:
                                    cand = found
                                result['mappings'].append({
                                    'source_relation': (source, src_nb),
                                    'target_candidate': cand,
                                })
                        except Exception:
                            continue
                    result['an_used'] = True
            except Exception:
                pass

        # AN이 없거나 결과 없으면 fallback
        if not result['mappings']:
            blend = self.blend_concepts(target, source)
            result['fallback_blend'] = blend

        return result

    # ============= 5. Incubate (Wallas preparation) =============
    def incubate(self, seed: str) -> bool:
        """
        Seed를 잠재의식(DMN)에 넣고 인큐베이션 시작.

        실패 조건:
        - 창의 모드 X (도파민↓ or cortisol↑)
        - 이미 incubating
        - max 초과 (가장 오래된 거 제거)

        Returns: 성공 시 True
        """
        if not self.is_creative_mode():
            return False
        if seed in self.incubating:
            return False

        # max 초과 시 가장 오래된 것 강제 종료
        if len(self.incubating) >= self.incubation_max:
            oldest = min(self.incubating.keys(),
                        key=lambda k: self.incubating[k]['time'])
            del self.incubating[oldest]

        # 약한 활성 (DMN focus 후보로)
        if self.wm is not None and hasattr(self.wm, 'add'):
            try:
                self.wm.add(seed, salience=0.4, source='creative_seed')
            except Exception:
                pass
        if seed in self.sa.neighbors or seed in self.sa.activations:
            self.sa.activate(seed, 0.3)

        baseline = 0
        if self.dmn is not None and hasattr(self.dmn, 'spontaneous_count'):
            baseline = self.dmn.spontaneous_count

        self.incubating[seed] = {
            'time': self.time,
            'baseline': baseline,
            'harvested': False,
        }
        self.incubate_count += 1
        return True

    # ============= 6. Harvest (Wallas illumination) =============
    def harvest_incubation(self,
                          seed: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        인큐베이션 동안 DMN이 자발 활성한 카테고리 中
        seed와 멀지만 연결된 것 = illumination.

        조건:
        - 시간이 incubation_window 이상 지남
        - DMN 자발 활성 카테고리가 seed의 2-hop 이웃 (직접 X, 매개 있음)

        Returns: [{'seed', 'elapsed', 'illuminations': [...], 'count'}, ...]
        """
        results = []
        targets = [seed] if seed else list(self.incubating.keys())

        for s in targets:
            if s not in self.incubating:
                continue
            info = self.incubating[s]
            if info['harvested']:
                continue

            elapsed = self.time - info['time']
            if elapsed < self.incubation_window:
                continue

            illuminations = []
            if self.dmn is not None and hasattr(self.dmn, 'wandering_history'):
                start_t = info['time']
                seed_nb = self.sa.neighbors.get(s, set())
                seen = set()
                for entry in self.dmn.wandering_history:
                    # entry: (time, category, mode) 추정
                    if not isinstance(entry, tuple) or len(entry) < 2:
                        continue
                    t = entry[0]
                    cat = entry[1]
                    mode = entry[2] if len(entry) >= 3 else 'unknown'
                    if t < start_t:
                        continue
                    if cat == s or cat in seed_nb:
                        continue  # 너무 가까움 (직접 연결)
                    if cat in seen:
                        continue
                    seen.add(cat)
                    # 2-hop 확인: seed의 이웃 중 하나의 이웃이면 의미적 다리 있음
                    for n in seed_nb:
                        if cat in self.sa.neighbors.get(n, set()):
                            illuminations.append({
                                'category': cat,
                                'time': t,
                                'mode': mode,
                                'distance': 2,
                                'via': n,
                            })
                            break

            # 결정론 정렬 (시간↑, 알파벳↑)
            illuminations.sort(key=lambda x: (x['time'], x['category']))

            res = {
                'seed': s,
                'elapsed': elapsed,
                'illuminations': illuminations[:5],
                'count': len(illuminations),
            }
            results.append(res)
            info['harvested'] = True

            if illuminations:
                self.illuminate_count += 1
                self.illuminations.append({
                    'time': self.time,
                    'seed': s,
                    'top': illuminations[0],
                })
                if len(self.illuminations) > 50:
                    self.illuminations = self.illuminations[-50:]

        # 수확된 거 정리
        self.incubating = {k: v for k, v in self.incubating.items()
                          if not v['harvested']}

        return results

    # ============= 7. Verify Idea =============
    def verify_idea(self,
                   idea: str,
                   target_context: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        아이디어 평가 (Wallas verification).

        - novelty: 그래프 이웃 수 (적을수록 새로움, 단 0이면 의미 없음)
        - coherence: target과 연결되는 비율
        - confidence: SD 활용 (없으면 0.5)
        - score: 가중 평균

        Returns: dict
        """
        target_context = target_context or set()

        nb = self.sa.neighbors.get(idea, set())
        if not nb:
            novelty = 0.0  # 그래프에 없음 = 평가 불가
            connectedness = 0.0
        else:
            # 5개 미만 = 새로움 (지수 감쇠)
            novelty = float(np.exp(-len(nb) / 5.0))
            connectedness = 1.0

        coherence = 0.0
        if target_context and nb:
            connected = sum(1 for t in target_context if t in nb)
            coherence = connected / len(target_context)

        confidence = 0.5
        if self.sd is not None:
            try:
                if hasattr(self.sd, 'evaluate_claim'):
                    res = self.sd.evaluate_claim(idea)
                    if isinstance(res, dict):
                        doubt = res.get('doubt_level', res.get('doubt', 0.5))
                        confidence = 1.0 - float(doubt)
            except Exception:
                pass

        score = (novelty * 0.4 + coherence * 0.3 + confidence * 0.3)

        self.verify_count += 1
        return {
            'idea': idea,
            'novelty': float(novelty),
            'coherence': float(coherence),
            'confidence': float(confidence),
            'connectedness': float(connectedness),
            'score': float(score),
        }

    # ============= tick =============
    def tick(self, dt: float):
        """시간 진행 + 만료된 인큐베이션 자동 수확."""
        self.time += dt
        self.tick_count += 1

        if self.incubating:
            ready = [s for s, info in self.incubating.items()
                    if not info['harvested']
                    and self.time - info['time'] >= self.incubation_window]
            for s in ready:
                self.harvest_incubation(s)

    # ============= 상태 =============
    def get_state(self) -> Dict:
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'creative_mode': self.is_creative_mode(),
            'incubating': sorted(self.incubating.keys()),
            'illuminations_total': self.illuminate_count,
            'remote_count': self.remote_count,
            'blend_count': self.blend_count,
            'incubate_count': self.incubate_count,
            'verify_count': self.verify_count,
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Creative(t={s['time']:.1f}, mode={s['creative_mode']}, "
                f"incubating={len(s['incubating'])}, "
                f"illuminate={s['illuminations_total']})")
