"""
EVE v35 - E7: CreativeAdvanced (창의 강화)
============================================

기존 Creative (v32) 위에 새 모듈 (VSA/HG/DR) 통합:
- E5 VSA → 더 풍부한 합성 (composite vectors → 새 카테고리)
- E2 HyperGraph → 사건 상상 (epistemic='hypothesis')
- E4 DeepReasoning → 근거 있는 통찰 다리
- 발산-수렴 사이클 (자동)
- Wallas 4단계 통합 자동화

기존 cr (Creative) 함수와의 차이:
| 기존 | E7 |
|------|----|
| blend_concepts(a, b) | compose_novel(seeds, n=5) — VSA 기반 |
| find_remote_associates | find_insight_bridge(a, b) — DR path 다양성 |
| incubate/harvest | creative_cycle(seed) — Wallas 4단계 자동 |
|  -- | imagine_event(seeds) — HG 기반 사건 상상 |
|  -- | divergent_converge(seed) — 놀라운 후보 선별 |

EVE 원칙 부합:
- 결정론적 (정렬 + heapq)
- 카테고리 = 의미 단위 유지
- read-only on 기존 그래프 (옵션 hg.add는 epistemic='hypothesis')
- 호르몬 게이트 (DA↑ + Cort↓ → 창의 모드)
- 학계 (Wallas, Mednick, Boden, Beaty, Hofstadter)

Author: 김민석 + Claude v35
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class CreativeAdvanced:
    """
    창의 사이클 통합 모듈.

    의존:
        sa: SpreadingActivation (필수)
        hs: HormoneSystem (필수)
        vsa: VSABinding (선택, 합성)
        hg: HyperGraph (선택, 사건 상상)
        dr: DeepReasoning (선택, 통찰 다리)
        cr: Creative (선택, 기존 RAT/blend 활용)
        dmn: DMN (선택, incubation)
        apc: APCLearner (선택, 놀라움 측정)
    """

    def __init__(self,
                 spreading_activation,
                 hormone_system,
                 vsa=None,
                 hypergraph=None,
                 deep_reasoning=None,
                 creative=None,
                 dmn=None,
                 apc=None,
                 da_threshold: float = 0.4,
                 cort_threshold: float = 0.6,
                 use_hormone_modulation: bool = True):
        """
        Args:
            da_threshold: DA가 이 이상이어야 창의 모드 (Csikszentmihalyi)
            cort_threshold: Cort가 이 이하여야 창의 모드 (스트레스 = 발산 X)
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.vsa = vsa
        self.hg = hypergraph
        self.dr = deep_reasoning
        self.cr = creative
        self.dmn = dmn
        self.apc = apc

        self.da_threshold = float(da_threshold)
        self.cort_threshold = float(cort_threshold)
        self.use_hormone_modulation = bool(use_hormone_modulation)

        # 사이클 상태
        self.active_cycles: Dict[str, Dict] = {}  # seed → {phase, started, harvest}
        self.completed_cycles: List[Dict] = []

        # 통계
        self.compose_novel_count = 0
        self.imagine_event_count = 0
        self.bridge_count = 0
        self.divergent_count = 0
        self.cycle_count = 0
        self.cycle_completed = 0

    # ============= 1. 호르몬 게이트 =============
    def is_creative_mode(self) -> bool:
        """창의 모드 진입 가능?"""
        if self.hs is None:
            return True
        da = self.hs.hormones['dopamine'].level
        cort = self.hs.hormones['cortisol'].level
        return da >= self.da_threshold and cort <= self.cort_threshold

    # ============= 2. compose_novel (VSA 기반) =============
    def compose_novel(self,
                     seeds: List[str],
                     n: int = 5,
                     min_novelty: float = 0.3) -> List[Dict[str, Any]]:
        """
        seed 카테고리들로 새로운 합성 N개 시도.

        VSA bind/compose로 다양한 조합 → 가장 *novel*한 것들.
        novelty = 1 - max(similarity to existing categories)

        Args:
            seeds: 합성 재료 카테고리 (최소 2개)
            n: 시도할 조합 수
            min_novelty: 이 이하면 진부 — 제외

        Returns:
            [{'composite_name', 'vector', 'novelty', 'parts'}]
        """
        if self.vsa is None or len(seeds) < 2:
            return []
        if not self.is_creative_mode():
            return []

        seeds = sorted(set(seeds))  # 결정론
        results: List[Dict[str, Any]] = []

        # 다양한 role-filler 조합 시도
        roles = ['_A', '_B', '_C', '_D', '_E']
        attempts = []
        # 결정론적 페어 생성 (i, j) i<j
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                # 2-binding
                attempts.append({roles[0]: seeds[i], roles[1]: seeds[j]})
                if len(seeds) >= 3:
                    for k in range(j+1, len(seeds)):
                        # 3-binding
                        attempts.append({roles[0]: seeds[i],
                                        roles[1]: seeds[j],
                                        roles[2]: seeds[k]})

        for attempt in attempts[:n*3]:  # 3배 후보 → 필터
            r = self.vsa.compose(attempt, register_in_sa=False)
            # novelty 측정 — 기존 카테고리와 가장 비슷한 것
            # (자기 자신 + seed들은 제외)
            exclude = set(attempt.values()) | {r['composite_name']}
            nearest = self.vsa.nearest_category(r['vector'], k=1,
                                               exclude=exclude)
            max_sim = nearest[0][1] if nearest else 0.0
            novelty = 1.0 - max_sim

            if novelty < min_novelty:
                continue

            results.append({
                'composite_name': r['composite_name'],
                'vector': r['vector'],
                'novelty': float(novelty),
                'parts': dict(attempt),
                'most_similar': nearest[0][0] if nearest else None,
                'similarity': float(max_sim),
            })

            if len(results) >= n:
                break

        # novelty ↓ 정렬
        results.sort(key=lambda x: (-x['novelty'], x['composite_name']))

        if self.use_hormone_modulation and self.hs is not None and results:
            self.hs.hormones['dopamine'].stimulate(0.04)

        self.compose_novel_count += 1
        return results[:n]

    # ============= 3. imagine_event (HG 기반) =============
    def imagine_event(self,
                     seeds: List[str],
                     event_type: str = 'imagined') -> Optional[Dict[str, Any]]:
        """
        seed 카테고리들로 새 가상 사건 등록.

        epistemic='hypothesis'로 표시 → 다른 모듈은 사실과 구분 가능.

        Args:
            seeds: 사건 구성 카테고리들
            event_type: source 라벨

        Returns:
            {'edge_id', 'nodes', 'roles_assigned', 'novelty'}
        """
        if self.hg is None or len(seeds) < 2:
            return None
        if not self.is_creative_mode():
            return None

        seeds = sorted(set(seeds))

        # role 자동 할당 (결정론적, 기본 패턴)
        # 첫 번째 → agent, 두 번째 → action/patient, 세 번째 → location
        role_map = {}
        default_roles = ['agent', 'patient', 'location', 'time', 'attribute']
        for i, seed in enumerate(seeds):
            role_map[seed] = default_roles[i] if i < len(default_roles) else f'role_{i}'

        # novelty: 이 카테고리 조합이 기존 hg에 있나?
        existing = self.hg.find_edges_with(*seeds, require_all=True)
        novelty = 1.0 if not existing else 0.5

        # hypothesis로 등록
        edge_id = self.hg.add_edge(
            nodes=set(seeds),
            roles=role_map,
            weight=0.4,  # 약하게 — 가설이므로
            epistemic='hypothesis',
            source=event_type,
        )

        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['dopamine'].stimulate(0.03)

        self.imagine_event_count += 1
        return {
            'edge_id': edge_id,
            'nodes': set(seeds),
            'roles_assigned': role_map,
            'novelty': float(novelty),
        }

    # ============= 4. find_insight_bridge (DR 기반) =============
    def find_insight_bridge(self,
                           a: str,
                           b: str,
                           k: int = 3,
                           prefer_surprising: bool = True) -> List[Dict[str, Any]]:
        """
        a → b 사이 *통찰* 다리 (먼 의미 거리 + 최단 path).

        '먼 카테고리 사이 짧은 경로' = 놀라움 = 통찰 (Mednick).

        Args:
            a, b: 시작/끝
            k: top-k 경로
            prefer_surprising: 가장 surprising한 거 우선

        Returns:
            [{'path', 'depth', 'surprise', 'score'}]
        """
        if self.dr is None:
            return []
        if not self.is_creative_mode():
            return []

        paths = self.dr.find_paths_top_k(a, b, k=k*2,
                                        mode='hybrid', max_depth=6)
        if not paths:
            return []

        # 각 path에 surprise 측정
        results = []
        for p in paths:
            depth = p['depth']
            score = p['score']
            # surprise = 깊이 / score 비율 (짧고 강한 경로일수록 통찰)
            # 실제로는 1단계 직접 연결보다는 2-3단계가 더 통찰
            if depth <= 1:
                surprise = 0.1  # 직접 연결은 trivial
            else:
                surprise = score * (1.0 / depth)
            results.append({
                'path': p['path'],
                'depth': depth,
                'score': float(score),
                'surprise': float(surprise),
            })

        # 결정론 정렬
        if prefer_surprising:
            results.sort(key=lambda x: (-x['surprise'], tuple(x['path'])))
        else:
            results.sort(key=lambda x: (x['depth'], tuple(x['path'])))

        if self.use_hormone_modulation and self.hs is not None and results:
            self.hs.hormones['dopamine'].stimulate(0.03)
            self.hs.hormones['acetylcholine'].stimulate(0.02)

        self.bridge_count += 1
        return results[:k]

    # ============= 5. divergent_converge =============
    def divergent_converge(self,
                          seed: str,
                          n_diverge: int = 5,
                          n_converge: int = 2) -> List[Dict[str, Any]]:
        """
        발산: seed 이웃 + 합성 후보 N개
        수렴: 가장 *놀라운* 또는 *novel*한 K개 선택

        Args:
            seed: 시작 카테고리
            n_diverge: 발산 후보 수
            n_converge: 최종 수렴 수

        Returns:
            [{'idea', 'kind', 'novelty/surprise'}]
        """
        if not self.is_creative_mode():
            return []

        # 발산: 다양한 후보
        candidates: List[Dict[str, Any]] = []

        # 1) SA 이웃 (결정론 정렬)
        if seed in self.sa.neighbors:
            neighbors = sorted(self.sa.neighbors[seed])  # 알파벳
            for n in neighbors[:n_diverge]:
                w = self.sa.get_weight(seed, n)
                candidates.append({
                    'idea': n,
                    'kind': 'neighbor',
                    'score': float(w),
                })

        # 2) VSA 합성 (있으면)
        if self.vsa is not None and len(candidates) > 0:
            # seed + 첫 이웃 합성
            first_n = candidates[0]['idea']
            r = self.vsa.compose({'_A': seed, '_B': first_n},
                               register_in_sa=False)
            candidates.append({
                'idea': r['composite_name'],
                'kind': 'composite',
                'score': 0.5,
            })

        # 3) HG 사건 (있으면) — seed가 포함된 사건의 다른 노드
        if self.hg is not None:
            n_via = self.hg.neighbors_via_edges(seed)
            for n, w in sorted(n_via.items(), key=lambda x: (-x[1], x[0]))[:3]:
                candidates.append({
                    'idea': n,
                    'kind': 'event_neighbor',
                    'score': float(w),
                })

        # 수렴: 가장 *놀라운* 후보 (predict_next에 안 나오는 거)
        # APC가 있으면 prediction 사용
        predicted = set()
        if self.apc is not None:
            try:
                preds = self.apc.predict_next({seed}, top_k=10)
                predicted = {p[0] if isinstance(p, tuple) else p
                            for p in (preds or [])}
            except Exception:
                pass

        # surprising = 예측 X
        for c in candidates:
            c['surprise'] = 0.0 if c['idea'] in predicted else 1.0

        # 결정론 정렬 (surprise ↓, score ↓, idea ↑)
        candidates.sort(key=lambda x: (-x['surprise'], -x['score'], x['idea']))

        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['dopamine'].stimulate(0.03)

        self.divergent_count += 1
        return candidates[:n_converge]

    # ============= 6. creative_cycle (Wallas 4단계) =============
    def creative_cycle(self,
                      seed: str,
                      auto_incubate: bool = True) -> Dict[str, Any]:
        """
        Wallas 4단계 자동 사이클:
        1. preparation: seed 활성, 자료 수집 (이웃, 합성 후보)
        2. incubation: DMN 자생 (호출자가 tick 진행)
        3. illumination: 발산-수렴, 새 합성, 통찰 다리
        4. verification: SD/cr.verify (있으면)

        Args:
            seed: 시작 카테고리
            auto_incubate: True면 즉시 incubate 마킹 (호출자가 tick 후 harvest)

        Returns:
            {
                'phase': 'illumination',
                'preparation': {neighbors, vocabulary},
                'illuminations': List of ideas,
                'cycle_id': seed,
            }
        """
        if not self.is_creative_mode():
            return {'phase': 'blocked', 'reason': 'not creative mode'}

        # === 1. Preparation ===
        prep = {
            'seed': seed,
            'neighbors': list(self.sa.neighbors.get(seed, set()))[:10],
            'vocabulary_size': len(self.sa.neighbors),
        }

        # === 2. Incubation 마크 ===
        if auto_incubate and self.cr is not None:
            try:
                self.cr.incubate(seed)
            except Exception:
                pass

        # === 3. Illumination — 다중 모듈 통합 ===
        illuminations = []

        # 3a) divergent-converge
        dc = self.divergent_converge(seed, n_diverge=8, n_converge=3)
        for d in dc:
            illuminations.append({
                'kind': 'divergent_converge',
                'content': d['idea'],
                'meta': d,
            })

        # 3b) compose_novel (다음 이웃 1-2개 + seed)
        if prep['neighbors'] and self.vsa is not None:
            seeds_for_compose = [seed] + sorted(prep['neighbors'])[:2]
            novel = self.compose_novel(seeds_for_compose, n=2)
            for nv in novel:
                illuminations.append({
                    'kind': 'compose_novel',
                    'content': nv['composite_name'],
                    'meta': nv,
                })

        # 3c) insight bridge — seed에서 멀리 있는 카테고리로
        if self.dr is not None and prep['neighbors']:
            # 2-hop 이웃 (먼 카테고리)
            far_cats = set()
            for n in prep['neighbors'][:3]:
                if n in self.sa.neighbors:
                    far_cats.update(self.sa.neighbors[n])
            far_cats.discard(seed)
            for n in prep['neighbors']:
                far_cats.discard(n)

            if far_cats:
                target = sorted(far_cats)[0]  # 결정론
                bridges = self.find_insight_bridge(seed, target, k=2)
                for br in bridges:
                    illuminations.append({
                        'kind': 'insight_bridge',
                        'content': ' → '.join(br['path']),
                        'meta': br,
                    })

        # === 4. Verification ===
        verified = []
        for ill in illuminations:
            ok = True
            if self.cr is not None and hasattr(self.cr, 'verify_idea'):
                try:
                    v = self.cr.verify_idea(ill['content'])
                    if isinstance(v, dict):
                        ok = v.get('keep', True)
                except Exception:
                    pass
            if ok:
                verified.append(ill)

        if self.use_hormone_modulation and self.hs is not None:
            self.hs.hormones['dopamine'].stimulate(0.04)

        self.cycle_count += 1
        if verified:
            self.cycle_completed += 1

        result = {
            'phase': 'illumination',
            'preparation': prep,
            'illuminations': verified,
            'illuminations_count': len(verified),
            'cycle_id': seed,
        }
        self.completed_cycles.append(result)
        # 메모리 cap
        if len(self.completed_cycles) > 50:
            self.completed_cycles = self.completed_cycles[-50:]

        return result

    # ============= 7. tick + stats =============
    def tick(self, dt: float = 0.5):
        """no-op (외부 자극 모듈)"""
        pass

    def stats(self) -> Dict[str, Any]:
        return {
            'creative_mode': self.is_creative_mode(),
            'compose_novel_count': self.compose_novel_count,
            'imagine_event_count': self.imagine_event_count,
            'bridge_count': self.bridge_count,
            'divergent_count': self.divergent_count,
            'cycle_count': self.cycle_count,
            'cycle_completed': self.cycle_completed,
            'completed_cycles_log': len(self.completed_cycles),
        }

    def __repr__(self):
        s = self.stats()
        return (f"CreativeAdvanced(cycles={s['cycle_count']}, "
                f"compose={s['compose_novel_count']}, "
                f"bridges={s['bridge_count']})")
