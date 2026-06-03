"""
EVE v32 - C5: Temporal (시간 추론)
====================================

EVE의 시간 인지 — 과거 회상 + 미래 시뮬 + 주관적 시간.

학계:
- Tulving 1985 — Chronesthesia (mental time travel)
- Schacter & Addis 2007 — Constructive Memory + future simulation
- Friedman 1993 — Memory for time
- Suddendorf & Corballis 2007 — Mental time travel evolution
- Wittmann 2009 — Subjective time + neuromodulation
- Droit-Volet 2013 — Emotion & time perception (cort↑ → 시간 길게)

핵심 메커니즘:

A. 과거 회상 (recall_past):
   EM에서 카테고리 포함 일화를 시간 역순. 가까운 것부터.

B. 최근 일화 (recent_episodes):
   sim 시간 기준 윈도우 안의 일화들.

C. 미래 시뮬 (imagine_future):
   CG.children() 따라 인과 사슬 펼침 (depth 만큼).
   "X 일어났으니 다음에 뭐가?" 시간순 미래 예측.

D. 시간 거리 (time_distance):
   카테고리 마지막 발생 ~ 현재 사이 경과.

E. 시간 순서 (precedes):
   EM에서 a, b 마지막 발생 시간 비교.

F. 주관적 시간 (perceived_time_factor):
   호르몬 따라 시간 길게/짧게 느낌.
   - cortisol↑ (스트레스) → factor > 1 (시간 길게)
   - dopamine↑ (즐거움) → factor < 1 (시간 짧게)
   - melatonin↑ (졸림) → factor > 1 (시간 길게)
   - DA + Dopamine 중심 (Wittmann 2009 검증된 발견)

EVE 절대 원칙 부합:
- 확률 X (random/sample/softmax 미사용)
- 결정론 (같은 EM/CG → 같은 시뮬)
- 호르몬 변조 (주관적 시간만, 실제 시간 X)
- 카테고리 단위
- Read-only on EM/CG/SA

핵심 함수:
1. recall_past(category, n)
2. recent_episodes(window)
3. imagine_future(seed, depth, top_n_per_step)
4. time_distance(category)
5. precedes(a, b)
6. perceived_time_factor()
7. tick(dt)

Author: 김민석 + Claude v32 (C5)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set


class Temporal:
    """EVE v32 C5 Temporal — 시간 추론"""

    DEFAULT_RECALL_N = 5
    DEFAULT_RECENT_WINDOW = 60.0  # 초
    DEFAULT_FUTURE_DEPTH = 3
    DEFAULT_FUTURE_BRANCH = 3  # depth마다 top N 자식

    # 주관적 시간 호르몬 영향 강도
    SUBJECTIVE_CORT_FACTOR = 0.5    # cort 0.5 이상부터 비례 증가
    SUBJECTIVE_DOPAMINE_FACTOR = 0.4
    SUBJECTIVE_MELATONIN_FACTOR = 0.3

    def __init__(self,
                 episodic_memory=None,
                 spreading_activation=None,
                 causal_graph=None,
                 hormone_system=None):
        """
        Args:
            episodic_memory: 옵션 (None이면 recall 기능 제한)
            spreading_activation: 옵션 (history fallback)
            causal_graph: 옵션 (imagine_future)
            hormone_system: 옵션 (주관적 시간)
        """
        self.em = episodic_memory
        self.sa = spreading_activation
        self.cg = causal_graph
        self.hs = hormone_system

        self.time = 0.0

        # 통계
        self.recall_count = 0
        self.future_count = 0
        self.precedes_count = 0

    # ============= 1. recall_past =============
    def recall_past(self, category: str,
                   n: int = None) -> List[Dict[str, Any]]:
        """
        카테고리 관련 과거 일화 시간 역순 (가장 최근부터).

        Returns: [{'time', 'categories', 'summary'}, ...]
                 EM 없으면 SA history fallback
        """
        if n is None:
            n = self.DEFAULT_RECALL_N
        self.recall_count += 1

        results: List[Dict[str, Any]] = []

        # 1) EM에서
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                episodes = list(self.em.episodes.values())
                # 카테고리 포함 일화만 + 시간 역순
                matching = [
                    ep for ep in episodes
                    if category in getattr(ep, 'categories', set())
                ]
                # encoded_at 기준 역순
                matching.sort(
                    key=lambda ep: getattr(ep, 'encoded_at', 0),
                    reverse=True
                )
                for ep in matching[:n]:
                    results.append({
                        'time': float(getattr(ep, 'encoded_at', 0)),
                        'categories': set(getattr(ep, 'categories', set())),
                        'summary': str(getattr(ep, 'summary', '')),
                        'source': 'em',
                    })
            except Exception:
                pass

        # 2) EM 부족 → SA history fallback
        if len(results) < n and self.sa is not None and hasattr(self.sa, 'activation_history'):
            seen_times: Set[float] = {r['time'] for r in results}
            history = self.sa.activation_history
            sa_matches = [
                (t, c) for t, c in reversed(history)
                if c == category
            ]
            for t, c in sa_matches[:n - len(results)]:
                if t in seen_times:
                    continue
                results.append({
                    'time': float(t),
                    'categories': {c},
                    'summary': '',
                    'source': 'sa',
                })

        return results[:n]

    # ============= 2. recent_episodes =============
    def recent_episodes(self, window: float = None,
                       max_n: int = 20) -> List[Dict[str, Any]]:
        """
        최근 N초 안의 일화들 (EM 기반).

        Args:
            window: 시간 윈도우 (초). None=DEFAULT
            max_n: 최대 반환 개수

        Returns: [{'time', 'categories', 'summary'}, ...]
        """
        if window is None:
            window = self.DEFAULT_RECENT_WINDOW

        cutoff = self.time - window
        results: List[Dict[str, Any]] = []

        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                episodes = list(self.em.episodes.values())
                # encoded_at 시간 역순
                episodes.sort(
                    key=lambda ep: getattr(ep, 'encoded_at', 0),
                    reverse=True
                )
                for ep in episodes[:max_n * 2]:  # 여유 있게
                    t = getattr(ep, 'encoded_at', 0)
                    if t < cutoff:
                        break  # 더 오래됨
                    results.append({
                        'time': float(t),
                        'categories': set(getattr(ep, 'categories', set())),
                        'summary': str(getattr(ep, 'summary', '')),
                    })
                    if len(results) >= max_n:
                        break
            except Exception:
                pass
        return results

    # ============= 3. imagine_future =============
    def imagine_future(self, seed: str,
                      depth: int = None,
                      top_n_per_step: int = None) -> Dict[str, Any]:
        """
        "X 일어났으니 다음엔?" 미래 인과 시뮬.

        CG.children() 따라 layer-by-layer 펼침. 각 layer에서 강한 자식 top_n.
        결정론 (가중치 ↓, 알파벳 ↑).

        Returns: {
            'seed': str,
            'timeline': [{'depth': N, 'category': str, 'weight': float}],
            'depth_reached': N,
        }
        """
        if depth is None:
            depth = self.DEFAULT_FUTURE_DEPTH
        if top_n_per_step is None:
            top_n_per_step = self.DEFAULT_FUTURE_BRANCH

        self.future_count += 1

        timeline: List[Dict[str, Any]] = []

        if self.cg is None:
            return {'seed': seed, 'timeline': [], 'depth_reached': 0}

        current_layer = {seed}
        visited: Set[str] = {seed}

        for d in range(1, depth + 1):
            # 현재 layer의 자식들 모으기
            next_candidates: Dict[str, float] = {}
            for cur in current_layer:
                for child, w in self.cg.children(cur).items():
                    if child in visited:
                        continue
                    # 같은 자식이 여러 부모에서 오면 max weight
                    if child in next_candidates:
                        next_candidates[child] = max(next_candidates[child], w)
                    else:
                        next_candidates[child] = w

            if not next_candidates:
                break

            # top_n 선택 (결정론)
            sorted_cands = sorted(
                next_candidates.items(),
                key=lambda kv: (-kv[1], kv[0])
            )[:top_n_per_step]

            new_layer = set()
            for cat, w in sorted_cands:
                timeline.append({
                    'depth': d,
                    'category': cat,
                    'weight': float(w),
                })
                new_layer.add(cat)
                visited.add(cat)

            current_layer = new_layer

        return {
            'seed': seed,
            'timeline': timeline,
            'depth_reached': max([t['depth'] for t in timeline], default=0),
        }

    # ============= 4. time_distance =============
    def time_distance(self, category: str) -> Optional[float]:
        """
        카테고리 마지막 발생 후 경과 시간 (초).

        Returns: 경과 시간 또는 None (한 번도 발생 X)
        """
        last_time = None

        # EM에서
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                for ep in self.em.episodes.values():
                    if category in getattr(ep, 'categories', set()):
                        t = getattr(ep, 'encoded_at', None)
                        if t is not None and (last_time is None or t > last_time):
                            last_time = t
            except Exception:
                pass

        # SA history도 검사
        if self.sa is not None and hasattr(self.sa, 'activation_history'):
            try:
                for t, c in reversed(self.sa.activation_history):
                    if c == category:
                        if last_time is None or t > last_time:
                            last_time = t
                        break  # 역순 중 첫 매치가 가장 최근
            except Exception:
                pass

        if last_time is None:
            return None
        return float(max(0.0, self.time - last_time))

    # ============= 5. precedes =============
    def precedes(self, a: str, b: str) -> Optional[bool]:
        """
        A가 B보다 시간상 먼저?

        Returns: True/False/None (둘 중 하나 발생 X)
        """
        self.precedes_count += 1

        a_last = self._last_seen(a)
        b_last = self._last_seen(b)

        if a_last is None or b_last is None:
            return None
        return a_last < b_last

    def _last_seen(self, category: str) -> Optional[float]:
        """카테고리 마지막 발생 시간 (절대 시간, time_distance와 다름)"""
        last_time = None
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                for ep in self.em.episodes.values():
                    if category in getattr(ep, 'categories', set()):
                        t = getattr(ep, 'encoded_at', None)
                        if t is not None and (last_time is None or t > last_time):
                            last_time = t
            except Exception:
                pass
        if self.sa is not None and hasattr(self.sa, 'activation_history'):
            try:
                for t, c in reversed(self.sa.activation_history):
                    if c == category:
                        if last_time is None or t > last_time:
                            last_time = t
                        break
            except Exception:
                pass
        return last_time

    # ============= 6. perceived_time_factor =============
    def perceived_time_factor(self) -> float:
        """
        주관적 시간 배수.
        > 1.0: 시간 길게 느껴짐 (cort↑, mel↑)
        < 1.0: 시간 짧게 느껴짐 (da↑)

        Returns: factor (대략 0.5 ~ 2.0)
        """
        factor = 1.0
        if self.hs is None:
            return factor

        # cortisol — 스트레스 시 시간 길게
        cort = self.hs.hormones.get('cortisol')
        if cort is not None and cort.level > 0.5:
            factor += (cort.level - 0.5) * self.SUBJECTIVE_CORT_FACTOR * 2

        # dopamine — 즐거울 때 시간 짧게
        da = self.hs.hormones.get('dopamine')
        if da is not None and da.level > 0.5:
            factor -= (da.level - 0.5) * self.SUBJECTIVE_DOPAMINE_FACTOR * 2

        # melatonin — 졸리면 시간 길게
        mel = self.hs.hormones.get('melatonin')
        if mel is not None and mel.level > 0.5:
            factor += (mel.level - 0.5) * self.SUBJECTIVE_MELATONIN_FACTOR * 2

        return float(np.clip(factor, 0.3, 3.0))

    # ============= 7. tick =============
    def tick(self, dt: float):
        """시간 누적."""
        if dt is None or dt <= 0:
            return
        self.time += float(dt)

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'recall_count': self.recall_count,
            'future_count': self.future_count,
            'precedes_count': self.precedes_count,
            'subjective_factor': self.perceived_time_factor(),
        }

    def __repr__(self):
        s = self.get_state()
        return (f"Temporal(t={s['time']:.1f}s, "
                f"recall={s['recall_count']}, "
                f"future={s['future_count']}, "
                f"subj={s['subjective_factor']:.2f}x)")
