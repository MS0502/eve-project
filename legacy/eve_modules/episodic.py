"""
EVE v32 - B2: EpisodicMemory (Hybrid)
======================================

EVE의 일화 기억 (생각도 일화로 저장).

Hybrid 표현 (3종):
1. tuple_data: 구조화 (actor, action, object, valence)
2. categories: 카테고리 set (검색 키, 결정론)
3. summary: 자연어 요약 (인간 읽기)

기능 7개:
1. encode()              - 일화 저장
2. recall(cue)           - 회상 (pattern completion: Jaccard 유사도)
3. remind(focus)         - DMN/WM이 떠올릴 때
4. auto_encode_from_wm() - WM focus 변경 시 자동 encode
5. consolidate(dt)       - 호르몬 변조 (BDNF/cortisol/endorphin)
6. forget(dt)            - Ebbinghaus 망각 (recall_count 보호)
7. get_state()           - 상태 조회

학계:
- Tulving 1972        - Episodic vs Semantic
- Teyler & DiScenna 1986 - Hippocampus indexing theory
- Hopfield 1982       - Pattern completion
- Squire 1992         - Memory consolidation
- McEwen 2007         - Cortisol & memory (chronic suppresses, acute strengthens)
- Ebbinghaus 1885     - Forgetting curve

EVE 절대 원칙:
- 확률 X (np.random/sample/softmax X) ✅
- 결정론 (Jaccard, 가중치 정렬) ✅
- 호르몬 변조 ✅
- 카테고리 단위 ✅

Author: 김민석 + Claude v32
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


# ============= Episode (Hybrid) =============

@dataclass
class Episode:
    """일화 (Hybrid 표현)"""
    id: str                                   # 'ep_NNN'
    time: float                               # 인코딩 시점
    last_recalled: float                      # 마지막 회상
    recall_count: float = 0.0                 # float (시간 감쇠 적용 - WM lock 교훈)

    # Hybrid 표현
    tuple_data: Dict[str, Any] = field(default_factory=dict)  # {actor, action, object, valence}
    categories: Set[str] = field(default_factory=set)         # 검색 키
    summary: str = ""                                          # 자연어 요약

    # 호르몬/감정 snapshot (저장 시점)
    mood: Dict[str, float] = field(default_factory=dict)
    primary_hormone: Tuple[str, float] = ("", 0.0)

    # 강도/보호
    salience: float = 0.5
    is_protected: bool = False


# ============= EpisodicMemory =============

class EpisodicMemory:
    """
    EVE v32 EpisodicMemory.

    사용:
        em = EpisodicMemory(sa, wm, hs, nl=nl)
        # 자동 encode (매 tick에서)
        em.update(dt)
        # 수동 encode
        em.encode(categories={'민석', '대화'}, tuple_data={...}, summary='...')
        # 회상
        eps = em.recall({'민석'}, top_n=3)
        # DMN 떠올림
        ep = em.remind('민석')
    """

    def __init__(self,
                 spreading_activation,
                 working_memory,
                 hormone_system,
                 natural_lang=None,
                 capacity: int = 500,
                 base_forget_rate: float = 0.001,
                 auto_encode_threshold: float = 0.5,
                 protected_categories: Optional[Set[str]] = None):
        """
        Args:
            sa, wm, hs: A2/A4/A1 인스턴스
            natural_lang: B1 인스턴스 (옵션, summary 자동 생성용)
            capacity: 최대 일화 수 (LRU evict)
            base_forget_rate: 기본 망각률
            auto_encode_threshold: WM focus salience >= 임계 시 자동 encode
            protected_categories: 보호 카테고리 (일화 보호 ✕ - 카테고리 자체 보호 X)
        """
        self.sa = spreading_activation
        self.wm = working_memory
        self.hs = hormone_system
        self.nl = natural_lang

        self.capacity = capacity
        self.base_forget_rate = base_forget_rate
        self.auto_encode_threshold = auto_encode_threshold
        self.protected_categories = protected_categories or set()

        self.episodes: Dict[str, Episode] = {}
        self.category_index: Dict[str, Set[str]] = {}  # cat → ep_ids
        self.next_id = 0
        self.time = 0.0

        # auto_encode 상태 (last focus → 변경 감지)
        self._last_auto_focus: Optional[str] = None
        self._last_auto_time: float = 0.0
        self._auto_encode_min_interval: float = 1.0  # 1초 이내 같은 focus → skip

        # 통계
        self.encode_count = 0
        self.recall_count = 0
        self.remind_count = 0
        self.forget_count = 0

    # ============= 1. encode =============
    def encode(self,
              categories: Set[str],
              tuple_data: Optional[Dict] = None,
              summary: Optional[str] = None,
              salience: float = 0.5,
              is_protected: bool = False) -> Optional[Episode]:
        """
        일화 저장.

        Args:
            categories: 활성 카테고리 set (필수)
            tuple_data: 구조화 데이터 {actor, action, object, ...}
            summary: 자연어 요약 (None이면 NL로 자동 생성)
            salience: 강도 0-1
            is_protected: 핵심 일화 (forget 보호)

        Returns: Episode 또는 None (categories 비면)
        """
        if not categories:
            return None

        # capacity 체크 - 가장 약한 일화 evict
        if len(self.episodes) >= self.capacity:
            self._evict_weakest()

        # 호르몬/감정 snapshot
        mood = self.hs.compute_mood()
        primary = self.hs.primary_hormone()

        # tuple_data 기본값
        if tuple_data is None:
            tuple_data = {
                'valence': mood['valence'],
                'arousal': mood['arousal'],
            }

        # summary 자동 생성 (NL 있으면)
        if summary is None:
            summary = self._auto_summary(categories, tuple_data, mood)

        # ID 생성
        ep_id = f"ep_{self.next_id:04d}"
        self.next_id += 1

        ep = Episode(
            id=ep_id,
            time=self.time,
            last_recalled=self.time,
            recall_count=0.0,
            tuple_data=dict(tuple_data),
            categories=set(categories),
            summary=summary,
            mood=dict(mood),
            primary_hormone=tuple(primary),
            salience=float(np.clip(salience, 0.0, 1.0)),
            is_protected=is_protected,
        )

        self.episodes[ep_id] = ep
        # 인덱스 갱신
        for cat in categories:
            self.category_index.setdefault(cat, set()).add(ep_id)

        self.encode_count += 1
        return ep

    def _auto_summary(self, categories, tuple_data, mood) -> str:
        """자연어 요약 자동 생성 (NL 있으면 정교, 없으면 단순)"""
        cats_list = sorted(categories)[:3]
        cats_str = ', '.join(cats_list)

        valence = mood.get('valence', 0)
        if valence > 0.3:
            tone = "기쁨"
        elif valence < -0.3:
            tone = "걱정"
        else:
            tone = "평온"

        return f"[{tone}] {cats_str}"

    # ============= 2. recall (pattern completion) =============
    def recall(self,
              cue_categories: Set[str],
              top_n: int = 5,
              min_overlap: int = 1) -> List[Episode]:
        """
        카테고리 cue로 회상.

        Pattern completion: Jaccard 유사도 기반 매칭.
        결정론 (정렬 기준: similarity → salience → time 역순).

        Args:
            cue_categories: 회상 단서 카테고리들
            top_n: 상위 N개
            min_overlap: 최소 카테고리 겹침 수

        Returns: Episode 리스트 (유사도 내림차순)
        """
        if not cue_categories:
            return []

        # 후보 일화 수집 (인덱스 활용)
        candidate_ids: Set[str] = set()
        for cat in cue_categories:
            candidate_ids |= self.category_index.get(cat, set())

        if not candidate_ids:
            return []

        # 유사도 계산 (Jaccard) + 가중 정렬
        scored = []
        for ep_id in candidate_ids:
            ep = self.episodes.get(ep_id)
            if not ep:
                continue
            overlap = ep.categories & cue_categories
            if len(overlap) < min_overlap:
                continue
            union = ep.categories | cue_categories
            jaccard = len(overlap) / len(union) if union else 0.0
            # 가중: similarity + salience * 0.3 + recency
            recency = 1.0 / (1.0 + (self.time - ep.last_recalled) / 60.0)
            score = jaccard + ep.salience * 0.3 + recency * 0.1
            scored.append((score, ep))

        # 결정론 정렬 (score 내림차순, ID 오름차순)
        scored.sort(key=lambda x: (-x[0], x[1].id))

        # 회상된 일화 갱신
        recalled = [ep for _, ep in scored[:top_n]]
        for ep in recalled:
            ep.last_recalled = self.time
            ep.recall_count += 1.0

        if recalled:
            self.recall_count += 1

        return recalled

    # ============= 3. remind (DMN/WM이 떠올림) =============
    def remind(self, focus_category: str) -> Optional[Episode]:
        """
        DMN episodic 모드 또는 WM focus 변경 시 호출.

        focus 카테고리에 가장 강하게 연결된 일화 1개 반환.
        해당 일화의 카테고리들 → SA에 활성 (회상 효과: 기억이 떠오름)
        """
        if not focus_category:
            return None

        results = self.recall({focus_category}, top_n=1)
        if not results:
            return None

        ep = results[0]
        # 회상 효과: 일화의 카테고리들을 SA에 활성 (threshold 통과하도록 충분히 강하게)
        for cat in ep.categories:
            if cat != focus_category:
                self.sa.activate(cat, strength=max(0.4, ep.salience * 0.6))

        self.remind_count += 1
        return ep

    # ============= 4. auto_encode_from_wm =============
    def auto_encode_from_wm(self, dt: float):
        """
        매 tick: WM focus 변경 감지 → 자동 encode.

        조건:
        - WM focus가 이전과 다름
        - focus salience >= auto_encode_threshold
        - 마지막 auto_encode 후 _auto_encode_min_interval 경과
        """
        focus = self.wm.get_focus()
        if focus is None:
            return

        # 같은 focus 연속이면 skip
        if focus == self._last_auto_focus:
            return

        # 너무 자주 encode 방지 (1초 minimum interval)
        if self.time - self._last_auto_time < self._auto_encode_min_interval:
            return

        # focus salience 임계 확인
        focus_slot = self.wm.slots.get(focus)
        if not focus_slot or focus_slot.salience < self.auto_encode_threshold:
            return

        # 인코딩 - 현재 WM에 있는 카테고리들 모두 (focus 중심)
        all_active = set(self.wm.slots.keys())
        # focus salience를 일화 salience에 반영
        ep = self.encode(
            categories=all_active,
            tuple_data={
                'focus': focus,
                'wm_size': len(self.wm.slots),
            },
            salience=focus_slot.salience,
        )

        if ep:
            self._last_auto_focus = focus
            self._last_auto_time = self.time

    # ============= 5. consolidate (호르몬 변조) =============
    def consolidate(self, dt: float):
        """
        호르몬 → 일화 강도 변조 (Squire 1992, McEwen 2007).

        - BDNF ↑ → 모든 일화 보존 (강화)
        - 급성 cortisol (자극 직후) → 강화
        - 만성 cortisol (장기 유지) → 약화 (스트레스 해마 손상)
        - endorphin ↑ → 최근 일화 강화 (positive 강화)
        """
        if not self.episodes:
            return

        # 호르몬 영향 계산
        bdnf = self.hs.hormones['bdnf'].level if 'bdnf' in self.hs.active_hormones else 0.3
        cort = self.hs.hormones['cortisol'].level
        endo = self.hs.hormones['endorphin'].level if 'endorphin' in self.hs.active_hormones else 0.2

        # BDNF 보존 효과
        bdnf_boost = (bdnf - 0.3) * 0.01 * dt  # +0.01 per dt at full BDNF

        # 만성 cortisol 약화 (>0.7 장기)
        cort_decay = max(0.0, (cort - 0.7) * 0.02 * dt) if cort > 0.7 else 0.0

        # endorphin → 최근 일화 강화
        endo_boost = (endo - 0.2) * 0.005 * dt if endo > 0.5 else 0.0

        for ep in self.episodes.values():
            if ep.is_protected:
                continue
            # BDNF 모든 일화 보존
            if bdnf_boost > 0:
                ep.salience = min(1.0, ep.salience + bdnf_boost)
            # 만성 cortisol 약화
            if cort_decay > 0:
                ep.salience = max(0.0, ep.salience - cort_decay)
            # endorphin 최근만 (마지막 60초)
            if endo_boost > 0 and (self.time - ep.time) < 60.0:
                ep.salience = min(1.0, ep.salience + endo_boost)

    # ============= 6. forget (Ebbinghaus) =============
    def forget(self, dt: float):
        """
        Ebbinghaus 망각: 시간 경과 + recall 적은 거 약화.

        recall_count 많은 일화는 천천히 (rehearsal 효과).
        단, floor 있어서 무한 보호 X (WM lock 교훈).
        is_protected는 절대 안 사라짐.
        """
        # recall_count 시간 감쇠 (반감기 5분 = 300초, 일화는 WM보다 느림)
        recall_decay = float(np.exp(-dt / 300.0))

        to_remove = []
        for ep_id, ep in self.episodes.items():
            if ep.is_protected:
                continue

            # recall_count 시간 감쇠
            if ep.recall_count > 0:
                ep.recall_count = ep.recall_count * recall_decay

            # decay_factor: recall 많은 거 천천히, 단 floor 0.4 (영구 lock 방지)
            decay_factor = max(0.4, 1.0 / (1.0 + ep.recall_count * 0.2))
            decay_amount = self.base_forget_rate * decay_factor * dt
            ep.salience = max(0.0, ep.salience - decay_amount)

            # salience floor 0.05 → evict
            if ep.salience < 0.05:
                to_remove.append(ep_id)

        for ep_id in to_remove:
            self._remove_episode(ep_id)
            self.forget_count += 1

    def _remove_episode(self, ep_id: str):
        """일화 제거 + 인덱스 정리"""
        ep = self.episodes.pop(ep_id, None)
        if not ep:
            return
        for cat in ep.categories:
            ids = self.category_index.get(cat)
            if ids:
                ids.discard(ep_id)
                if not ids:
                    del self.category_index[cat]

    def _evict_weakest(self):
        """capacity 초과 시 가장 약한 일화 (보호 X) 제거"""
        candidates = [(ep.salience, ep.id) for ep in self.episodes.values()
                     if not ep.is_protected]
        if not candidates:
            return  # 모두 보호됨 - 새 일화 못 추가
        candidates.sort()  # 최소 salience 먼저
        weakest_id = candidates[0][1]
        self._remove_episode(weakest_id)

    # ============= 7. update (메인 tick에서 호출) =============
    def update(self, dt: float):
        """
        매 tick 호출:
        - time 누적
        - auto_encode (WM focus 변경 시)
        - consolidate (호르몬 → 강도)
        - forget (Ebbinghaus)
        """
        self.time += dt
        self.auto_encode_from_wm(dt)
        self.consolidate(dt)
        self.forget(dt)

    # ============= 8. 조회 =============
    def get_episodes_by_category(self, category: str) -> List[Episode]:
        """카테고리로 일화 직접 조회 (회상 X)"""
        ep_ids = self.category_index.get(category, set())
        return [self.episodes[eid] for eid in ep_ids if eid in self.episodes]

    def get_state(self) -> Dict:
        if self.episodes:
            saliences = [ep.salience for ep in self.episodes.values()]
            mean_sal = float(np.mean(saliences))
            max_sal = float(np.max(saliences))
        else:
            mean_sal = max_sal = 0.0

        return {
            'time': self.time,
            'num_episodes': len(self.episodes),
            'capacity': self.capacity,
            'unique_categories': len(self.category_index),
            'mean_salience': mean_sal,
            'max_salience': max_sal,
            'encode_count': self.encode_count,
            'recall_count': self.recall_count,
            'remind_count': self.remind_count,
            'forget_count': self.forget_count,
        }

    def get_recent(self, n: int = 5) -> List[Episode]:
        """최근 n개 일화 (시간 역순)"""
        eps = sorted(self.episodes.values(), key=lambda e: -e.time)
        return eps[:n]

    def __repr__(self):
        s = self.get_state()
        return (f"EpisodicMemory(eps={s['num_episodes']}/{s['capacity']}, "
                f"cats={s['unique_categories']}, mean_sal={s['mean_salience']:.2f})")
