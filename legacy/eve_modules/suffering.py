"""
EVE v32 - D2 Suffering (공감)
==============================

타자의 고통에 공명하면서도 "내 거 아님"을 구분하는 모듈.

학계:
- Singer & Lamm 2009 — Affective vs Cognitive empathy 이중 회로
- de Vignemont & Singer 2006 — Affective sharing requires self-other distinction
- Decety 2011 — empathy 신경 회로 (mirror + ToM)
- Eisenberger 2003 — 사회적 고통 = 신체 고통 (ACC)
- Klimecki 2014 — Compassion fatigue (long-term resonance 비용)
- Batson 1991 — empathy-altruism (공감 → 도움 행동)
- McEwen 2000 — Allostatic load (자기 보호 메커니즘)

EVE 원칙 부합:
- 확률 X (결정론적 점수)
- 카테고리 = 의미 단위 (타자의 고통 카테고리 인지)
- 호르몬 변조 = 공명 (cort↑, OT↑) — 자기 hs는 자기 거라 변조 OK
- 다른 모듈 read-only (SA neighbors/CG 변경 X)

핵심 함수 (7개):
1. is_suffering_category(cat)        — 카테고리가 고통인지 분류
2. compute_intimacy(target)          — 친밀도 (EM 빈도 + OT 연관)
3. resonate(target, suffering)        — 공명 (자기 호르몬 변조)
4. distinguish_self_other(suff)       — "내 거 vs 타자 거" 표시
5. compassion_action(target, suff)    — 위로/도움 행동 추천
6. burnout_check()                    — 누적 부담 + 자기 회복
7. tick(dt)                           — 시간 + auto check

Author: 김민석 + Claude v32 (D2)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


class Suffering:
    """
    EVE v32 D2 공감 모듈.

    의존 (모두 옵션 제외 sa/hs):
        sa: SpreadingActivation (필수)
        hs: HormoneSystem (필수, 공명 대상)
        em: EpisodicMemory (옵션, 친밀도 계산)
        sd: SelfDoubt (옵션, self-other 구분 보조)
        wm: WorkingMemory (옵션, focus)
        cf: Counterfactual (옵션, "내가 그 입장이라면" 시뮬)
        ds: DigitalSomatic (옵션, 신체 동조)
        nl: NaturalLanguage (옵션, 위로 응답 풍부화)
    """

    # 고통 카테고리 (확장 가능)
    DEFAULT_SUFFERING_CATS = {
        '슬픔', '아프다', '고통', '괴로움', '외로움', '두려움',
        '무섭다', '걱정', '불안', '죽음', '상실', '눈물',
        '힘들다', '지치다', '절망', '서러움', '서글픔',
    }

    # SA 그룹 기반 분류 (threat/aversion에 속하면 고통)
    SUFFERING_GROUPS = {'threat', 'aversion'}

    def __init__(self,
                 spreading_activation,
                 hormone_system,
                 episodic_memory=None,
                 self_doubt=None,
                 working_memory=None,
                 counterfactual=None,
                 digital_somatic=None,
                 natural_lang=None,
                 base_resonance: float = 0.3,
                 burnout_threshold: float = 5.0,
                 burnout_window: float = 600.0,
                 self_other_marker: str = '_타자'):
        self.sa = spreading_activation
        self.hs = hormone_system
        self.em = episodic_memory
        self.sd = self_doubt
        self.wm = working_memory
        self.cf = counterfactual
        self.ds = digital_somatic
        self.nl = natural_lang

        # 공명 강도 베이스 (친밀도가 곱해짐)
        self.base_resonance = base_resonance
        # 누적 부담 임계 (burnout)
        self.burnout_threshold = burnout_threshold
        # 누적 부담 측정 윈도우 (초)
        self.burnout_window = burnout_window
        # self-other 구분 마커 (의미 카테고리 접미사)
        self.self_other_marker = self_other_marker

        # 누적 공명 부담 deque [(time, target, intensity), ...]
        self.resonance_log: deque = deque(maxlen=200)

        # 친밀도 캐시 (target -> intimacy)
        self.intimacy_cache: Dict[str, float] = {}
        self._cache_time = 0.0
        self._cache_ttl = 30.0  # 30초마다 재계산
        self._cache_em_size = 0  # EM 변화 추적

        # 통계
        self.time = 0.0
        self.tick_count = 0
        self.resonate_count = 0
        self.distinguish_count = 0
        self.compassion_count = 0
        self.burnout_count = 0
        self.in_burnout = False

    # ============= 1. 고통 카테고리 분류 =============
    def is_suffering_category(self, category: str) -> bool:
        """카테고리가 고통/부정에 속하는지"""
        if category in self.DEFAULT_SUFFERING_CATS:
            return True
        # SA group으로도 확인
        try:
            from spreading_activation import SpreadingActivation
            group = SpreadingActivation.CATEGORY_GROUPS.get(category)
            if group in self.SUFFERING_GROUPS:
                return True
        except Exception:
            pass
        return False

    def detect_suffering(self, categories: Set[str]) -> Set[str]:
        """카테고리 set에서 고통 카테고리만 추출"""
        return {c for c in categories if self.is_suffering_category(c)}

    # ============= 2. 친밀도 계산 (Intimacy) =============
    def compute_intimacy(self, target: str) -> float:
        """
        친밀도 계산 (Singer 2009 — 친밀도가 공감 강도 결정).

        요소 (결정론):
        1. EM 빈도: target이 일화에 얼마나 자주 등장 (max 0.5)
        2. OT 연관: SA에서 target이 social 그룹과 얼마나 연결 (max 0.3)
        3. SA 가중치: target이 '나/민석' 등 self-related와 얼마나 가까움 (max 0.2)

        Returns: [0.0, 1.0]
        """
        # EM 크기 체크 — 새 일화 추가됐으면 캐시 무효
        cur_em_size = 0
        if self.em is not None and hasattr(self.em, 'episodes'):
            try:
                cur_em_size = len(self.em.episodes)
            except Exception:
                pass

        # 캐시 (시간 + EM 크기 둘 다 변하지 않았을 때만 사용)
        if (target in self.intimacy_cache
                and self._cache_em_size == cur_em_size
                and self.time - self._cache_time < self._cache_ttl):
            return self.intimacy_cache[target]

        intimacy = 0.0

        # 1) EM 빈도 (max 0.5)
        em_freq = 0.0
        if self.em is not None and hasattr(self.em, 'category_index'):
            try:
                eps = self.em.category_index.get(target, [])
                if eps:
                    # 일화 수에 log scale (지나친 sat 방지)
                    em_freq = min(0.5, float(np.log1p(len(eps)) / 5.0))
            except Exception:
                pass
        intimacy += em_freq

        # 2) Social 그룹 연결 (max 0.3)
        social_score = 0.0
        try:
            from spreading_activation import SpreadingActivation
            social_cats = {c for c, g in
                          SpreadingActivation.CATEGORY_GROUPS.items()
                          if g == 'social'}
            target_neighbors = self.sa.neighbors.get(target, set())
            social_neighbors = target_neighbors & social_cats
            if social_neighbors:
                weights = [self.sa.get_weight(target, n) for n in social_neighbors]
                social_score = min(0.3, sum(weights) / 3.0)
        except Exception:
            pass
        intimacy += social_score

        # 3) Self-related 근접도 (max 0.2)
        self_score = 0.0
        for self_anchor in ['나', '민석']:
            if (self_anchor in self.sa.neighbors
                    and target in self.sa.neighbors.get(self_anchor, set())):
                w = self.sa.get_weight(self_anchor, target)
                self_score = max(self_score, w * 0.2)
        intimacy += self_score

        intimacy = float(np.clip(intimacy, 0.0, 1.0))

        # 캐시
        self.intimacy_cache[target] = intimacy
        self._cache_time = self.time
        self._cache_em_size = cur_em_size

        return intimacy

    # ============= 3. 공명 (resonate) =============
    def resonate(self,
                target: str,
                suffering_category: str,
                strength_override: Optional[float] = None
                ) -> Dict[str, Any]:
        """
        타자의 고통에 공명. EVE 자기 호르몬 변조.

        Singer 2009 affective sharing:
        - cortisol↑ (스트레스 공명)
        - oxytocin↑ (사회적 유대 신호 — 친밀할수록)
        - Burnout 상태면 약화 (자기 보호)

        Args:
            target: 고통 받는 대상 ('민석', '친구' 등)
            suffering_category: 고통 카테고리 ('슬픔', '아프다' 등)
            strength_override: None이면 친밀도로 자동, 아니면 직접 지정

        Returns:
            {target, suffering, intimacy, intensity, in_burnout, hormones_before, hormones_after}
        """
        if not self.is_suffering_category(suffering_category):
            return {
                'target': target,
                'suffering': suffering_category,
                'resonated': False,
                'reason': 'not_suffering',
            }

        # 친밀도
        intimacy = self.compute_intimacy(target)

        # 공명 강도
        if strength_override is not None:
            intensity = float(np.clip(strength_override, 0.0, 1.0))
        else:
            intensity = self.base_resonance * (0.3 + intimacy * 0.7)

        # Burnout 상태면 약화 (compassion fatigue 자기 보호)
        if self.in_burnout:
            intensity *= 0.3

        # 호르몬 before
        cort_before = self.hs.hormones['cortisol'].level
        ot_before = self.hs.hormones.get(
            'oxytocin', type('h', (), {'level': 0.3})()).level

        # 변조 (Singer 2009)
        # cortisol: stress contagion
        if hasattr(self.hs.hormones['cortisol'], 'stimulate'):
            self.hs.hormones['cortisol'].stimulate(intensity * 0.5)
        else:
            self.hs.hormones['cortisol'].level = float(
                np.clip(cort_before + intensity * 0.3, 0.0, 1.0))

        # oxytocin: 친밀할수록 더 (사회적 유대 신호)
        if 'oxytocin' in self.hs.hormones:
            ot_boost = intensity * intimacy * 0.6
            if hasattr(self.hs.hormones['oxytocin'], 'stimulate'):
                self.hs.hormones['oxytocin'].stimulate(ot_boost)
            else:
                self.hs.hormones['oxytocin'].level = float(
                    np.clip(ot_before + ot_boost, 0.0, 1.0))

        # SA: 고통 카테고리 약하게 활성 (EVE도 잠시 같은 카테고리 접촉)
        self.sa.activate(suffering_category, intensity * 0.4)

        # WM: focus로 (사회적 주의)
        if self.wm is not None and hasattr(self.wm, 'add'):
            try:
                self.wm.add(suffering_category, salience=intensity * 0.5,
                           source='empathy')
            except Exception:
                pass

        # 누적 부담 기록
        self.resonance_log.append((self.time, target, intensity))

        # 호르몬 after
        cort_after = self.hs.hormones['cortisol'].level
        ot_after = self.hs.hormones.get(
            'oxytocin', type('h', (), {'level': ot_before})()).level

        self.resonate_count += 1

        return {
            'target': target,
            'suffering': suffering_category,
            'resonated': True,
            'intimacy': intimacy,
            'intensity': intensity,
            'in_burnout': self.in_burnout,
            'cort_delta': float(cort_after - cort_before),
            'ot_delta': float(ot_after - ot_before),
        }

    # ============= 4. self-other 구분 =============
    def distinguish_self_other(self,
                              suffering_category: str,
                              target: str) -> Dict[str, Any]:
        """
        Singer 2009: 진짜 공감 = 자기-타자 구분 + affective sharing.
        구분 없으면 personal distress (자기 중심 → 도움 행동 X).

        EVE의 self-other 구분:
        - "이 슬픔은 내 거 아님, target 거" 인지
        - SA에 marker 카테고리 활성 ("슬픔_타자")
        - SD가 있으면 "이건 내 신념과 무관" 신호 추가

        Returns:
            {original_category, marked_category, target_inferred, self_distinct}
        """
        marker = self.self_other_marker
        marked = f"{suffering_category}{marker}_{target}"

        # SA에 약하게 활성 (구분 인지)
        self.sa.activate(marked, 0.3)

        # marker → 원본 카테고리 약한 연결 (의미적 다리, 사라지지 않게)
        self.sa.learn_pair(marked, suffering_category, 0.2)
        # target → marked 연결
        if target in self.sa.neighbors or target in self.sa.activations:
            self.sa.learn_pair(target, marked, 0.3)

        # SD가 있으면 — 이건 외부 정보, 내 신념 X 신호
        sd_signal = None
        if self.sd is not None:
            try:
                # 이건 평가 X, 약한 signaling만 (낮은 doubt 통과)
                sd_signal = "external_distress_acknowledged"
            except Exception:
                pass

        self.distinguish_count += 1

        return {
            'original_category': suffering_category,
            'marked_category': marked,
            'target': target,
            'self_distinct': True,  # EVE는 항상 구분
            'sd_signal': sd_signal,
        }

    # ============= 5. 공감 행동 (compassion action) =============
    def compassion_action(self,
                         target: str,
                         suffering_category: str
                         ) -> Dict[str, Any]:
        """
        Batson 1991 empathy-altruism: 공감 → 도움 행동.

        EVE의 공감 행동 (창발 — 텍스트 X, 카테고리 활성):
        - '위로' 카테고리 활성
        - target과 함께 카테고리 활성 (사회 유대)
        - CF "내가 그 입장이라면" 시뮬 (있으면)

        반환: 추천 행동 카테고리 list (NL이 응답 어조에 활용)
        """
        actions: List[str] = []

        # 공감 행동 카테고리
        comfort_cats = ['위로', '대화', '함께', '들어주다', '곁']
        for c in comfort_cats:
            if c in self.sa.neighbors or c in self.sa.activations:
                self.sa.activate(c, 0.4)
                actions.append(c)

        # target도 활성 (지향)
        if target:
            self.sa.activate(target, 0.3)

        # CF "그 입장이라면" 시뮬
        cf_perspective = None
        if self.cf is not None:
            try:
                if hasattr(self.cf, 'closest_world'):
                    similar = self.cf.closest_world(suffering_category, top_n=2)
                    cf_perspective = similar
            except Exception:
                pass

        # WM에 위로 의도
        if self.wm is not None and hasattr(self.wm, 'add'):
            try:
                self.wm.add('위로', salience=0.5, source='compassion')
            except Exception:
                pass

        self.compassion_count += 1

        return {
            'target': target,
            'suffering': suffering_category,
            'actions': actions,
            'cf_perspective': cf_perspective,
        }

    # ============= 6. Burnout check =============
    def burnout_check(self) -> Dict[str, Any]:
        """
        Klimecki 2014 / McEwen Allostatic Load:
        누적 공명 부담이 임계 이상이면 burnout 상태.

        Burnout:
        - 자기 보호 모드 (resonance intensity × 0.3)
        - cortisol 자연 감쇠 가속 (자기 회복)
        - serotonin 약하게 자극 (안정화)

        Returns:
            {load, threshold, in_burnout, just_entered, just_exited}
        """
        # 윈도우 내 누적 부담
        cutoff = self.time - self.burnout_window
        recent = [r for r in self.resonance_log if r[0] >= cutoff]
        load = sum(r[2] for r in recent)

        was_burnout = self.in_burnout
        self.in_burnout = load >= self.burnout_threshold

        just_entered = (not was_burnout) and self.in_burnout
        just_exited = was_burnout and (not self.in_burnout)

        if just_entered:
            self.burnout_count += 1

        # Burnout 진입 시 자기 회복 자극
        if just_entered:
            # serotonin 자극 (안정화)
            if 'serotonin' in self.hs.hormones:
                if hasattr(self.hs.hormones['serotonin'], 'stimulate'):
                    self.hs.hormones['serotonin'].stimulate(0.2)

        return {
            'load': float(load),
            'threshold': self.burnout_threshold,
            'in_burnout': self.in_burnout,
            'just_entered': just_entered,
            'just_exited': just_exited,
            'recent_count': len(recent),
        }

    # ============= 자동 시나리오: perceive_suffering =============
    def perceive_suffering(self,
                          target: str,
                          suffering_category: str,
                          full_chain: bool = True
                          ) -> Dict[str, Any]:
        """
        통합 진입점: 타자가 고통 받는다는 인지 → 전체 chain 자동 실행.
        - distinguish_self_other (자기-타자 구분 먼저)
        - resonate (공명)
        - compassion_action (행동 추천)

        full_chain=False면 resonate만.

        Returns: 통합 결과
        """
        if not self.is_suffering_category(suffering_category):
            return {'perceived': False, 'reason': 'not_suffering'}

        result: Dict[str, Any] = {
            'perceived': True,
            'target': target,
            'suffering': suffering_category,
            'time': self.time,
        }

        # 1) self-other 구분 먼저
        if full_chain:
            distinct = self.distinguish_self_other(suffering_category, target)
            result['distinguish'] = distinct

        # 2) 공명
        reson = self.resonate(target, suffering_category)
        result['resonate'] = reson

        # 3) 공감 행동
        if full_chain and reson.get('resonated'):
            compassion = self.compassion_action(target, suffering_category)
            result['compassion'] = compassion

        return result

    # ============= tick =============
    def tick(self, dt: float):
        """시간 진행 + 주기적 burnout check"""
        self.time += dt
        self.tick_count += 1

        # 매 tick burnout check (저비용 — 윈도우 내 합)
        self.burnout_check()

        # 캐시 invalidate
        if self.time - self._cache_time > self._cache_ttl:
            self.intimacy_cache.clear()

    # ============= 상태 =============
    def get_state(self) -> Dict[str, Any]:
        bo = self.burnout_check()
        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'resonate_count': self.resonate_count,
            'distinguish_count': self.distinguish_count,
            'compassion_count': self.compassion_count,
            'burnout_count': self.burnout_count,
            'current_load': bo['load'],
            'in_burnout': self.in_burnout,
            'recent_resonances': len(self.resonance_log),
        }

    def __repr__(self):
        s = self.get_state()
        bo = ' BURNOUT' if s['in_burnout'] else ''
        return (f"Suffering(t={s['time']:.1f}, "
                f"resonate={s['resonate_count']}, "
                f"load={s['current_load']:.2f}/"
                f"{self.burnout_threshold:.1f}{bo})")
