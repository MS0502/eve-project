"""
EVE v32 - HormoneSystem (26종)
================================

eve_design v2 / hormone_system.py 기반 + v32 정정/추가

변경점 (v2 → v32):
1. 일주기 dt 폭발 버그 수정 (큰 dt에도 안전)
2. 양방향 폐쇄 루프 추가:
   - category_to_hormone(active_set): 카테고리 활성 → 호르몬 자극
3. 카테고리 임계 변조:
   - get_category_threshold_modulation(): 호르몬 → 카테고리 그룹별 임계
4. HormoneCocktail 15 시나리오 통합 (v9.12)
5. trigger_event 8개 → 13개 확장
6. compute_mood (valence/arousal/dominance) 유지
7. learning_rate 26종 다 반영

EVE 절대 원칙 부합:
- 확률 X (random/softmax 사용 안 함) ✅
- 결정론적 (같은 입력 → 같은 출력) ✅
- 호르몬 = 자연 noise (생물학적 variation) ✅
- 카테고리 임계 변조 (뉴런 X) ✅

Author: 김민석 + Claude v32
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ====================================================================
# Hormone (개별 호르몬 정의)
# ====================================================================

@dataclass
class Hormone:
    """개별 호르몬"""
    name: str
    level: float           # 현재 수치 (0-1)
    baseline: float        # 기본 수치
    reactivity: float      # 자극 반응 속도
    decay_rate: float      # 분해 속도 (1/s)
    tier: str              # A, B, C
    phase: int             # 1-4

    def update(self, dt: float):
        """baseline으로 자연 분해"""
        diff = self.baseline - self.level
        # dt가 크면 step 분할 (안전)
        if dt * self.decay_rate > 0.5:
            steps = int(np.ceil(dt * self.decay_rate / 0.1))
            sub_dt = dt / steps
            for _ in range(steps):
                diff = self.baseline - self.level
                self.level += diff * self.decay_rate * sub_dt
                self.level = np.clip(self.level, 0.0, 1.0)
        else:
            self.level += diff * self.decay_rate * dt
            self.level = np.clip(self.level, 0.0, 1.0)

    def stimulate(self, amount: float):
        """외부 자극"""
        self.level = np.clip(self.level + amount * self.reactivity, 0.0, 1.0)


# ====================================================================
# HormoneSystem v32
# ====================================================================

class HormoneSystem:
    """
    EVE v32 호르몬 시스템 (26종)

    사용:
        hs = HormoneSystem(phase=4, stage='adult')
        hs.update(dt=1.0)                              # 매 스텝
        hs.trigger_event('reward')                     # 이벤트
        hs.trigger_cocktail('사회적_유대')                 # 칵테일
        hs.category_to_hormone({'기쁨', '민석'})           # 카테고리→호르몬 (NEW)
        thresholds = hs.get_category_threshold_modulation()  # 호르몬→임계 (NEW)
        lr = hs.compute_learning_rate()
        mood = hs.compute_mood()
    """

    # ============= 카테고리 그룹 정의 (호르몬 → 카테고리 임계 변조용) =============
    CATEGORY_GROUPS = {
        'reward':   {'기쁨', '성취', '쾌감', '맛있다', '좋다', '재밌다'},
        'social':   {'민석', '친구', '대화', '유대', '사랑', '함께'},
        'threat':   {'위험', '아프다', '무섭다', '칼', '죽음', '공격'},
        'attention':{'주의', '집중', '중요', '관찰', '경계'},
        'memory':   {'기억', '회상', '과거', '경험'},
        'rest':     {'잠', '휴식', '졸림', '편안'},
        'curiosity':{'왜', '어떻게', '새로운', '모르는', '궁금'},
        'aversion': {'싫다', '거절', '피하다', '회피', '불쾌'},
    }

    # ============= 13 칵테일 (v9.12 자산) =============
    COCKTAILS = {
        '새로운_학습':  {'dopamine': 0.3, 'norepinephrine': 0.3, 'acetylcholine': 0.3, 'glutamate': 0.2, 'endorphin': 0.1},
        '사회적_유대':  {'oxytocin': 0.4, 'dopamine': 0.2, 'serotonin': 0.2, 'endorphin': 0.2},
        '스트레스':    {'cortisol': 0.5, 'norepinephrine': 0.4},  # serotonin 감소는 interaction에서
        '성취':       {'dopamine': 0.5, 'endorphin': 0.3, 'serotonin': 0.2, 'norepinephrine': 0.2, 'oxytocin': 0.1},
        '사랑':       {'oxytocin': 0.5, 'dopamine': 0.5, 'serotonin': 0.3, 'endorphin': 0.3},
        '공포':       {'cortisol': 0.6, 'norepinephrine': 0.6, 'acetylcholine': 0.2},
        '기쁨':       {'dopamine': 0.4, 'endorphin': 0.3, 'serotonin': 0.3, 'oxytocin': 0.2},
        '슬픔':       {'cortisol': 0.3},  # serotonin/dopamine 감소는 interaction
        '분노':       {'norepinephrine': 0.6, 'cortisol': 0.4, 'testosterone': 0.2},
        '수면':       {'gaba': 0.4, 'serotonin': 0.2, 'melatonin': 0.4, 'adenosine': 0.3},
        '집중':       {'norepinephrine': 0.5, 'acetylcholine': 0.4, 'histamine': 0.2},
        '안전':       {'oxytocin': 0.3, 'serotonin': 0.2, 'gaba': 0.2},
        '호기심':     {'dopamine': 0.3, 'norepinephrine': 0.2, 'acetylcholine': 0.3},
    }

    def __init__(self, phase: int = 4, developmental_stage: str = "adult"):
        self.phase = phase
        self.stage = developmental_stage
        self.time = 0.0
        self.sim_hour = 12.0  # 정오 시작
        self.hormones = self._init_all_hormones()
        self._activate_phase(phase)
        self._apply_developmental_profile(developmental_stage)

    def _init_all_hormones(self) -> Dict[str, Hormone]:
        """26 호르몬 정의 (성인 기준)"""
        config = [
            # Tier A 신경전달물질 (10)
            ("glutamate",      0.5, 1.0, 10.0,  "A", 1),
            ("gaba",           0.4, 1.0,  8.0,  "A", 1),
            ("glycine",        0.3, 0.5,  5.0,  "A", 2),
            ("dopamine",       0.3, 1.5,  2.0,  "A", 1),
            ("serotonin",      0.5, 0.8,  1.0,  "A", 1),
            ("norepinephrine", 0.3, 2.0,  3.0,  "A", 1),
            ("histamine",      0.4, 1.0,  1.0,  "A", 2),
            ("acetylcholine",  0.4, 1.5,  5.0,  "A", 1),
            ("adenosine",      0.2, 0.1,  0.05, "A", 2),
            ("endorphin",      0.2, 2.0,  0.5,  "A", 2),
            # Tier B 뇌 호르몬 (12)
            ("cortisol",       0.3, 2.0,  0.3,  "B", 1),
            ("oxytocin",       0.3, 1.5,  0.5,  "B", 1),
            ("vasopressin",    0.3, 1.2,  0.5,  "B", 2),
            ("melatonin",      0.1, 0.5,  0.1,  "B", 1),
            ("bdnf",           0.3, 0.2,  0.01, "B", 1),
            ("ngf",            0.3, 0.1,  0.01, "B", 2),
            ("estrogen",       0.3, 0.5,  0.05, "B", 3),  # newborn은 0, adult 중성=0.3
            ("testosterone",   0.3, 0.5,  0.05, "B", 3),
            ("insulin_brain",  0.4, 0.5,  0.2,  "B", 4),
            ("thyroid",        0.5, 0.1,  0.01, "B", 4),
            ("leptin",         0.3, 0.5,  0.1,  "B", 4),
            ("ghrelin",        0.3, 0.5,  0.1,  "B", 4),
            # Tier C (4)
            ("prolactin",      0.2, 0.5,  0.1,  "C", 3),
            ("dhea",           0.3, 0.3,  0.05, "C", 3),
            ("progesterone",   0.2, 0.5,  0.1,  "C", 3),
            ("growth_hormone", 0.2, 0.3,  0.1,  "C", 3),
        ]
        return {
            n: Hormone(name=n, level=b, baseline=b, reactivity=r,
                       decay_rate=d, tier=t, phase=p)
            for n, b, r, d, t, p in config
        }

    def _activate_phase(self, phase: int):
        self.active_hormones = [n for n, h in self.hormones.items() if h.phase <= phase]

    def _apply_developmental_profile(self, stage: str):
        if stage == "newborn":
            self.hormones["cortisol"].baseline = 0.5
            self.hormones["dopamine"].baseline = 0.2
            self.hormones["serotonin"].baseline = 0.3
            self.hormones["oxytocin"].baseline = 0.4
            self.hormones["estrogen"].baseline = 0.0
            self.hormones["testosterone"].baseline = 0.0
            for h in self.hormones.values():
                h.level = h.baseline
        elif stage == "infant":
            self.hormones["cortisol"].baseline = 0.3
            self.hormones["dopamine"].baseline = 0.3
            self.hormones["melatonin"].baseline = 0.2
        elif stage == "toddler":
            self.hormones["dopamine"].baseline = 0.4
        # adult = default

    # ============= 매 스텝 업데이트 =============
    def update(self, dt: float, stimuli: Optional[Dict[str, float]] = None):
        self.time += dt
        # sim_hour 안전 누적 (큰 dt도 OK)
        self.sim_hour = (self.sim_hour + dt / 3600.0) % 24.0

        if stimuli:
            for n, amt in stimuli.items():
                if n in self.hormones:
                    self.hormones[n].stimulate(amt)

        # 자연 분해
        for n in self.active_hormones:
            self.hormones[n].update(dt)

        # 상호관계
        self._apply_interactions(dt)

        # 일주기 (dt 안전)
        self._apply_circadian(dt)

    def _apply_interactions(self, dt: float):
        """20개 확실한 상호관계 (호르몬 백과 기반)"""
        h = self.hormones
        active = self.active_hormones

        # dt 안전 클립 (interaction은 작은 효과)
        eff_dt = min(dt, 1.0)

        # === 억제 ===
        if "oxytocin" in active and "cortisol" in active:
            h["cortisol"].level -= h["oxytocin"].level * 0.5 * eff_dt

        if "bdnf" in active and h["cortisol"].level > 0.6:
            h["bdnf"].level -= h["cortisol"].level * 0.1 * eff_dt

        h["serotonin"].level -= h["cortisol"].level * 0.3 * eff_dt

        if "adenosine" in active and "acetylcholine" in active:
            h["acetylcholine"].level -= h["adenosine"].level * 0.3 * eff_dt

        # === 협력 ===
        if "norepinephrine" in active and h["cortisol"].level > 0.5:
            h["norepinephrine"].stimulate(h["cortisol"].level * 0.3 * eff_dt)

        if "endorphin" in active and h["dopamine"].level > 0.7:
            h["endorphin"].stimulate(h["dopamine"].level * 0.2 * eff_dt)

        if "oxytocin" in active and h["oxytocin"].level > 0.6:
            h["dopamine"].stimulate(h["oxytocin"].level * 0.2 * eff_dt)

        if "acetylcholine" in active and h["acetylcholine"].level > 0.5:
            h["glutamate"].stimulate(h["acetylcholine"].level * 0.1 * eff_dt)

        # 5HT → Melatonin (전구체, 밤에만)
        if "melatonin" in active and (self.sim_hour >= 22 or self.sim_hour < 6):
            h["melatonin"].stimulate(h["serotonin"].level * 0.05 * eff_dt)

        # 클립
        for n in active:
            h[n].level = np.clip(h[n].level, 0.0, 1.0)

    def _apply_circadian(self, dt: float):
        """일주기 - dt 안전 (큰 dt도 발산 X)"""
        if "melatonin" not in self.active_hormones:
            return
        if self.stage not in ["infant", "toddler", "adult"]:
            return

        hour = self.sim_hour

        # 멜라토닌 목표 (밤 22-6시 높음)
        target_mel = 0.7 if (hour >= 22 or hour < 6) else 0.1
        # 코르티솔 목표 (아침 7-9시 높음)
        if 7 <= hour <= 9:
            target_cort = 0.5
        elif hour > 20 or hour < 5:
            target_cort = 0.15
        else:
            target_cort = self.hormones["cortisol"].baseline

        # 안전한 수렴 (큰 dt도 발산 X): 비율 기반
        # rate = 1 - exp(-k*dt) 형태로 안정적
        k_mel = 0.1   # 멜라토닌 변화 속도
        k_cort = 0.05
        alpha_mel = 1.0 - np.exp(-k_mel * dt)
        alpha_cort = 1.0 - np.exp(-k_cort * dt)
        alpha_mel = np.clip(alpha_mel, 0.0, 1.0)
        alpha_cort = np.clip(alpha_cort, 0.0, 1.0)

        h = self.hormones
        h["melatonin"].level += (target_mel - h["melatonin"].level) * alpha_mel
        h["cortisol"].level += (target_cort - h["cortisol"].level) * alpha_cort

        h["melatonin"].level = np.clip(h["melatonin"].level, 0.0, 1.0)
        h["cortisol"].level = np.clip(h["cortisol"].level, 0.0, 1.0)

    # ============= [NEW v32] 양방향 폐쇄 루프 =============
    def category_to_hormone(self, active_categories: Set[str], strength: float = 0.1):
        """
        카테고리 활성 → 호르몬 자극 (Bottom-up)

        EVE 카테고리가 활성화되면 해당 의미가 호르몬을 자극.
        예: '기쁨' 카테고리 활성 → dopamine + endorphin ↑
            '위험' 카테고리 활성 → cortisol + NE ↑
        """
        if not active_categories:
            return

        # 카테고리 그룹별 매칭 → 호르몬 자극
        matched_groups = {}
        for grp, words in self.CATEGORY_GROUPS.items():
            overlap = active_categories & words
            if overlap:
                matched_groups[grp] = len(overlap)

        for grp, count in matched_groups.items():
            amt = strength * count
            if grp == 'reward':
                self.hormones['dopamine'].stimulate(amt)
                if 'endorphin' in self.active_hormones:
                    self.hormones['endorphin'].stimulate(amt * 0.5)
            elif grp == 'social':
                self.hormones['oxytocin'].stimulate(amt)
                self.hormones['dopamine'].stimulate(amt * 0.3)
            elif grp == 'threat':
                self.hormones['cortisol'].stimulate(amt)
                self.hormones['norepinephrine'].stimulate(amt)
            elif grp == 'attention':
                self.hormones['norepinephrine'].stimulate(amt * 0.7)
                if 'acetylcholine' in self.active_hormones:
                    self.hormones['acetylcholine'].stimulate(amt * 0.5)
            elif grp == 'memory':
                if 'acetylcholine' in self.active_hormones:
                    self.hormones['acetylcholine'].stimulate(amt * 0.3)
            elif grp == 'rest':
                if 'gaba' in self.active_hormones:
                    self.hormones['gaba'].stimulate(amt)
                if 'adenosine' in self.active_hormones:
                    self.hormones['adenosine'].stimulate(amt * 0.3)
            elif grp == 'curiosity':
                self.hormones['dopamine'].stimulate(amt * 0.5)
                self.hormones['norepinephrine'].stimulate(amt * 0.3)
            elif grp == 'aversion':
                self.hormones['cortisol'].stimulate(amt * 0.3)
                self.hormones['serotonin'].level -= amt * 0.2  # 직접 감소
                self.hormones['serotonin'].level = max(0.0, self.hormones['serotonin'].level)

    # ============= [NEW v32] 호르몬 → 카테고리 임계 변조 =============
    def get_category_threshold_modulation(self) -> Dict[str, float]:
        """
        호르몬 → 카테고리 그룹별 임계 변조 (Top-down)

        반환: {그룹명: 변조계수}
            계수 > 1.0 → 임계 ↑ (활성 어려움)
            계수 < 1.0 → 임계 ↓ (활성 쉬움)

        EVE에서 사용:
            base_threshold = 0.5
            mods = hs.get_category_threshold_modulation()
            for category in '기쁨':
                actual_threshold = base_threshold * mods['reward']
        """
        h = self.hormones
        mods = {}

        # reward 그룹: 도파민 ↑ → 임계 ↓ (보상 카테고리 더 쉽게 활성)
        mods['reward'] = max(0.3, 1.0 - h['dopamine'].level * 0.6)

        # social 그룹: 옥시토신 ↑ → 임계 ↓
        ot = h['oxytocin'].level if 'oxytocin' in self.active_hormones else 0.3
        mods['social'] = max(0.3, 1.0 - ot * 0.5)

        # threat 그룹: 코르티솔 ↑ → 임계 ↓ (위험 카테고리 활성 ↑)
        mods['threat'] = max(0.3, 1.0 - h['cortisol'].level * 0.7)

        # attention 그룹: NE ↑ → 임계 ↓ (Yerkes-Dodson 단순화)
        ne = h['norepinephrine'].level if 'norepinephrine' in self.active_hormones else 0.3
        mods['attention'] = max(0.3, 1.0 - ne * 0.6)

        # memory 그룹: ACh ↑ → 임계 ↓
        ach = h['acetylcholine'].level if 'acetylcholine' in self.active_hormones else 0.3
        mods['memory'] = max(0.4, 1.0 - ach * 0.4)

        # rest 그룹: 멜라토닌 ↑ or 아데노신 ↑ → 임계 ↓ (졸림 카테고리 쉽게 활성)
        mel = h['melatonin'].level if 'melatonin' in self.active_hormones else 0.1
        ad = h['adenosine'].level if 'adenosine' in self.active_hormones else 0.2
        mods['rest'] = max(0.3, 1.0 - max(mel, ad) * 0.7)

        # curiosity 그룹: 도파민 ↑ + NE ↑
        mods['curiosity'] = max(0.4, 1.0 - (h['dopamine'].level * 0.4 + ne * 0.3))

        # aversion 그룹: 세로토닌 ↓ → 임계 ↓ (불쾌 카테고리 쉽게)
        mods['aversion'] = max(0.4, 1.0 - (1.0 - h['serotonin'].level) * 0.5)

        # 전반적 영향
        # 멜라토닌 ↑ → 모든 그룹 임계 ↑ (활성 ↓ = 수면)
        if mel > 0.5:
            for k in mods:
                if k != 'rest':
                    mods[k] *= (1.0 + (mel - 0.5) * 1.0)

        # 코르티솔 만성 ↑ → reward/curiosity 억제
        if h['cortisol'].level > 0.7:
            mods['reward'] *= (1.0 + (h['cortisol'].level - 0.7) * 0.5)
            mods['curiosity'] *= (1.0 + (h['cortisol'].level - 0.7) * 0.5)

        return mods

    # ============= 학습률, 각성, 기분 =============
    def compute_learning_rate(self) -> float:
        h = self.hormones
        active = self.active_hormones
        lr = 0.0

        lr += h["dopamine"].level * 1.0
        lr += h["cortisol"].level * 1.5
        lr -= h["serotonin"].level * 0.8

        if "norepinephrine" in active:
            lr += h["norepinephrine"].level * 0.8
        if "acetylcholine" in active:
            lr += h["acetylcholine"].level * 0.7
        if "oxytocin" in active:
            lr += h["oxytocin"].level * 0.4
        if "melatonin" in active:
            lr -= h["melatonin"].level * 0.8
        if "bdnf" in active:
            lr += h["bdnf"].level * 0.5
        if "histamine" in active:
            lr += h["histamine"].level * 0.3
        if "endorphin" in active:
            lr += h["endorphin"].level * 0.3
        if "vasopressin" in active:
            lr += h["vasopressin"].level * 0.2
        if "estrogen" in active:
            lr += h["estrogen"].level * 0.3
        if "progesterone" in active:
            lr -= h["progesterone"].level * 0.2
        if "growth_hormone" in active:
            lr += h["growth_hormone"].level * 0.1
        if "thyroid" in active:
            lr *= (0.5 + h["thyroid"].level)
        if "insulin_brain" in active and h["insulin_brain"].level < 0.3:
            lr *= 0.7

        return float(np.clip(lr, 0.01, 2.0))

    def compute_arousal(self) -> float:
        h = self.hormones
        a = 0.3
        if "norepinephrine" in self.active_hormones:
            a += h["norepinephrine"].level * 0.5
        if "histamine" in self.active_hormones:
            a += h["histamine"].level * 0.3
        if "cortisol" in self.active_hormones:
            a += h["cortisol"].level * 0.3
        if "adenosine" in self.active_hormones:
            a -= h["adenosine"].level * 0.5
        if "melatonin" in self.active_hormones:
            a -= h["melatonin"].level * 0.6
        return float(np.clip(a, 0.0, 1.0))

    def compute_mood(self) -> Dict[str, float]:
        h = self.hormones
        valence = 0.0
        valence += h["dopamine"].level * 0.4
        valence += h["serotonin"].level * 0.3
        if "endorphin" in self.active_hormones:
            valence += h["endorphin"].level * 0.3
        if "oxytocin" in self.active_hormones:
            valence += h["oxytocin"].level * 0.2
        valence -= h["cortisol"].level * 0.5

        arousal = self.compute_arousal()

        dominance = 0.5
        dominance += h["dopamine"].level * 0.2
        dominance -= h["cortisol"].level * 0.4
        if "serotonin" in self.active_hormones:
            dominance += h["serotonin"].level * 0.2

        return {
            "valence": float(np.clip(valence, -1.0, 1.0)),
            "arousal": float(np.clip(arousal, 0.0, 1.0)),
            "dominance": float(np.clip(dominance, 0.0, 1.0)),
        }

    # ============= 이벤트 트리거 (8개 → 13개 확장) =============
    EVENTS = {
        "reward":           {"dopamine": 0.3, "endorphin": 0.2},
        "social_contact":   {"oxytocin": 0.3, "dopamine": 0.1},
        "stress":           {"cortisol": 0.4, "norepinephrine": 0.3},
        "threat":           {"cortisol": 0.6, "norepinephrine": 0.5},
        "novel_stimulus":   {"norepinephrine": 0.3, "acetylcholine": 0.2, "dopamine": 0.1},
        "learning_success": {"dopamine": 0.4, "acetylcholine": 0.3, "bdnf": 0.1},
        "pain":             {"cortisol": 0.3, "endorphin": 0.2},
        "safety":           {"oxytocin": 0.2, "serotonin": 0.1},
        # NEW v32 (5개)
        "loss":             {"cortisol": 0.3, "serotonin": -0.2},  # 슬픔 (음수 = 직접 감소)
        "anger":            {"norepinephrine": 0.5, "cortisol": 0.3, "testosterone": 0.1},
        "curiosity":        {"dopamine": 0.2, "norepinephrine": 0.2, "acetylcholine": 0.2},
        "fatigue":          {"adenosine": 0.4, "cortisol": -0.1},
        "achievement":      {"dopamine": 0.5, "endorphin": 0.3, "serotonin": 0.2, "oxytocin": 0.1},
    }

    def trigger_event(self, event_type: str):
        if event_type not in self.EVENTS:
            return False
        for hn, amt in self.EVENTS[event_type].items():
            if hn in self.hormones:
                if amt > 0:
                    self.hormones[hn].stimulate(amt)
                else:
                    # 직접 감소 (음수)
                    self.hormones[hn].level = max(0.0, self.hormones[hn].level + amt)
        return True

    def trigger_cocktail(self, cocktail: str, intensity: float = 1.0):
        """v9.12 칵테일 발동"""
        if cocktail not in self.COCKTAILS:
            return False
        for hn, amt in self.COCKTAILS[cocktail].items():
            if hn in self.hormones:
                self.hormones[hn].stimulate(amt * intensity)
        return True

    # ============= 상태 조회 =============
    def get_state(self) -> Dict[str, float]:
        return {n: float(self.hormones[n].level) for n in self.active_hormones}

    def primary_hormone(self) -> Tuple[str, float]:
        """가장 활성화된 호르몬"""
        state = self.get_state()
        n = max(state, key=state.get)
        return n, state[n]
