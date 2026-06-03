"""
EVE v35 - E3: APCLearner (Active Predictive Coding)
====================================================

목표: 매 tick 자동 학습 → 정체 없는 평생 학습 (continual lifelong learning).

기존 학습 정체 문제:
- learn_text/learn_corpus는 외부 say()/learn_X() 호출이 와야 학습.
- EVE가 자발적으로 살아있는 동안에는 학습 X.
- 결과: 외부 자극 없으면 그래프 정체.

해결: 매 tick에
1. 현재 활성 카테고리 → 다음 활성 예측 (predict_next)
2. 다음 tick에서 실제 관측
3. 예측 vs 실제 → 학습 신호:
   - HIT (예측 맞음): Hebbian 강화 + DA↑
   - SURPRISE (예측 못한 카테고리 활성): 새 패턴 학습 + NE↑
   - MISS (예측은 했는데 안 옴): 학습 X (안전성 우선, 약화는 SA.forget이 담당)

EVE 절대 원칙 부합:
- 결정론적 (random/sample/probability X)
- SA 가중치 직접 조작 X — sa.learn_pair만 사용
- 호르몬 자연 변조 (Schultz 1998 RPE 영감)
- 카테고리 의미 단위

학계 기반:
- Active Predictive Coding (Rao 2024 MIT Press)
- Hybrid Predictive Coding (Tschantz 2023)
- Friston 2010 Free Energy Principle
- Schultz 1998 — Reward Prediction Error → DA
- Hasselmo 1995 — ACh = 학습률 변조
- Yu & Dayan 2005 — NE = 놀람/주의 변조

핵심 함수:
1. predict_next_state(current_active, focus) → Set[str]
2. observe_and_learn(observed_active, focus)  → dict (hit/miss/surprise)
3. tick(dt)                                    → observe_and_learn 자동 호출
4. stats()                                     → 누적 학습 통계

Author: 김민석 + Claude v35
"""

from typing import Dict, Set, Optional, Any
from collections import deque


class APCLearner:
    """
    Active Predictive Coding Learner.

    매 tick에 SA의 다음 활성 패턴을 예측하고, 실제 관측 시 비교 → 학습.

    의존성:
        - SpreadingActivation (필수, get_active/learn_pair/neighbors/weights)
        - HormoneSystem (선택, 호르몬 변조)
        - WorkingMemory (선택, focus 기반 예측)

    설계:
        - sa.weights 직접 수정 X — sa.learn_pair만 사용 (회귀 안전)
        - miss는 약화 X (Bjork 보호 — SA.forget이 담당)
        - 호르몬 변조: DA(hit), NE(surprise), Cort(predominant miss)
    """

    def __init__(self,
                 spreading_activation,
                 hormone_system=None,
                 working_memory=None,
                 lr_base: float = 0.05,
                 lr_surprise_bonus: float = 0.5,
                 predict_threshold_focus: float = 0.3,
                 predict_threshold_other: float = 0.4,
                 max_predictions: int = 30,
                 use_hormone_modulation: bool = True,
                 enable_when_quiet: bool = True):
        """
        Args:
            spreading_activation: SA 인스턴스
            hormone_system: HS 인스턴스 (선택)
            working_memory: WM 인스턴스 (선택, focus 기반 예측 강화)
            lr_base: HIT 시 기본 학습률
            lr_surprise_bonus: SURPRISE 학습 시 추가 배수 (기본 1.5x)
            predict_threshold_focus: focus 이웃을 예측에 포함할 가중치 임계
            predict_threshold_other: 다른 활성의 이웃 임계 (더 엄격)
            max_predictions: 한 tick 예측 최대 카테고리 수 (폭발 방지)
            use_hormone_modulation: 호르몬 변조 사용 여부
            enable_when_quiet: True면 활성 0개여도 tick 학습 X (조용한 tick)
        """
        self.sa = spreading_activation
        self.hs = hormone_system
        self.wm = working_memory

        self.lr_base = float(lr_base)
        self.lr_surprise_bonus = float(lr_surprise_bonus)
        self.predict_threshold_focus = float(predict_threshold_focus)
        self.predict_threshold_other = float(predict_threshold_other)
        self.max_predictions = int(max_predictions)
        self.use_hormone_modulation = bool(use_hormone_modulation)
        self.enable_when_quiet = bool(enable_when_quiet)

        # 예측 버퍼 (전 tick의 예측 + 활성)
        self.last_prediction: Set[str] = set()
        self.last_focus: Optional[str] = None
        self.last_active: Set[str] = set()
        self._initialized: bool = False  # 첫 tick 체크 (빈 set과 구분)

        # 누적 통계
        self.observe_count = 0
        self.predict_count = 0
        self.skipped_quiet = 0  # 활성 0개라 skip한 tick
        self.hits_total = 0
        self.misses_total = 0
        self.surprises_total = 0
        self.pairs_learned = 0  # learn_pair 호출 수

        # 최근 결과 (디버그)
        self.recent_results: deque = deque(maxlen=20)

    # ============= 1. 다음 상태 예측 =============
    def predict_next_state(self,
                          current_active: Set[str],
                          focus: Optional[str] = None) -> Set[str]:
        """
        현재 활성 → 다음 활성 카테고리 예측.

        결정론적 알고리즘:
        1. focus의 이웃 中 weight ≥ predict_threshold_focus → 예측
        2. 다른 활성의 이웃 中 weight ≥ predict_threshold_other → 예측
        3. 자기 자신 (current_active) 제외
        4. max_predictions 초과 시 weight 높은 순 결정론적 자르기

        Args:
            current_active: 현재 활성 카테고리
            focus: WM의 현재 focus (있으면 우선)

        Returns:
            예측 카테고리 set
        """
        scored: Dict[str, float] = {}

        # 1) focus 이웃 (낮은 임계 = 더 많이 예측)
        if focus and focus in self.sa.neighbors:
            for neighbor in self.sa.neighbors[focus]:
                if neighbor in current_active:
                    continue
                w = self.sa.get_weight(focus, neighbor)
                if w >= self.predict_threshold_focus:
                    # focus 이웃에 가중치 부스트 (1.2x)
                    scored[neighbor] = max(scored.get(neighbor, 0), w * 1.2)

        # 2) 다른 활성의 이웃 (높은 임계 = 보수적)
        for cat in current_active:
            if cat == focus:
                continue
            if cat not in self.sa.neighbors:
                continue
            for neighbor in self.sa.neighbors[cat]:
                if neighbor in current_active:
                    continue
                w = self.sa.get_weight(cat, neighbor)
                if w >= self.predict_threshold_other:
                    scored[neighbor] = max(scored.get(neighbor, 0), w)

        # 3) max_predictions 자르기 (결정론: weight ↓, 알파벳 ↑)
        items = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
        items = items[:self.max_predictions]

        self.predict_count += 1
        return {cat for cat, _ in items}

    # ============= 2. 관측 + 학습 =============
    def observe_and_learn(self,
                         observed_active: Set[str],
                         focus: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        실제 활성 관측 → 전 tick 예측과 비교 → 학습.

        분류:
        - HIT      = last_prediction ∩ observed (예측 맞음)
        - MISS     = last_prediction - observed  (예측 했는데 안 옴, 학습 X)
        - SURPRISE = (observed - last_active) - last_prediction
                     (새로 활성됐는데 예측 못함 → 새 패턴 학습 신호)

        Returns:
            결과 dict 또는 None (첫 tick이라 비교 X)
        """
        # 첫 tick — 비교 불가, 다음을 위해 예측만 저장
        if not self._initialized:
            self.last_active = set(observed_active)
            self.last_focus = focus
            self.last_prediction = self.predict_next_state(observed_active, focus)
            self._initialized = True
            return None

        # 활성 0개 (조용한 EVE) — 학습 skip
        if self.enable_when_quiet and len(observed_active) == 0 and len(self.last_active) == 0:
            self.skipped_quiet += 1
            self.last_focus = focus
            return None

        # 분류
        observed_new = observed_active - self.last_active  # 새로 활성된 것
        hit = self.last_prediction & observed_active
        miss = self.last_prediction - observed_active
        surprise = observed_new - self.last_prediction

        # 학습률 (호르몬 변조)
        lr_hit = self.lr_base
        lr_surprise = self.lr_base * (1.0 + self.lr_surprise_bonus)

        if self.use_hormone_modulation and self.hs is not None:
            ach = self.hs.hormones['acetylcholine'].level
            ne = self.hs.hormones['norepinephrine'].level
            # ACh ↑ → 학습률 ↑ (Hasselmo 1995)
            lr_hit *= (0.5 + ach)
            lr_surprise *= (0.5 + ach)
            # NE ↑ → 놀람 학습 강화 (Yu & Dayan 2005)
            lr_surprise *= (1.0 + ne * 0.5)

        # ============= 학습: HIT 강화 =============
        # 결정론적 정렬로 페어 생성
        anchor = self.last_focus if self.last_focus else None
        if not anchor and self.last_active:
            anchor = sorted(self.last_active)[0]  # 결정론적 fallback

        if anchor:
            for cat in sorted(hit):
                if cat == anchor:
                    continue
                self.sa.learn_pair(anchor, cat, lr_hit)
                self.pairs_learned += 1

            # ============= 학습: SURPRISE 강화 (더 강하게) =============
            for cat in sorted(surprise):
                if cat == anchor:
                    continue
                self.sa.learn_pair(anchor, cat, lr_surprise)
                self.pairs_learned += 1

        # ============= 호르몬 신호 =============
        if self.use_hormone_modulation and self.hs is not None:
            # HIT → DA (Schultz RPE +)
            if hit and len(self.last_prediction) > 0:
                hit_ratio = len(hit) / len(self.last_prediction)
                self.hs.hormones['dopamine'].stimulate(0.05 * hit_ratio)

            # SURPRISE → NE (놀람)
            if surprise:
                # 놀람 강도: surprise 수에 비례 (max 1.0 cap)
                surprise_strength = min(1.0, len(surprise) / 5.0)
                self.hs.hormones['norepinephrine'].stimulate(0.05 * surprise_strength)

            # MISS 우세 → 약한 Cort (예측 실패)
            if len(miss) > 0 and len(miss) > len(hit) * 2:
                self.hs.hormones['cortisol'].stimulate(0.02)

            # 활발한 학습 → ACh (Hasselmo)
            if hit or surprise:
                self.hs.hormones['acetylcholine'].stimulate(0.03)

        # ============= 다음 예측 =============
        next_prediction = self.predict_next_state(observed_active, focus)

        # 카운터 갱신
        self.observe_count += 1
        self.hits_total += len(hit)
        self.misses_total += len(miss)
        self.surprises_total += len(surprise)

        result = {
            'hit': len(hit),
            'miss': len(miss),
            'surprise': len(surprise),
            'hit_cats': hit,
            'surprise_cats': surprise,
            'predicted_size': len(self.last_prediction),
            'observed_size': len(observed_active),
            'next_prediction_size': len(next_prediction),
            'lr_hit': lr_hit,
            'lr_surprise': lr_surprise,
        }
        self.recent_results.append(result)

        # 버퍼 갱신
        self.last_active = set(observed_active)
        self.last_focus = focus
        self.last_prediction = next_prediction

        return result

    # ============= 3. tick 자동 학습 =============
    def tick(self, dt: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        매 tick 호출 — 평생 학습 핵심.
        EVE main tick의 가장 마지막 (모든 모듈 정리 후) 호출 권장.
        """
        current_active = self.sa.get_active()
        current_focus = self.wm.get_focus() if self.wm else None
        return self.observe_and_learn(current_active, current_focus)

    # ============= 4. 통계 =============
    def stats(self) -> Dict[str, Any]:
        """학습 누적 통계"""
        decided = self.hits_total + self.misses_total
        accuracy = self.hits_total / max(1, decided)

        # 최근 정확도 (최근 20 결과)
        recent_hits = sum(r['hit'] for r in self.recent_results)
        recent_decided = sum(r['hit'] + r['miss'] for r in self.recent_results)
        recent_accuracy = recent_hits / max(1, recent_decided)

        return {
            'observe_count': self.observe_count,
            'predict_count': self.predict_count,
            'skipped_quiet': self.skipped_quiet,
            'hits_total': self.hits_total,
            'misses_total': self.misses_total,
            'surprises_total': self.surprises_total,
            'pairs_learned': self.pairs_learned,
            'accuracy_overall': accuracy,
            'accuracy_recent': recent_accuracy,
            'last_prediction_size': len(self.last_prediction),
        }

    def reset(self):
        """예측 버퍼 + 통계 리셋 (학습 가중치는 유지)"""
        self.last_prediction = set()
        self.last_focus = None
        self.last_active = set()
        self._initialized = False
        self.observe_count = 0
        self.predict_count = 0
        self.skipped_quiet = 0
        self.hits_total = 0
        self.misses_total = 0
        self.surprises_total = 0
        self.pairs_learned = 0
        self.recent_results.clear()

    def __repr__(self):
        s = self.stats()
        return (f"APCLearner(observed={s['observe_count']}, "
                f"hits={s['hits_total']}, "
                f"surprises={s['surprises_total']}, "
                f"acc={s['accuracy_overall']:.2f})")
