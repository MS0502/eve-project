"""
SafetyAdapter — 라운드 35 (C-3)

GPT 분석 (eve3.md) 기반 4개 안전장치:

1. Confidence — 의도 분류 신뢰도 낮으면 "다시 말해줄래?"
2. Decay — 안 쓰는 학습 자연 약화 (망각)
3. Conflict — 모순 감지 + 해소 ("어제 X 좋아했는데 오늘 X 싫어")
4. Exploration — 같은 응답 반복 시 다른 시도

학계 근거:
- Confidence: Bayesian inference (Friston 2010 Active Inference)
- Decay: Ebbinghaus 1885 forgetting curve, Wixted 2004
- Conflict: Festinger 1957 cognitive dissonance, Bouton 2004 retrieval interference
- Exploration: Sutton & Barto 2018 ε-greedy, Optimism under uncertainty (Kaelbling 1996)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 1. Confidence — 의도 분류 신뢰도 검사
# =============================================================================

CONFIDENCE_THRESHOLD = 0.5   # 이 미만은 "다시 말해줄래?"
CONFIDENCE_HIGH = 0.8        # 이 이상은 자신있게 처리

CONFIDENCE_RESPONSES = [
    "음... 잘 못 알아들었어. 다시 말해줄래?",
    "어... 잘 모르겠어. 다른 말로 해줄 수 있어?",
    "잠깐, 그 말 의미 다시 알려줄래?",
]


# =============================================================================
# 2. Decay — 학습 망각
# =============================================================================

DECAY_RATE_PER_DAY = 0.99    # 매일 1% 약화 (안 쓰면)
DECAY_FLOOR = 0.05           # 이 아래로는 안 떨어짐 (완전 잊지 않기)
USAGE_BOOST = 0.05           # 사용할 때마다 보강


# =============================================================================
# 3. Conflict — 모순 감지
# =============================================================================

@dataclass
class Belief:
    """믿음 — 사용자에 대한 정보 한 조각"""
    subject: str           # 누구 (민석/너)
    predicate: str         # 무엇을 한다 (좋아하다/싫어하다)
    object_: str           # 대상 (단 거/PT)
    polarity: str          # positive / negative
    timestamp: float       # 언제 학습됐나
    confidence: float = 1.0


# 호환되는 술어 (모순 검사용)
PREDICATE_OPPOSITES = {
    "좋아하": "싫어하",
    "싫어하": "좋아하",
    "사랑하": "미워하",
    "미워하": "사랑하",
    "기뻐하": "슬퍼하",
}


# =============================================================================
# 4. Exploration — 응답 다양화
# =============================================================================

EXPLORATION_THRESHOLD = 3        # 같은 응답 N회 반복 시 force_alternative
RECENT_RESPONSE_WINDOW = 10      # 최근 N개 응답 추적


# =============================================================================
# SafetyAdapter
# =============================================================================

class SafetyAdapter:
    """4개 안전장치 통합."""

    def __init__(self, engine=None):
        self.engine = engine
        
        # Confidence 통계
        self.confidence_checks = 0
        self.confidence_low_count = 0
        
        # Decay 통계
        self.last_decay_run = time.time()
        self.decay_runs = 0
        
        # Conflict
        self.beliefs: list = []           # Belief 리스트
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        # Exploration
        self.recent_responses = deque(maxlen=RECENT_RESPONSE_WINDOW)
        self.exploration_triggers = 0

    # ===== 1. Confidence =====
    
    def check_confidence(self, intent: str, confidence: float) -> Optional[str]:
        """
        의도 분류 신뢰도 검사.
        낮으면 "다시 말해줄래?" 응답 후보 반환, 충분하면 None.
        """
        self.confidence_checks += 1
        
        if confidence < CONFIDENCE_THRESHOLD:
            self.confidence_low_count += 1
            # 응답 회차에 따라 다른 표현 (반복 방지)
            idx = self.confidence_low_count % len(CONFIDENCE_RESPONSES)
            return CONFIDENCE_RESPONSES[idx]
        return None

    def is_confident(self, confidence: float) -> bool:
        return confidence >= CONFIDENCE_HIGH

    # ===== 2. Decay =====
    
    def apply_decay(self, force: bool = False):
        """
        학습된 응답/모듈 utility 약화.
        하루에 한 번만 (force=True면 무시).
        """
        now = time.time()
        elapsed_days = (now - self.last_decay_run) / 86400.0
        
        if not force and elapsed_days < 1.0:
            return False
        
        self.decay_runs += 1
        self.last_decay_run = now
        decay_factor = DECAY_RATE_PER_DAY ** elapsed_days
        
        # teaching_adapter의 학습된 응답 약화
        if (self.engine is not None
                and hasattr(self.engine, "teaching_adapter")
                and self.engine.teaching_adapter is not None):
            for lr in self.engine.teaching_adapter.learned:
                # 사용 안 한 거 약화 (times_used 적은 거)
                if hasattr(lr, "strength"):
                    lr.strength = max(DECAY_FLOOR, lr.strength * decay_factor)
        
        # module_learning_adapter의 utility 약화
        if (self.engine is not None
                and hasattr(self.engine, "module_learning_adapter")
                and self.engine.module_learning_adapter is not None):
            ml = self.engine.module_learning_adapter
            if hasattr(ml, "utility") and isinstance(ml.utility, dict):
                for key in list(ml.utility.keys()):
                    ml.utility[key] = max(DECAY_FLOOR, ml.utility[key] * decay_factor)
        
        return True

    def boost_on_use(self, item_type: str, item_key: str):
        """학습된 거 사용 시 강화 (decay 반대)."""
        # teaching
        if (item_type == "teaching" and self.engine is not None
                and hasattr(self.engine, "teaching_adapter")):
            for lr in self.engine.teaching_adapter.learned:
                if hasattr(lr, "response") and lr.response == item_key:
                    if hasattr(lr, "strength"):
                        lr.strength = min(2.0, lr.strength + USAGE_BOOST)
                    break

    # ===== 3. Conflict =====
    
    def add_belief(self, subject: str, predicate: str, object_: str,
                   polarity: str = "positive", confidence: float = 1.0):
        """믿음 추가 — 모순 자동 감지."""
        new_belief = Belief(
            subject=subject,
            predicate=predicate,
            object_=object_,
            polarity=polarity,
            timestamp=time.time(),
            confidence=confidence,
        )
        
        # 기존 믿음과 모순 체크
        opposite_pred = PREDICATE_OPPOSITES.get(predicate, "")
        for existing in self.beliefs:
            if existing.subject != subject:
                continue
            if existing.object_ != object_:
                continue
            
            # 같은 술어 + 반대 polarity = 모순
            if existing.predicate == predicate and existing.polarity != polarity:
                self.conflicts_detected += 1
                self._resolve_conflict(existing, new_belief)
                return
            # 반대 술어 + 같은 polarity = 모순
            if existing.predicate == opposite_pred and existing.polarity == polarity:
                self.conflicts_detected += 1
                self._resolve_conflict(existing, new_belief)
                return
        
        self.beliefs.append(new_belief)

    def _resolve_conflict(self, old: Belief, new: Belief):
        """
        모순 해소 정책 — 최근 + 신뢰도 우선.
        새 것 confidence > 0.7 면 새 것으로 대체.
        그 외엔 기록만 (사용자에게 물어볼 수 있게).
        """
        if new.confidence > 0.7:
            # 새 것으로 대체
            self.beliefs.remove(old)
            self.beliefs.append(new)
            self.conflicts_resolved += 1

    def get_belief(self, subject: str, predicate: str, object_: str) -> Optional[Belief]:
        """기존 믿음 조회."""
        for b in self.beliefs:
            if b.subject == subject and b.predicate == predicate and b.object_ == object_:
                return b
        return None

    def has_conflict(self, subject: str, predicate: str, object_: str,
                     polarity: str) -> bool:
        """이 새 정보가 기존 믿음과 모순?"""
        existing = self.get_belief(subject, predicate, object_)
        if existing is None:
            return False
        if existing.polarity != polarity:
            return True
        return False

    # ===== 4. Exploration =====
    
    def record_response(self, response: str):
        """응답 기록 (반복 검출용)."""
        self.recent_responses.append(response.strip())

    def is_repetitive(self, response: str) -> bool:
        """이 응답이 최근 너무 자주 나왔는지."""
        if not self.recent_responses:
            return False
        count = sum(1 for r in self.recent_responses if r == response.strip())
        return count >= EXPLORATION_THRESHOLD

    def force_alternative(self, original: str, alternatives: list) -> str:
        """반복 감지 시 alternative 선택."""
        if not alternatives:
            return original
        self.exploration_triggers += 1
        # 최근 안 쓴 것 우선
        for alt in alternatives:
            if alt not in self.recent_responses:
                return alt
        # 다 썼으면 첫번째
        return alternatives[0]

    # ===== 통계 =====

    def stats(self) -> dict:
        return {
            "confidence_checks": self.confidence_checks,
            "confidence_low": self.confidence_low_count,
            "decay_runs": self.decay_runs,
            "beliefs_count": len(self.beliefs),
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "exploration_triggers": self.exploration_triggers,
            "recent_responses_size": len(self.recent_responses),
        }
