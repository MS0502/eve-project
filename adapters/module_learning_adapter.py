"""
ModuleLearningAdapter — 라운드 32 (B-4)

Orchestrator가 어떤 상황에 어떤 모듈을 썼는지 → 효과 측정 → 가중치 학습.

학습 신호:
- 좋음 (+): 자연스러운 이어가기 ("응", "그래", small_talk continuation)
- 나쁨 (-): 가르침 발생, 거부 키워드, 같은 질문 재시도, 명백한 빗나감

원칙:
- 결정론 (random 0)
- 사용자별 (현재는 단일 사용자라 글로벌)
- Hebbian-like: 효과 좋으면 강화
- 영속화 (persistence_adapter에 통합)

학계:
- Sutton & Barto 2018 — RL credit assignment
- Anderson 1996 — ACT-R production utility learning
- Hebb 1949 — 함께 활성된 것은 함께 강화
"""

import time
from typing import Optional


# 다음 turn 분석 — 효과 신호
POSITIVE_FOLLOWUP_PATTERNS = [
    "응", "그래", "맞아", "좋아", "그치", "오 그러네",
    "고마워", "좋네",
]

NEGATIVE_FOLLOWUP_PATTERNS = [
    "아니야", "아니", "이상해", "어색", "그게 아니",
    "내 말은", "다시 말하면", "아 그거 말고",
]

# 가르침 패턴 (이미 raw_text에서 잡히니 여기선 단어만)
TEACHING_INDICATORS = ["라고 말해", "라고 부르면", "이럴때", "이럴 때"]

# 같은 질문 재시도 (반복 키워드)
RETRY_INDICATORS = ["그러니까", "내 말은", "다시", "아니 그게"]


class ModuleLearningAdapter:

    def __init__(self, engine):
        self.engine = engine
        # (situation, module_name) → utility (0~2.0, 1.0 = 중립)
        self.module_utility: dict[tuple[str, str], float] = {}
        # 마지막 선택 모듈 (다음 turn에 효과 측정)
        self._last_selection: Optional[dict] = None
        # 통계
        self.adjustments_count = 0
        self.positive_count = 0
        self.negative_count = 0

    # ===== 효과 측정 =====

    def detect_effect(self, current_input: str, last_response: str) -> str:
        """
        현재 사용자 입력 → 마지막 EVE 응답 평가.
        Returns: "positive" | "negative" | "neutral"
        """
        text = current_input.strip()
        if not text:
            return "neutral"

        # 1. 명백한 부정 — 가르침 발생
        for ind in TEACHING_INDICATORS:
            if ind in text:
                return "negative"
        # 2. 명백한 부정 — 거부 키워드
        for kw in NEGATIVE_FOLLOWUP_PATTERNS:
            if text.startswith(kw) or text == kw:
                return "negative"
        # 3. 재시도 (응답이 빗나감)
        for ind in RETRY_INDICATORS:
            if ind in text:
                return "negative"
        # 4. 명백한 긍정 — 짧은 동의
        for kw in POSITIVE_FOLLOWUP_PATTERNS:
            if text == kw or text.startswith(kw + " ") or text == kw + ".":
                return "positive"
        # 5. 길어진 답변 (자연스러운 대화 이어가기) — 약한 긍정
        if 5 < len(text) < 100:
            return "weak_positive"
        return "neutral"

    # ===== 모듈 가중치 조정 =====

    def adjust(self, situation: str, modules: list[str], effect: str):
        """효과 신호로 모듈 utility 조정."""
        if effect == "neutral":
            return

        delta = 0.0
        if effect == "positive":
            delta = +0.15
        elif effect == "weak_positive":
            delta = +0.05
        elif effect == "negative":
            delta = -0.20

        for mod in modules:
            key = (situation, mod)
            current = self.module_utility.get(key, 1.0)
            new_val = max(0.1, min(2.0, current + delta))
            self.module_utility[key] = new_val

        self.adjustments_count += 1
        if delta > 0:
            self.positive_count += 1
        elif delta < 0:
            self.negative_count += 1

    def record_selection(self, situation: str, modules: list[str]):
        """이번 turn 어떤 모듈 썼는지 기록 (다음 turn에 평가)."""
        self._last_selection = {
            "situation": situation,
            "modules": list(modules),
            "time": time.time(),
        }

    def evaluate_last(self, current_input: str, last_response: str = ""):
        """마지막 turn 모듈 효과 측정 + 조정."""
        if self._last_selection is None:
            return None
        effect = self.detect_effect(current_input, last_response)
        self.adjust(
            self._last_selection["situation"],
            self._last_selection["modules"],
            effect,
        )
        return effect

    # ===== Orchestrator의 모듈 선택 보정 =====

    def reorder_modules(self, situation: str, modules: list[str]) -> list[str]:
        """
        모듈 리스트를 utility 순으로 재정렬 (utility 낮은 거 끝으로).
        utility < 0.5인 모듈은 제외.
        """
        scored = []
        for m in modules:
            u = self.module_utility.get((situation, m), 1.0)
            if u >= 0.5:        # 너무 약한 거 빼기
                scored.append((u, m))
        # utility 높은 순 (결정론적, 같은 점수면 원래 순서)
        scored.sort(key=lambda x: (-x[0], modules.index(x[1])))
        return [m for u, m in scored]

    # ===== 통계 + 영속화 =====

    def stats(self) -> dict:
        # 상위 학습된 (situation, module) 5개
        top = sorted(
            self.module_utility.items(),
            key=lambda x: -x[1],
        )[:5]
        bottom = sorted(
            self.module_utility.items(),
            key=lambda x: x[1],
        )[:5]
        return {
            "adjustments": self.adjustments_count,
            "positive": self.positive_count,
            "negative": self.negative_count,
            "learned_pairs": len(self.module_utility),
            "top_5": top,
            "bottom_5": bottom,
        }

    def to_dict(self) -> dict:
        return {
            "module_utility": [
                [list(k), v] for k, v in self.module_utility.items()
            ],
            "adjustments_count": self.adjustments_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }

    def from_dict(self, data: dict):
        self.module_utility = {}
        for item in data.get("module_utility", []):
            key = tuple(item[0])
            self.module_utility[key] = item[1]
        self.adjustments_count = data.get("adjustments_count", 0)
        self.positive_count = data.get("positive_count", 0)
        self.negative_count = data.get("negative_count", 0)
