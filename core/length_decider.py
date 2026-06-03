"""
LengthDecider — GPT 라운드 12 핵심.

길이 = 상태의 결과 (고정 X).

기준 4개:
1. 감정 강도 (emotion intensity)
2. 질문 여부 (question)
3. 대화 깊이 (context depth)
4. 관계 친밀도 (intimacy)

추가:
- 호르몬 (cort↓ + ot↑ → 더 길게)
- 호르몬 결정론 보정 (random 안 씀 — cort/ot로 ±1 흔들림)

LLM 0, random 0.
"""

from utils.types import Meaning, WorkspaceState


def decide_response_length(state: WorkspaceState,
                           meaning: Meaning,
                           context_depth: int = 0,
                           intimacy: float = 0.0,
                           hormone_adapter=None) -> int:
    """
    Returns: total response 줄 수 (core_message + sub_points).
    1줄 (가벼움) ~ 5줄 (깊은 대화).
    """
    length = 1   # 기본: core only

    # 1) 질문이면 +1
    if meaning.intent == "question":
        length += 1

    # 2) 감정 강도
    if meaning.emotions:
        emotion_max = max(meaning.emotions.values())
        if emotion_max > 0.7:
            length += 2
        elif emotion_max > 0.4:
            length += 1

    # 3) 대화 깊이
    if context_depth > 3:
        length += 1

    # 4) 친밀도
    if intimacy > 0.5:
        length += 1

    # 5) 호르몬 결정론 흔들림 (random 대체)
    # cort 높으면 짧게 (말 줄이고 싶음), ot 높으면 길게 (더 나누고 싶음)
    if hormone_adapter is not None:
        hs = hormone_adapter.hs
        cort = hs.hormones["cortisol"].level
        ot = hs.hormones["oxytocin"].level
        if cort > 0.7:
            length -= 1
        elif ot > 0.6 and cort < 0.4:
            length += 1

    # clamp
    return max(1, min(length, 5))
