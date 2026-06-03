"""
Language Generator

eve2.md 3.10 규칙:
- 짧은 문장 위주
- 점진적 생성
- 감정 반영

4단계: stage1(망설임) -> core -> expand(sub points) -> finalize(마무리)
"""

from utils.types import ResponsePlan


# 톤별 stage1 (망설임/주의 환기) — 결정론적, 한 톤당 1개
STAGE1_BY_TONE: dict[str, str] = {
    "calm":    "음...",
    "warm":    "어...",
    "firm":    "잠깐,",
    "curious": "오?",
    "neutral": "음...",
}

# 자율 발화용 stage1 풀 (단조로움 방지)
PROACTIVE_STAGE1_POOL: dict[str, list] = {
    "warm":    ["어...", "음...", "그러고 보니,", "아,", "흠..."],
    "calm":    ["음...", "그래...", "흠...", "조용하네.", ""],
    "curious": ["오?", "엇?", "흠?", "잠깐,", "어라,"],
    "neutral": ["음...", "그래.", "흠.", "어,", ""],
    "firm":    ["응.", "그래,", "맞아,", ""],
}

# 톤별 마무리 — 감정 톤에 맞춤. 다양한 풀에서 선택 (반복 방지)
# 라운드 52: 단일 고정 X — 풀로 다양화 (사용자 지시: "응 듣고 있어" 강제 제거)
FINALIZE_BY_TONE: dict[str, str] = {
    "calm":    "조금 쉬어가도 괜찮아.",
    "warm":    "옆에 있을게.",
    "firm":    "그렇게 하자.",
    "curious": "더 얘기해줄래?",
    "neutral": "",   # 라운드 52: 빈 문자열 (강제 마무리 X)
}

# 라운드 52: neutral 톤일 때 풀에서 선택 (없는 게 디폴트)
NEUTRAL_FINALIZE_POOL: list = [
    "",          # 마무리 없음 (자연스러움)
    "",          # 가중치 — 비울 확률 높음
    "",
    "음.",
    "그렇구나.",
    "그래.",
]

# 자율 발화용 — 마무리 없거나 다양 (혼잣말 같은 느낌)
PROACTIVE_FINALIZE_POOL: dict[str, list] = {
    "warm":    ["", "음.", "그런 생각이 들어.", "괜히.", "그냥."],
    "calm":    ["", "그렇네.", "조용하다.", "음."],
    "curious": ["", "왜 그런 거지?", "흠.", "재밌네."],
    "neutral": ["", "음.", "그렇구나.", "그냥."],
    "firm":    ["", "응.", "그래."],
}


class LanguageGenerator:

    def generate_stage1(self, plan: ResponsePlan) -> str:
        if plan.stage1_hint:
            return plan.stage1_hint
        return STAGE1_BY_TONE.get(plan.emotional_tone, STAGE1_BY_TONE["neutral"])

    def generate_core(self, plan: ResponsePlan) -> str:
        text = plan.core_message.rstrip(".")
        return text + "."

    def expand(self, sub: str) -> str:
        sub = sub.rstrip(".")
        return sub + "."

    def finalize(self, plan: ResponsePlan) -> str:
        # 라운드 52: neutral은 풀에서 결정론 선택 (강제 X)
        tone = plan.emotional_tone
        if tone == "neutral" or tone not in FINALIZE_BY_TONE:
            # 결정론 — core_message hash로 선택
            seed = sum(ord(c) for c in (plan.core_message or "")) % len(NEUTRAL_FINALIZE_POOL)
            return NEUTRAL_FINALIZE_POOL[seed]
        return FINALIZE_BY_TONE.get(tone, "")
