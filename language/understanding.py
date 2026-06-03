"""
Language Understanding: text -> Meaning

규칙 (eve2.md 3.1):
- 문장 → 의미로 변환
- 감정 반드시 추출
- 시간/맥락 태그 생성
- task 추출 (계산기 등)

v41 첫 버전: 한국어 키워드 룰 기반.
"""

import re

from utils.types import Meaning


# 감정 키워드 사전 (확장 가능)
EMOTION_KEYWORDS: dict[str, dict[str, float]] = {
    "fatigue":  {"힘들": 0.8, "피곤": 0.9, "지쳤": 0.85, "졸려": 0.6, "쉬고싶": 0.7},
    "sadness":  {"슬프": 0.8, "우울": 0.85, "외로": 0.7, "눈물": 0.75},
    "joy":      {"기쁘": 0.8, "행복": 0.85, "좋아": 0.5, "신나": 0.7},
    "anger":    {"짜증": 0.7, "화나": 0.8, "빡쳐": 0.85, "열받": 0.75},
    "anxiety":  {"불안": 0.8, "걱정": 0.7, "초조": 0.75},
    "warmth":   {"고마": 0.7, "사랑": 0.8, "따뜻": 0.6},
}

# 의도 마커
QUESTION_MARKERS = ("?", "뭐", "어디", "언제", "왜", "어떻", "누구")
COMMAND_MARKERS = ("해줘", "해라", "해봐", "하자")

# 산술 task — 한국어 연산자 + 숫자 추출
KOREAN_ARITH_OPS = {
    "더하기": "+", "빼기": "-", "곱하기": "*", "나누기": "/",
    "+": "+", "-": "-", "*": "*", "/": "/",
    "플러스": "+", "마이너스": "-",
}

# 시뮬레이션 패턴: "X 하면 ~?", "Y 하면 어때?"
_SIMULATION_PATTERN = re.compile(r"(\S+?)\s*(?:이|가|을|를)?\s*(?:하|먹|자|쉬)면")

# 산술 표현 패턴: "3 + 5", "3 더하기 5", "3 곱하기 5는"
_ARITH_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(\+|-|\*|/|더하기|빼기|곱하기|나누기|플러스|마이너스)\s*(-?\d+(?:\.\d+)?)"
)

# 문자열 길이: "X 길이", "X 글자수"
_STRLEN_PATTERN = re.compile(r"['\"](.+?)['\"]\s*(?:글자\s*수|길이)|(\S+?)\s+글자\s*수")


def extract_simulation_query(text: str) -> str | None:
    """'X 하면', 'X 먹으면', 'X 쉬면' 패턴 → simulation seed."""
    m = _SIMULATION_PATTERN.search(text)
    if not m:
        return None
    seed = m.group(1).strip()
    for suf in ("을", "를", "이", "가", "은", "는"):
        if seed.endswith(suf) and len(seed) > 1:
            seed = seed[:-1]
            break
    if seed and len(seed) >= 1:
        return seed
    return None


def extract_task(text: str) -> dict | None:
    """text에서 결정론적 task 추출. 없으면 None."""
    # 1) 산술
    m = _ARITH_PATTERN.search(text)
    if m:
        left, op_text, right = m.group(1), m.group(2), m.group(3)
        op_sym = KOREAN_ARITH_OPS.get(op_text, op_text)
        return {"type": "arithmetic", "expr": f"{left} {op_sym} {right}"}
    # 2) 문자열 길이
    m = _STRLEN_PATTERN.search(text)
    if m:
        target = m.group(1) or m.group(2)
        if target:
            return {"type": "string_length", "target": target}
    return None


class LanguageUnderstanding:

    def __init__(self, nl_adapter=None):
        self.nl_adapter = nl_adapter

    def parse(self, text: str) -> Meaning:
        text = text.strip()
        meaning = Meaning(
            intent=self._detect_intent(text),
            entities=self._extract_entities(text),
            actions=self._extract_actions(text),
            emotions=self._extract_emotions(text),
            context={},
            raw_text=text,
        )
        # task 추출 → meaning.context 에 박음
        task = extract_task(text)
        if task is not None:
            meaning.context["task"] = task
            # task가 있는 입력은 question 의도로 보정
            if meaning.intent in ("statement", "emotion"):
                meaning.intent = "question"

        # simulation 추출 (CF/temporal 보다 우선 안 함 — context만 채움)
        sim_seed = extract_simulation_query(text)
        if sim_seed:
            meaning.context["sim_query"] = sim_seed

        # NL 어댑터 있으면 entities/intent 강화
        if self.nl_adapter is not None:
            meaning = self.nl_adapter.enhance(meaning)
        return meaning

    def _detect_intent(self, text: str) -> str:
        if any(m in text for m in QUESTION_MARKERS):
            return "question"
        if any(m in text for m in COMMAND_MARKERS):
            return "command"
        emotions = self._extract_emotions(text)
        if emotions and max(emotions.values()) >= 0.7:
            return "emotion"
        return "statement"

    def _extract_emotions(self, text: str) -> dict[str, float]:
        found: dict[str, float] = {}
        for emotion, kws in EMOTION_KEYWORDS.items():
            best = 0.0
            for kw, weight in kws.items():
                if kw in text:
                    best = max(best, weight)
            if best > 0:
                found[emotion] = best
        return found

    def _extract_entities(self, text: str) -> list[str]:
        # v41 first cut: 한글 2자 이상 토큰만 추림
        # 나중에 frame_semantics.py 어댑터로 교체.
        tokens = text.replace(",", " ").replace(".", " ").split()
        return [t for t in tokens if len(t) >= 2 and any("가" <= c <= "힣" for c in t)]

    def _extract_actions(self, text: str) -> list[str]:
        # 단순 동사 어간 일부. frame_semantics 들어오면 교체.
        actions = []
        for stem in ["가다", "오다", "먹다", "쉬다", "자다", "공부", "운동", "일하"]:
            if stem in text or stem[:-1] in text:
                actions.append(stem)
        return actions
