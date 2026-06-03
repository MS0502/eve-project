"""Deterministic Korean conversation fixtures for v3 round43 smoke sampling."""

KOREAN_CONVERSATION_FIXTURES = [
    {"category": "greeting", "text": "안녕"},
    {"category": "greeting", "text": "안녕하세요"},
    {"category": "daily", "text": "뭐해"},
    {"category": "daily", "text": "오늘 기분 어때"},
    {"category": "daily", "text": "잘 지내"},
    {"category": "daily", "text": "밥 먹었어"},
    {"category": "emotion", "text": "좋아"},
    {"category": "emotion", "text": "슬퍼"},
    {"category": "emotion", "text": "기뻐"},
    {"category": "emotion", "text": "힘들어"},
    {"category": "identity", "text": "너는 누구야"},
    {"category": "identity", "text": "너는 살아있어"},
    {"category": "identity", "text": "어떤 존재야"},
    {"category": "minsok", "text": "군대 생활 어때"},
    {"category": "minsok", "text": "코딩 좋아해"},
    {"category": "minsok", "text": "EVE 프로젝트"},
    {"category": "reasoning", "text": "왜 그래"},
    {"category": "reasoning", "text": "그게 뭐야"},
    {"category": "reasoning", "text": "어떻게 생각해"},
    {"category": "reasoning", "text": "오늘 날씨 좋다"},
]

REQUIRED_FIXTURE_CATEGORIES = frozenset(
    {"greeting", "daily", "emotion", "identity", "reasoning"}
)


def fixture_texts() -> list[str]:
    """Return fixture texts in stable round43 order."""
    return [row["text"] for row in KOREAN_CONVERSATION_FIXTURES]


def fixture_categories() -> list[str]:
    """Return fixture categories in stable round43 order."""
    return [row["category"] for row in KOREAN_CONVERSATION_FIXTURES]
