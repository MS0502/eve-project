"""
korean_endings — 라운드 65

한국어 어미 변형. GPT 짚은 "개념 → 반응 변환".

원칙:
- 끝 어절 검사로 *동사/형용사/명사* 구별
- 단순 변형 (X했어 → X했구나)
- 어색한 케이스는 fallback 명사 처리

학계: 한국어 형태소 - 종결어미 활용
"""


# 동사 과거 종결어미 → 인정 어미 (구나/네)
PAST_VERB_ENDINGS = {
    "했어": "했구나",
    "했다": "했구나",
    "했어요": "했네",
    "했지": "했구나",
    "갔어": "갔구나",
    "왔어": "왔구나",
    "봤어": "봤구나",
    "먹었어": "먹었구나",
    "잤어": "잤구나",
    "마셨어": "마셨구나",
}

# 형용사/감정 → 추측 어미 (겠다)
ADJ_ENDINGS = {
    "다": "겠다",            # 피곤하다 → 피곤하겠다
    "해": "하겠다",          # 피곤해 → 피곤하겠다
}

# 어색한 단어 (변형 X — 명사 처리)
NO_TRANSFORM = {"이", "그", "저", "뭐", "왜", "응", "어"}


# 라운드 69: 깨진 어절 차단
# 한 글자 잔재 / 어미 잔재만 있는 거 — 완성된 단어 X
BROKEN_FRAGMENTS = {"동", "했", "어", "는", "은", "이", "가", "을", "를",
                    "다", "고", "지", "기", "면", "야", "아", "응"}


def _is_safe_word(word: str) -> bool:
    """라운드 69: 깨진 어절 차단. *길이 검사 X* — 한국어 1자 어간 허용."""
    if not word:
        return False
    if word in BROKEN_FRAGMENTS:
        return False
    return True


def _is_safe_input(word: str) -> bool:
    """*원본 입력* 안전 — 1자 단독 입력은 너무 모호."""
    if not word or len(word) < 2:
        return False
    if word in BROKEN_FRAGMENTS:
        return False
    return True


def reaction_for(entity: str) -> str | None:
    """
    entity → 사람 반응 변환.
    
    "운동했어" → "운동했구나"
    "먹었어" → "먹었구나"
    "동했어" → None  ← 깨진 어절 차단
    """
    if not _is_safe_input(entity):
        return None
    if entity in NO_TRANSFORM:
        return None
    
    # 라운드 69: entity가 키 자체면 — 그 replacement 그대로 반환
    if entity in PAST_VERB_ENDINGS:
        return PAST_VERB_ENDINGS[entity]
    
    # 동사 과거 어미 우선 — 어간 + 어미
    for ending, replacement in sorted(PAST_VERB_ENDINGS.items(), key=lambda x: -len(x[0])):
        if entity.endswith(ending) and len(entity) > len(ending):
            stem = entity[:-len(ending)]
            if not _is_safe_word(stem):
                return None
            result = f"{stem}{replacement}"
            if not result.startswith(stem):
                return None
            return result
    
    # 형용사 어미
    for ending, replacement in sorted(ADJ_ENDINGS.items(), key=lambda x: -len(x[0])):
        if entity.endswith(ending) and len(entity) > len(ending):
            if ending == "다" and len(entity) < 3:
                continue
            stem = entity[:-len(ending)]
            if not _is_safe_word(stem):
                return None
            result = f"{stem}{replacement}"
            if not result.startswith(stem):
                return None
            return result
    
    return None


def is_action_word(entity: str) -> bool:
    """동사/행동 어절인지 (과거형 어미 검사)."""
    return any(entity.endswith(e) for e in PAST_VERB_ENDINGS)


def is_state_word(entity: str) -> bool:
    """형용사/상태 어절인지."""
    return any(entity.endswith(e) for e in ADJ_ENDINGS)
