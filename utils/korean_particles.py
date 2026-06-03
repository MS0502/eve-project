"""
korean_particles.py — 한국어 조사 자동 처리

원칙: 받침 있으면 "은/이/을", 없으면 "는/가/를"
- 운동 (받침 ㅇ) + 은 → "운동은"
- 야구 (받침 X) + 는 → "야구는"

학계: 표준 한국어 문법 (의존명사 + 조사 결합)
"""


def has_jongseong(word: str) -> bool:
    """단어 마지막 글자에 받침 있는지.
    - 한글: 종성 코드로 판정
    - 영어: 발음 기준 (자음 끝나도 한국어 발음 시 모음 추가되는 경우 多)
      PT → 피티 (받침 X), K → 케이 (받침 X)
      L → 엘 (받침 ㄹ), M → 엠 (받침 ㅁ), N → 엔 (받침 ㄴ), 
      P → 피 (받침 X), B → 비 (X), G → 지 (X)
    """
    if not word:
        return False
    last = word[-1]
    if '\uac00' <= last <= '\ud7af':
        # 한글
        code = ord(last) - 0xAC00
        return code % 28 != 0
    # 영어 — 한국어 발음에서 받침 남는 자음만
    last_l = last.lower()
    if last_l in "lmnr":
        return True
    return False  # 모음, 또는 발음 시 모음 추가되는 자음


def topic_particle(word: str) -> str:
    """은/는 — 화제 조사."""
    return word + ("은" if has_jongseong(word) else "는")


def subject_particle(word: str) -> str:
    """이/가 — 주격 조사."""
    return word + ("이" if has_jongseong(word) else "가")


def object_particle(word: str) -> str:
    """을/를 — 목적격 조사."""
    return word + ("을" if has_jongseong(word) else "를")


def with_particle(word: str) -> str:
    """과/와 — 접속 조사."""
    return word + ("과" if has_jongseong(word) else "와")


def quote_ending(word: str) -> str:
    """이라는/라는 — 인용 정의."""
    return word + ("이라는" if has_jongseong(word) else "라는")


def copula(word: str) -> str:
    """이야/야 — 서술격 조사."""
    return word + ("이야" if has_jongseong(word) else "야")


def copula_ji(word: str) -> str:
    """이지/지 — 단정 종결."""
    return word + ("이지" if has_jongseong(word) else "지")


def vocative(word: str) -> str:
    """아/야 — 호격 조사 (사람 이름 부르기)."""
    return word + ("아" if has_jongseong(word) else "야")
