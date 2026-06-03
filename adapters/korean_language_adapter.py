"""
KoreanLanguageAdapter — 라운드 34 (C-2) Wrapper

morph_analyzer + sentence_structure를 engine에 통합.
다른 어댑터(orchestrator, teaching 등)에서 이 어댑터로 한국어 분석.
"""

from adapters.korean.morph_analyzer import KoreanMorphAnalyzer
from adapters.korean.sentence_structure import (
    SentenceStructureAnalyzer,
    IntentStructure,
)


class KoreanLanguageAdapter:
    """한국어 형태소 + 구조 + 의도 분석 통합."""

    def __init__(self, engine=None):
        self.engine = engine
        self.morph = KoreanMorphAnalyzer()
        self.structure = SentenceStructureAnalyzer(self.morph)
        # 분석 캐시 (같은 turn에 여러 번 호출 가능)
        self._cache: dict = {}
        self._cache_max = 100

    def analyze(self, text: str) -> IntentStructure:
        """텍스트 → 의미 구조 + 의도."""
        if text in self._cache:
            return self._cache[text]
        result = self.structure.analyze(text)
        if len(self._cache) >= self._cache_max:
            # FIFO 제거
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[text] = result
        return result

    def analyze_morph(self, text: str):
        """형태소 분석만."""
        return self.morph.analyze(text)

    # 편의 메서드
    def is_question(self, text: str) -> bool:
        return self.analyze(text).sentence_type == "interrogative"

    def is_teaching(self, text: str) -> bool:
        return self.analyze(text).intent == "teaching"

    def is_definition(self, text: str) -> bool:
        return self.analyze(text).intent == "definition"

    def is_meta_self(self, text: str) -> bool:
        return self.analyze(text).intent == "meta_self"

    def is_meta_user(self, text: str) -> bool:
        return self.analyze(text).intent == "meta_user"

    def is_emotional(self, text: str) -> bool:
        return self.analyze(text).intent == "emotional"
