"""
CorpusAdapter

v40 CorpusLearner (한국어 텍스트 → 카테고리 페어) ↔ v41 텍스트 학습.

흐름:
1. learn_sentence(문장) — NL 카테고리 추출 + Hebbian 페어
2. learn_document(긴 글) — 단락별 학습
3. discover_groups — 자주 같이 나타나는 카테고리 그룹

v41 통합:
- /learn 명령으로 사용자가 텍스트 주입 가능
- 매 turn 입력도 자동 학습 (조용한 학습)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from corpus_learner import CorpusLearner

from utils.types import Meaning


class CorpusAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 nl_adapter,                          # 필수 — NL 의존
                 cl: CorpusLearner | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.nl_adapter = nl_adapter
        if cl is not None:
            self.cl = cl
        else:
            self.cl = CorpusLearner(
                activation_adapter.sa,
                nl_adapter.nl,
                hormone_system=hormone_adapter.hs,
            )

    def auto_learn_from_input(self, meaning: Meaning):
        """매 turn 입력에서 자동 학습 (조용히)."""
        text = meaning.raw_text or ""
        if len(text) < 5:        # 너무 짧으면 skip
            return None
        # 라운드 46: 오타/노이즈 거르기 — 단어 길이 평균 1자 이하면 skip
        words = text.split()
        if not words:
            return None
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 1.5:
            return None
        # 너무 노이즈 많은 입력 (한 단어가 다른 한국어 글자 무관계 조합)
        # 간단 휴리스틱: 한국어 글자 비율 + 의문문 / 명령문이면 OK
        try:
            return self.cl.learn_sentence(text)
        except Exception:
            return None

    def learn_text(self, text: str) -> dict:
        """명시적 학습 — 문장/단락/문서 자동 분기."""
        try:
            if len(text) < 100:
                return self.cl.learn_sentence(text)
            elif len(text) < 500:
                return self.cl.learn_paragraph(text)
            else:
                return self.cl.learn_document(text)
        except Exception as e:
            return {"error": str(e)}

    def discover(self) -> dict[str, set[str]]:
        """자주 같이 나타나는 카테고리 그룹 발견."""
        try:
            return self.cl.discover_groups() or {}
        except Exception:
            return {}

    def stats(self) -> dict:
        try:
            return self.cl.stats() or {}
        except Exception:
            return {}
