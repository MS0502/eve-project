"""
NLAdapter

v40 NaturalLanguage ↔ v41 LanguageUnderstanding 강화.

v40 NL의 강점:
- 한국어 조사 떼기 ("PT가" → "PT")
- 어미 처리 ("힘들었다" → "힘들다")
- 4977개 신념 검색
- 의도 추론 (question/emotion/statement)

v41 LU와 합침:
- entities는 NL.categories로 교체 (정확도 ↑)
- intent는 NL이 우선 (한국어 detection 더 정확)
- emotions는 v41 EMOTION_KEYWORDS 유지 (강도 가져오려)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from natural_lang import NaturalLanguage

from utils.types import Meaning


# v40 NL.intent → v41 Meaning.intent 매핑
INTENT_MAP = {
    "question":  "question",
    "command":   "command",
    "emotion":   "emotion",
    "statement": "statement",
    "refusal":   "statement",
    "greeting":  "statement",
    None:        "statement",
}


class NLAdapter:

    def __init__(self,
                 hormone_adapter=None,
                 activation_adapter=None,
                 nl: NaturalLanguage | None = None):
        if nl is not None:
            self.nl = nl
        else:
            from spreading_activation import SpreadingActivation
            from working_memory import WorkingMemory
            from hormone_system import HormoneSystem
            sa = activation_adapter.sa if activation_adapter else SpreadingActivation()
            wm = activation_adapter.wm if activation_adapter else WorkingMemory()
            hs = hormone_adapter.hs if hormone_adapter else HormoneSystem()
            self.nl = NaturalLanguage(sa, wm, hs)

    def enhance(self, meaning: Meaning) -> Meaning:
        """v41 Meaning을 v40 NL 결과로 보강."""
        try:
            understanding = self.nl.understand(meaning.raw_text)
        except Exception:
            return meaning

        # entities: NL의 categories로 교체 (조사 떼진 깨끗한 형태)
        cats = understanding.get("categories", set())
        if cats:
            meaning.entities = sorted(cats)

        # intent: NL이 더 정확 → 우선
        nl_intent = understanding.get("intent")
        if nl_intent:
            meaning.intent = INTENT_MAP.get(nl_intent, "statement")

        # context에 NL 메타 정보
        meaning.context = meaning.context or {}
        meaning.context["nl_sentiment"] = understanding.get("sentiment")
        meaning.context["nl_has_doubt"] = understanding.get("has_doubt", False)
        meaning.context["nl_is_refusal_target"] = understanding.get("is_refusal_target", False)

        return meaning
