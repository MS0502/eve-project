"""
EnhancerAdapter

v40 ResponseEnhancer (응답 패턴 12종) ↔ v41 자연스러운 한국어 응답.

12 패턴:
- refusal, identity, self_intro, why, memory, location_query
- 3rd_party_event, 1st_party_event, physical, emotion, greeting
- name_call, affirm, statement

v41 통합:
- Planner의 generic core_message 대체 (특정 패턴 인식 시)
- 패턴 분류 → meaning.context["enhancer_pattern"]
- 짧은 직접 응답이 더 자연스러울 때만 사용
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from response_enhancer import ResponseEnhancer

from utils.mock_eve import MockEve


# 어느 패턴일 때 RE 응답으로 대체할지
ENHANCER_REPLACEABLE_PATTERNS = {
    "greeting", "identity", "self_intro", "name_call",
    "refusal", "why",
}


class EnhancerAdapter:

    def __init__(self, engine, re: ResponseEnhancer | None = None):
        self.engine = engine
        self._mock = MockEve(engine)
        if re is not None:
            self.re = re
        else:
            self.re = ResponseEnhancer(
                self._mock.nl,
                self._mock.sa,
                self._mock.wm,
                self._mock.hs,
                em=self._mock.em,
                cg=getattr(self._mock, "cg", None),
                fs=getattr(self._mock, "fs", None),
                ds=getattr(self._mock, "ds", None),
                eve=self._mock,
            )

    def classify(self, text: str, understanding: dict) -> str:
        try:
            return self.re.classify_pattern(text, understanding)
        except Exception:
            return "statement"

    def enhance(self, text: str, understanding: dict) -> tuple[str | None, str]:
        """
        Returns (enhanced_response_or_None, pattern_name).
        replaceable 패턴이면 응답 반환, 아니면 None.
        """
        pattern = self.classify(text, understanding)
        if pattern not in ENHANCER_REPLACEABLE_PATTERNS:
            return (None, pattern)
        try:
            response = self.re.respond_v3(text, understanding)
            return (response, pattern)
        except Exception:
            return (None, pattern)

    def get_stats(self) -> dict:
        try:
            return self.re.get_stats() or {}
        except Exception:
            return {}
