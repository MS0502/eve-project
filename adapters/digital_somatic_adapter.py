"""
DigitalSomaticAdapter

v40 DigitalSomatic (Damasio 1999 Protoself) ↔ v41 신체 감각.

흐름:
1. compute_somatic_state — 호르몬 → 8 차원 신체 상태
   (energy, tension, warmth, clarity, fatigue, mental_load, activity_level, emotional_intensity)
2. get_dominant_need — 가장 강한 신체 욕구
3. get_feeling — 한 줄 한국어 표현
4. get_gut_signal — 행동 후보에 대한 직관 신호 (Agency 활용)

v41 통합:
- 응답 후 tick (호르몬 변화 → 신체 상태 갱신)
- "지금 어때?" 같은 질문 → get_feeling 응답
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from digital_somatic import DigitalSomatic

from utils.types import Meaning


# 신체 상태 질문 트리거 키워드
BODY_QUERY_KEYWORDS = ("기분", "상태", "느낌이", "지금 어때")


class DigitalSomaticAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 ds: DigitalSomatic | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if ds is not None:
            self.ds = ds
        else:
            self.ds = DigitalSomatic(
                hormone_adapter.hs,
                activation_adapter.wm,
                activation_adapter.sa,
            )

    def is_body_query(self, meaning: Meaning) -> bool:
        text = meaning.raw_text or ""
        return any(kw in text for kw in BODY_QUERY_KEYWORDS)

    def feeling_phrase(self) -> str:
        try:
            return self.ds.get_feeling()
        except Exception:
            return "잘 모르겠어"

    def somatic_state(self) -> dict[str, float]:
        try:
            return self.ds.compute_somatic_state() or {}
        except Exception:
            return {}

    def dominant_need(self) -> tuple[str, float] | None:
        try:
            return self.ds.get_dominant_need()
        except Exception:
            return None

    def gut_signal(self, action: str) -> dict[str, float]:
        try:
            return self.ds.get_gut_signal(action) or {}
        except Exception:
            return {}

    def notify_dmn(self, mode: str | None):
        try:
            self.ds.notify_dmn_activity(mode)
        except Exception:
            pass
