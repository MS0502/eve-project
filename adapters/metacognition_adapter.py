"""
MetacognitionAdapter

v40 MetaCognition (Flavell 1979, Nelson & Narens 1990) ↔ v41 자기 평가.

흐름:
1. evaluate_confidence — 응답 후 자기 확신도 평가
2. endorse_or_override — 신념 충돌 시 endorse / override 결정
3. 낮은 confidence면 응답 톤이 자연스럽게 'curious'/'uncertain'으로 흐름

v41 통합:
- 응답 후 confidence 기록 (학습 신호)
- proactive 발화 시 메타 인지 사용 가능
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from metacognition import MetaCognition

from utils.types import Meaning


class MetacognitionAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 sd_adapter=None,
                 ai_adapter=None,
                 dmn_adapter=None,
                 mc: MetaCognition | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if mc is not None:
            self.mc = mc
        else:
            self.mc = MetaCognition(
                hormone_adapter.hs,
                activation_adapter.sa,
                working_memory=activation_adapter.wm,
                self_doubt=sd_adapter.sd if sd_adapter else None,
                active_inference=ai_adapter.ai if ai_adapter else None,
                dmn=dmn_adapter.dmn if dmn_adapter else None,
            )

    def confidence(self, meaning: Meaning, response: str) -> float:
        """현재 응답의 confidence 0~1."""
        try:
            r = self.mc.evaluate_confidence()
            if isinstance(r, dict):
                return float(r.get("confidence", 0.5))
            return float(r) if r is not None else 0.5
        except Exception:
            return 0.5

    def get_state(self) -> dict:
        try:
            return self.mc.get_state()
        except Exception:
            return {}
