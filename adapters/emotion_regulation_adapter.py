"""
EmotionRegulationAdapter

v40 EmotionRegulation (Gross 1998) ↔ v41 감정 자기 조절.

흐름:
1. detect_need → cort/tension 임계 넘으면 자동 트리거
2. regulate(strategy=auto) — reappraisal / suppression / distraction
3. 호르몬 자동 조정 (cort↓, 5HT↑ 등)

v41 통합:
- 응답 후 ER tick → 부정 감정 누적 시 자기 조절
- Suffering burnout과 함께 작동
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from emotion_regulation import EmotionRegulation


class EmotionRegulationAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 memory_adapter=None,
                 cf_adapter=None,
                 tm_adapter=None,
                 er: EmotionRegulation | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if er is not None:
            self.er = er
        else:
            self.er = EmotionRegulation(
                hormone_adapter.hs,
                activation_adapter.sa,
                working_memory=activation_adapter.wm,
                counterfactual=cf_adapter.cf if cf_adapter else None,
                temporal=tm_adapter.tm if tm_adapter else None,
                episodic_memory=memory_adapter.em if memory_adapter else None,
            )

    def detect_need(self) -> dict | None:
        try:
            return self.er.detect_need()
        except Exception:
            return None

    def auto_regulate(self) -> dict | None:
        """필요 시 자동 조절."""
        try:
            return self.er.regulate(auto=True)
        except Exception:
            return None
