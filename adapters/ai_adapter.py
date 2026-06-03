"""
AIAdapter

v40 ActiveInference (Friston 2010 FEP) ↔ v41 응답 전후 학습 루프.

흐름:
1. 응답 전: predict(horizon=1.0) — 호르몬 변화 예측
2. 응답 후: observe — 실제 결과 기록
3. update_from_outcome — SD 학습 (예측 doubt vs 실제 결과)

v41 응답 영향:
- 직접 응답 텍스트 변경하진 않음 (학습만)
- 백그라운드에서 호르몬-카테고리 매핑 강화
- 시간 지나면서 호르몬 패턴이 더 정교해짐
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from active_inference import ActiveInference


class AIAdapter:

    def __init__(self,
                 hormone_adapter,                    # 필수
                 activation_adapter,                  # 필수
                 sd_adapter=None,                     # optional
                 ai: ActiveInference | None = None):
        if ai is not None:
            self.ai = ai
        else:
            sd = sd_adapter.sd if sd_adapter else None
            self.ai = ActiveInference(
                hormone_adapter.hs,
                activation_adapter.sa,
                activation_adapter.wm,
                self_doubt=sd,
            )
        self._last_prediction = None

    def predict_before_response(self, horizon: float = 1.0) -> dict | None:
        """응답 만들기 전에 호출. 호르몬/카테고리 변화 예측."""
        try:
            self._last_prediction = self.ai.predict(horizon=horizon)
            return self._last_prediction
        except Exception:
            self._last_prediction = None
            return None

    def observe_after_response(self, outcome: str = "neutral") -> dict | None:
        """
        응답 만든 후 호출.
        outcome: 'positive' (사용자 만족 추정) / 'negative' / 'neutral'
        """
        if self._last_prediction is None:
            return None
        pred_id = self._last_prediction.get("id") if isinstance(self._last_prediction, dict) else None
        try:
            return self.ai.observe(prediction_id=pred_id, actual_outcome=outcome)
        except Exception:
            return None

    def tick(self, dt: float = 1.0):
        try:
            self.ai.tick(dt=dt)
        except Exception:
            pass

    def get_state(self) -> dict:
        try:
            return self.ai.get_state()
        except Exception:
            return {}
