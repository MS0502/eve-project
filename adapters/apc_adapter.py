"""
APCAdapter

v40 APCLearner (Active Predictive Coding) ↔ v41 자율 학습.

흐름:
1. 매 turn — observe_and_learn(현재 활성 카테고리 set)
2. APC가 prediction error 보고 SA 가중치 업데이트
3. 시간 지나면서 EVE의 카테고리 연결이 정교해짐

predict_next_state로 다음 turn 예측도 가능.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from apc_learner import APCLearner

from utils.types import Meaning


class APCAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 apc: APCLearner | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if apc is not None:
            self.apc = apc
        else:
            self.apc = APCLearner(
                activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
                working_memory=activation_adapter.wm,
            )

    def observe(self, meaning: Meaning):
        """현재 active 카테고리를 APC에 학습 신호로."""
        active = set(meaning.entities or [])
        active |= set((meaning.emotions or {}).keys())
        if not active:
            return None
        focus = meaning.entities[0] if meaning.entities else None
        try:
            return self.apc.observe_and_learn(active, focus=focus)
        except Exception:
            return None

    def predict_next(self, active: set[str], focus: str | None = None) -> set[str]:
        try:
            return self.apc.predict_next_state(active, focus=focus) or set()
        except Exception:
            return set()

    def stats(self) -> dict:
        try:
            return self.apc.stats() or {}
        except Exception:
            return {}
