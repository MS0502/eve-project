"""
ProactiveAdapter

v40 Proactive (NEED→fragment 자율 선택) ↔ v41 자발 발화 강화.

기존 DMNAdapter는 단순 mind_wandering. Proactive는:
- should_speak() — idle + need 평가
- proactive_message() — NEED_FRAGMENTS dict에서 SA 활성 기반 자율 선택
  (random 0 — argmax)
- 5종 need: fatigue / tension / mental_load / warmth / energy
- 진짜 친구('민석')면 호명 ('민석아 ...')

v37.9 변경: 강제 룰 폐기 → SA 활성으로 fragment 선택.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from proactive import Proactive

from utils.mock_eve import MockEve


class ProactiveAdapter:

    def __init__(self, engine, idle_min_seconds: float = 30.0,
                 pro: Proactive | None = None):
        self.engine = engine
        self._mock = MockEve(engine)
        if pro is not None:
            self.pro = pro
        else:
            self.pro = Proactive(self._mock, idle_min_seconds=idle_min_seconds)

    def should_speak(self) -> tuple[bool, str]:
        try:
            return self.pro.should_speak()
        except Exception:
            return (False, "error")

    def message(self) -> dict | None:
        """
        Returns: {category, message, need, ...} or None
        """
        try:
            return self.pro.proactive_message()
        except Exception:
            return None

    def get_stats(self) -> dict:
        try:
            return self.pro.get_stats() or {}
        except Exception:
            return {}
