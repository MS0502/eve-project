"""
MultiStreamAdapter

v40 MultiStream ↔ v41 평행 사고 흐름.

흐름:
1. emotion / goal / memory / dmn / nl 등 5가지 스트림 동시 진행
2. tick(dt) — 모든 스트림 한 스텝
3. compete_for_focus — 가장 강한 스트림이 focus 차지
4. cross_talk(focus) — 스트림간 정보 공유

v41 통합:
- 응답 후 tick → 백그라운드 사고 진행
- Workspace focus는 MS의 winner 카테고리 (Globalbroadcast)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from multi_stream import MultiStream


class MultiStreamAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 dmn_adapter=None,
                 memory_adapter=None,
                 tm_adapter=None,
                 goal_adapter=None,
                 nl_adapter=None,
                 ms: MultiStream | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if ms is not None:
            self.ms = ms
        else:
            self.ms = MultiStream(
                activation_adapter.sa,
                hormone_adapter.hs,
                activation_adapter.wm,
                dmn=dmn_adapter.dmn if dmn_adapter else None,
                episodic_memory=memory_adapter.em if memory_adapter else None,
                temporal=tm_adapter.tm if tm_adapter else None,
                goal_management=goal_adapter.gm if goal_adapter else None,
                natural_lang=nl_adapter.nl if nl_adapter else None,
            )

    def tick(self, dt: float = 0.5):
        try:
            return self.ms.tick(dt=dt)
        except Exception:
            return None

    def compete(self) -> str | None:
        try:
            r = self.ms.compete_for_focus()
            if isinstance(r, dict):
                return r.get("winner") or r.get("focus")
            return r
        except Exception:
            return None

    def cross_talk(self, focus_category: str):
        try:
            self.ms.cross_talk(focus_category)
        except Exception:
            pass
