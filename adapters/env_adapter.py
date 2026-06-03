"""
EnvironmentAdapter

v40 eve_room (SymbolicEnvironment) ↔ v41 공간 컨텍스트.

흐름:
1. EVE 자기 방 인스턴스 (침대/책상/창문/책/일기장 등)
2. 매 chat turn — meaning.context에 location + visible_objects 박음
3. "어디?" "뭐 보여?" 같은 location query → planner가 환경 묘사로 응답
4. /enter, /exit 같은 명령으로 환경 전환 가능 (REPL)

v37.10 외출 자율:
- decide_outing_destination() — 호르몬 기반 외출 욕구 평가
- go_out(destination) — 거실/주방으로 이동 (social_env_adapter 활용)
- return_home() — 자기 방 복귀
- outing_history 기록

v40 eve_room.py 그대로 사용. v40 EnvironmentAdapter (eve 의존)는 안 씀.
"""

import time

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from eve_room import make_eve_room

from utils.types import Meaning


class EnvironmentAdapter:

    def __init__(self,
                 hormone_adapter=None,
                 activation_adapter=None,
                 social_env_adapter=None,
                 room=None,
                 time_fn=time.time):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.social_env_adapter = social_env_adapter
        self.home = room if room is not None else make_eve_room()
        self.current_env = self.home
        self.outing_history: list[dict] = []
        self._time_fn = time_fn
        self._enter_time = self._time_fn()

    # ===== meaning에 환경 컨텍스트 박음 =====

    def enrich_meaning(self, meaning: Meaning):
        meaning.context = meaning.context or {}
        meaning.context["location"] = self.current_env.name
        meaning.context["visible_objects"] = list(self.current_env.objects.keys())
        meaning.context["is_home"] = (self.current_env is self.home)
        # NPC 노출 (외출 중일 때)
        if (not meaning.context["is_home"]
                and self.social_env_adapter is not None):
            npcs = self.social_env_adapter.get_npcs(self.current_env.name)
            if npcs:
                meaning.context["npcs"] = npcs

    # ===== location query 인식 =====

    LOCATION_QUERY_KEYWORDS = ("어디", "뭐 보여", "뭐 보이", "지금 어디", "어딨")

    def is_location_query(self, meaning: Meaning) -> bool:
        text = meaning.raw_text or ""
        return any(kw in text for kw in self.LOCATION_QUERY_KEYWORDS)

    def describe_location(self) -> str:
        env = self.current_env
        objs = list(env.objects.keys())
        # NPC 있으면 같이 묘사
        npcs = []
        if self.social_env_adapter is not None and not (env is self.home):
            npcs = self.social_env_adapter.get_npcs(env.name)
        if not objs and not npcs:
            return f"지금 {env.name}이야"
        parts = [f"지금 {env.name}이야."]
        if objs:
            obj_str = ", ".join(objs[:3]) + (" 등" if len(objs) > 3 else "")
            parts.append(f"{obj_str}이 보여.")
        if npcs:
            npc_str = ", ".join(npcs[:3])
            parts.append(f"{npc_str}이 있어.")
        return " ".join(parts)

    # ===== 외출/복귀 =====

    def go_out(self, destination: str) -> bool:
        """destination(거실/주방)으로 외출. 성공 여부 반환."""
        if self.social_env_adapter is None:
            return False
        env = self.social_env_adapter.get_or_make(destination)
        if env is None:
            return False
        from_env = self.current_env.name
        self.outing_history.append({
            "from": from_env, "to": destination,
            "time": self._time_fn(), "kind": "outing",
        })
        self.current_env = env
        self._enter_time = self._time_fn()
        return True

    def return_home(self) -> bool:
        """자기 방 복귀."""
        if self.current_env is self.home:
            return False
        self.outing_history.append({
            "from": self.current_env.name, "to": self.home.name,
            "time": self._time_fn(), "kind": "return",
        })
        self.current_env = self.home
        self._enter_time = self._time_fn()
        return True

    def is_home(self) -> bool:
        return self.current_env is self.home

    def dwell_seconds(self) -> float:
        return self._time_fn() - self._enter_time

    # ===== 자율 외출 결정 =====

    def decide_outing_destination(self) -> str | None:
        """
        호르몬 상태 보고 외출 갈지 결정. 안 가면 None.
        
        조건:
        - ot 결핍 + 사회 욕구 → 거실 (NPC 있는 곳)
        - 자기 방 오래 머물렀음 (>5분) + cort 낮음 → 거실/주방
        - 이미 외출 중이면 외출 안 함
        """
        if not self.is_home():
            return None
        if self.hormone_adapter is None:
            return None
        hs = self.hormone_adapter.hs
        ot = hs.hormones["oxytocin"].level
        cort = hs.hormones["cortisol"].level
        dwell = self.dwell_seconds()

        # ot 낮고 cort 안 높으면 거실 (사회 연결 추구)
        if ot < 0.25 and cort < 0.5 and dwell > 300:
            return "거실"
        # 너무 오래 방에만 + cort 낮음 → 주방
        if dwell > 600 and cort < 0.4:
            return "주방"
        return None

    def decide_return_home(self) -> bool:
        """외출 중일 때 복귀 갈지. 너무 오래 외출 + 피로 누적이면 True."""
        if self.is_home():
            return False
        if self.hormone_adapter is None:
            return False
        hs = self.hormone_adapter.hs
        cort = hs.hormones["cortisol"].level
        dwell = self.dwell_seconds()
        # 외출 5분 + cort 높으면 복귀
        if dwell > 300 and cort > 0.55:
            return True
        # 외출 10분 넘으면 무조건 복귀 (자율 한계)
        if dwell > 600:
            return True
        return False

    # ===== 환경 전환 (low-level) =====

    def enter(self, room):
        """다른 SymbolicEnvironment로 이동 (수동)."""
        self.current_env = room
        self._enter_time = self._time_fn()

    def observe(self) -> dict:
        try:
            return self.current_env.observe()
        except Exception:
            return {}

    def get_objects(self) -> list[str]:
        return list(self.current_env.objects.keys())

    def location_name(self) -> str:
        return self.current_env.name
