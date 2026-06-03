"""
SocialEnvAdapter

v40 SocialEnvironment ↔ v41 NPC 환경.

핵심 디자인:
- 거실/주방 = "외출 갈 수 있는 곳" (UserPresence와 별개)
- NPC = 시뮬 친구 (진수/엄마/손님 등) — 김민석 = 진짜 와 명확 구분
- 외출 시 자기 방 → 거실/주방 일시 이동, 일정 시간 후 복귀

v41 통합:
- env_adapter.outing(destination) — 환경 전환
- env_adapter.return_home() — 자기 방 복귀
- 외출 중엔 NPC와 상호작용 가능
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from social_env import (
    SocialEnvironment,
    make_social_living_room,
    make_first_meeting_env,
)
from symbolic_env import make_kitchen_env


# 외출 가능 목적지
OUTING_FACTORIES = {
    "거실": make_social_living_room,
    "주방": make_kitchen_env,
}


class SocialEnvAdapter:

    def __init__(self):
        # 미리 만들어 둠 (반복 외출에도 NPC 상태 유지)
        self.environments: dict[str, SocialEnvironment | object] = {}

    def get_or_make(self, name: str):
        """name에 해당하는 환경 인스턴스 반환. 없으면 lazy 생성."""
        if name in self.environments:
            return self.environments[name]
        factory = OUTING_FACTORIES.get(name)
        if factory is None:
            return None
        env = factory()
        self.environments[name] = env
        return env

    def list_destinations(self) -> list[str]:
        return list(OUTING_FACTORIES.keys())

    def get_npcs(self, env_name: str) -> list[str]:
        env = self.environments.get(env_name)
        if env is None:
            return []
        if hasattr(env, "npcs"):
            return list(env.npcs.keys())
        return []
