"""
CreativeAdapter

v40 Creative (Wallas 1926, Mednick 1962 RAT, Fauconnier-Turner blending)
↔ v41 창의적 결합.

흐름:
1. DA 충분 + cort 낮을 때 incubate(seed)
2. 일정 시간 지난 후 harvest → 'aha' 경험 (concept blend)
3. blend 결과는 sub_point 로 표시 가능 (새로 발견한 카테고리 등)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from creative import Creative

from utils.types import Meaning


class CreativeAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 memory_adapter=None,
                 dmn_adapter=None,
                 creative: Creative | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if creative is not None:
            self.cr = creative
        else:
            em = memory_adapter.em if memory_adapter else None
            dmn = dmn_adapter.dmn if dmn_adapter else None
            self.cr = Creative(
                activation_adapter.sa,
                hormone_adapter.hs,
                episodic_memory=em,
                dmn=dmn,
                working_memory=activation_adapter.wm,
            )

    def maybe_incubate(self, meaning: Meaning) -> bool:
        """입력의 첫 entity 보고 incubate 시도. 호르몬 안 맞으면 자동 거절."""
        if not meaning.entities:
            return False
        try:
            return self.cr.incubate(meaning.entities[0])
        except Exception:
            return False

    def harvest_phrase(self) -> str | None:
        """incubate된 거 익으면 한 줄 표현."""
        try:
            illum = self.cr.harvest_incubation()
        except Exception:
            return None
        if not illum:
            return None
        # illum은 list of dict일 수도, dict일 수도
        first = illum[0] if isinstance(illum, list) and illum else illum
        if isinstance(first, dict):
            seed = first.get("seed", "")
            blend = first.get("blend") or first.get("result")
            if blend:
                return f"{seed} 생각하다가 {blend}이 떠올랐어"
        return None

    def blend(self, a: str, b: str) -> dict | None:
        try:
            return self.cr.blend_concepts(a, b)
        except Exception:
            return None
