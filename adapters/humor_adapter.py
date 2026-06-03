"""
HumorAdapter

v40 Humor (Suls 1972 Incongruity-Resolution) ↔ v41 농담.

흐름:
1. dopamine 임계 이상 + 평이한 입력 (감정 강도 낮음)일 때만 시도
2. generate_joke(seed) → setup/punchline pair
3. sub_points에 짧게 끼움 (실패하면 그냥 일반 응답)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from humor import Humor

from utils.types import Meaning


class HumorAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 humor: Humor | None = None,
                 dopamine_threshold: float = 0.6):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.dopamine_threshold = dopamine_threshold
        if humor is not None:
            self.humor = humor
        else:
            self.humor = Humor(activation_adapter.sa, hormone_adapter.hs)

    def is_eligible(self, meaning: Meaning) -> bool:
        """농담 시도해도 되는 상황인지."""
        # 강한 부정 감정 있으면 농담 X (suffering이 우선)
        for emo, strength in (meaning.emotions or {}).items():
            if emo in ("sadness", "anger", "anxiety", "fatigue") and strength >= 0.5:
                return False
        # DA 충분
        da = self.hormone_adapter.hs.hormones["dopamine"].level
        return da >= self.dopamine_threshold

    def try_joke(self, meaning: Meaning) -> str | None:
        """sub_point용 농담 한 줄. 실패면 None."""
        if not meaning.entities:
            return None
        seed = meaning.entities[0]
        try:
            result = self.humor.generate_joke(seed)
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        setup = result.get("setup")
        punchline = result.get("punchline")
        if setup and punchline:
            return f"{setup}... {punchline}"
        return None
