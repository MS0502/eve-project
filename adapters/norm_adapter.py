"""
NormAdapter

v40 NormInternalization (Niazi & Hutter 2026) ↔ v41 규범 학습.

흐름:
1. meaning.intent == 'command' → observe_command
2. 같은 명령 누적 → internalize (자기 규범으로 흡수)
3. 활성 규범은 planner에서 약하게 반영 (introspect)

이게 v41의 진짜 학습 모듈. 같은 입력 패턴 반복되면
EVE 행동 양식이 변함 (호르몬 baseline, DMN 모드 가중치 등).
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from norm_internal import NormInternalization

from utils.types import Meaning


class NormAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 nl_adapter=None,
                 dmn_adapter=None,
                 norm: NormInternalization | None = None,
                 default_agent: str = "민석"):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.dmn_adapter = dmn_adapter

        if norm is not None:
            self.norm = norm
        else:
            nl_inst = nl_adapter.nl if nl_adapter is not None else None
            dmn_inst = dmn_adapter.dmn if dmn_adapter is not None else None
            self.norm = NormInternalization(
                activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
                natural_lang=nl_inst,
                dmn=dmn_inst,
                default_agent=default_agent,
            )

    def observe_meaning(self, meaning: Meaning):
        """입력이 명령이면 norm에 노출."""
        if meaning.intent != "command":
            return None
        try:
            return self.norm.observe_command(meaning.raw_text)
        except Exception:
            return None

    def active_norms(self) -> list[dict]:
        try:
            norms = self.norm.get_active_norms()
            # legacy 형식 통일
            return [dict(n) if isinstance(n, dict) else getattr(n, "__dict__", {}) for n in norms]
        except Exception:
            return []

    def internalized_count(self) -> int:
        return getattr(self.norm, "internalized_count", 0)

    def apply_to_dmn(self):
        """규범이 DMN 모드 가중치에 영향 주게."""
        try:
            return self.norm.apply_norm_to_dmn()
        except Exception:
            return None
