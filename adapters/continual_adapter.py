"""
ContinualAdapter

v40 ContinualRehearsal (Schapiro 2025) ↔ v41 정체성 보존.

흐름:
1. 보호 카테고리 등록 ("민석", "EVE", 사용자 이름 등) — 잊혀지지 않게
2. 매 tick — 자동 rehearse (활성 약하게 줘서 유지)
3. detect_drift — 정체성 변화 감지

v41 통합:
- streaming engine.tick(dt) 호출 시 cr.tick 같이
- 핵심 카테고리는 user_presence (김민석) 기반 자동 등록
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from continual_rehearsal import ContinualRehearsal


# 시작부터 보호되는 핵심 카테고리 (정체성)
DEFAULT_PROTECTED_CATEGORIES = [
    "민석",        # 사용자 이름
    "EVE",         # 자기 이름
    "친구",        # 관계
    "대화",        # 핵심 활동
]


class ContinualAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 cr: ContinualRehearsal | None = None,
                 protected: list[str] | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if cr is not None:
            self.cr = cr
        else:
            self.cr = ContinualRehearsal(
                activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
                working_memory=activation_adapter.wm,
            )
        # 핵심 보호 카테고리 자동 등록
        for cat in (protected if protected is not None else DEFAULT_PROTECTED_CATEGORIES):
            self.mark_protected(cat)

    def mark_protected(self, category: str):
        try:
            if hasattr(self.cr, "mark_protected"):
                self.cr.mark_protected(category)
            elif hasattr(self.cr, "protect"):
                self.cr.protect(category)
        except Exception:
            pass

    def tick(self, dt: float = 1.0):
        """매 턴/틱마다 호출. 보호 카테고리 약하게 강화."""
        try:
            self.cr.tick(dt) if hasattr(self.cr, "tick") else None
            if hasattr(self.cr, "rehearse_now"):
                self.cr.rehearse_now()
        except Exception:
            pass

    def detect_drift(self) -> dict | None:
        try:
            if hasattr(self.cr, "detect_drift"):
                return self.cr.detect_drift()
        except Exception:
            pass
        return None

    def protected_categories(self) -> set[str]:
        try:
            if hasattr(self.cr, "get_protected_categories"):
                return set(self.cr.get_protected_categories())
            return set(getattr(self.cr, "protected_categories", set()))
        except Exception:
            return set()
