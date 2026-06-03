"""
GoalAdapter

v40 GoalManagement (Carver & Scheier 1998) ↔ v41 의도-목표 추적.

흐름:
1. meaning.intent == 'command' 또는 명시적 의도 카테고리 → goal_set
2. 활성 목표는 planner sub_points에 약하게 반영 ("~ 하기로 했지")
3. 카테고리 활성 충분히 강하면 자동 완료 (v40 GoalManagement 내부 로직)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from goal_management import GoalManagement

from utils.types import Meaning


# 의도/명령 동사 → 목표 카테고리 매핑
COMMAND_TO_GOAL: dict[str, str] = {
    "쉬다": "쉬다", "쉬어": "쉬다",
    "공부": "공부", "공부하": "공부",
    "운동": "운동", "운동하": "운동",
    "먹다": "먹다", "먹어": "먹다",
    "자다": "자다", "자야": "자다",
    "PT":  "PT",
}


class GoalAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 gm: GoalManagement | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if gm is not None:
            self.gm = gm
        else:
            self.gm = GoalManagement(
                activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
            )

    def observe_meaning(self, meaning: Meaning):
        """입력에서 목표 추출."""
        # 1) 명령/의도형 텍스트에서 키워드 매칭
        text = meaning.raw_text or ""
        for kw, cat in COMMAND_TO_GOAL.items():
            if kw in text:
                if not self._has_active_goal(cat):
                    try:
                        self.gm.goal_set(cat, priority=0.5, source="command")
                    except Exception:
                        pass

    def _has_active_goal(self, category: str) -> bool:
        try:
            actives = self.gm.active_goals()
            for g in actives:
                # active_goals는 dict 또는 list[Goal]일 수 있음
                gc = g.get("category") if isinstance(g, dict) else getattr(g, "category", None)
                if gc == category:
                    return True
        except Exception:
            pass
        return False

    def active_goal_categories(self) -> list[str]:
        """현재 활성 목표 카테고리들."""
        try:
            actives = self.gm.active_goals()
            cats: list[str] = []
            for g in actives:
                gc = g.get("category") if isinstance(g, dict) else getattr(g, "category", None)
                if gc:
                    cats.append(gc)
            return cats
        except Exception:
            return []

    def goal_phrases(self, max_n: int = 1) -> list[str]:
        """planner sub_points에 끼울 자연어 목표 표현."""
        cats = self.active_goal_categories()[:max_n]
        return [f"{c} 하기로 한 거 생각났어" for c in cats]

    def tick(self, dt: float = 1.0):
        try:
            self.gm.tick(dt=dt)
        except Exception:
            pass
