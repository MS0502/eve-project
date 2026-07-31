"""
GoalAdapter

v40 GoalManagement (Carver & Scheier 1998) ↔ v41 의도-목표 추적.

흐름:
1. meaning.intent == 'command' 또는 명시적 의도 카테고리 → goal_set
2. 활성 목표는 planner sub_points에 약하게 반영 ("~ 하기로 했지")
3. 카테고리 활성 충분히 강하면 자동 완료 (v40 GoalManagement 내부 로직)
"""

from collections.abc import Callable, Mapping
from typing import Any

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
                 gm: GoalManagement | None = None,
                 production_origin_shadow_tap=None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.production_origin_shadow_tap = production_origin_shadow_tap
        if gm is not None:
            self.gm = gm
        else:
            self.gm = GoalManagement(
                activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
            )

    def _decision_epoch(self) -> int:
        value = getattr(self.gm, "tick_count", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    def _run_authoritative_goal_call(
        self,
        *,
        operation_kind: str,
        legacy_goal_code: str,
        source_material: Mapping[str, Any],
        authoritative_call: Callable[[], Any],
    ) -> Any:
        """Run legacy exactly once; shadow comparison is opt-in and non-authoritative."""
        tap = self.production_origin_shadow_tap
        if tap is None:
            return authoritative_call()

        # Lazy import keeps the default engine free of the M3-C-M implementation,
        # pins, state capture, v4 evaluation, and comparison activity.
        from core.m3_c_m_dormant_production_origin_shadow_tap import (
            ProductionGoalOperation,
        )

        operation = ProductionGoalOperation.from_source_material(
            operation_kind=operation_kind,
            legacy_goal_code=legacy_goal_code,
            decision_epoch=self._decision_epoch(),
            source_material=source_material,
        )
        execution = tap.execute_authoritative_once(
            goal_management=self.gm,
            operation=operation,
            authoritative_call=authoritative_call,
        )
        return execution.authoritative_result

    def observe_meaning(self, meaning: Meaning):
        """입력에서 목표 추출."""
        text = meaning.raw_text or ""
        for kw, cat in COMMAND_TO_GOAL.items():
            if kw in text:
                if not self._has_active_goal(cat):
                    try:
                        self._run_authoritative_goal_call(
                            operation_kind="goal_set",
                            legacy_goal_code="legacy_goal_set_command",
                            source_material={
                                "category": cat,
                                "priority": 0.5,
                                "source": "command",
                            },
                            authoritative_call=lambda: self.gm.goal_set(
                                cat,
                                priority=0.5,
                                source="command",
                            ),
                        )
                    except Exception:
                        pass

    def _has_active_goal(self, category: str) -> bool:
        try:
            actives = self.gm.active_goals()
            for g in actives:
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
            self._run_authoritative_goal_call(
                operation_kind="tick",
                legacy_goal_code="legacy_goal_tick",
                source_material={"dt": dt},
                authoritative_call=lambda: self.gm.tick(dt=dt),
            )
        except Exception:
            pass
