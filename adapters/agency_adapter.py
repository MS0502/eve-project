"""
AgencyAdapter

v40 Agency (Haggard 2017 sense-of-agency) ↔ v41 행동 결정.

- evaluate_action(action_category) → {decision: do/hesitate/abstain, confidence, ...}
- attribute_action — 행동 후 self/external 귀속

v41 자율 루프에서:
1. 후보 행동 여러 개 중 evaluate_action으로 do 점수 가장 높은 거 선택
2. 실행 후 attribute_action으로 self 귀속 (자아감)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from agency import Agency


class AgencyAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 sd_adapter=None,
                 dmn_adapter=None,
                 ai_adapter=None,
                 ag: Agency | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if ag is not None:
            self.ag = ag
        else:
            self.ag = Agency(
                hormone_adapter.hs,
                activation_adapter.sa,
                working_memory=activation_adapter.wm,
                self_doubt=sd_adapter.sd if sd_adapter else None,
                dmn=dmn_adapter.dmn if dmn_adapter else None,
                active_inference=ai_adapter.ai if ai_adapter else None,
            )

    def evaluate(self, action_category: str, source: str = "internal") -> dict:
        try:
            r = self.ag.evaluate_action(action_category, source=source) or {}
            # v40 키: 'choice' / 'confidence'. 통일 위해 'decision'도 추가.
            if "decision" not in r and "choice" in r:
                r["decision"] = r["choice"]
            return r
        except Exception:
            return {"decision": "abstain", "choice": "abstain", "confidence": 0.0}

    def best_action(self, candidates: list[str]) -> tuple[str, dict] | None:
        """후보 중 do 점수 가장 높은 거 (do 미달이면 None)."""
        best = None
        best_score = 0.0
        best_eval = None
        for cat in candidates:
            ev = self.evaluate(cat)
            if ev.get("decision") != "do":
                continue
            score = ev.get("confidence", 0.0)
            if score > best_score:
                best, best_score, best_eval = cat, score, ev
        if best is None:
            return None
        return (best, best_eval)

    def attribute(self, action_category: str, source: str = "self"):
        try:
            return self.ag.attribute_action(action_category, source=source)
        except Exception:
            return None
