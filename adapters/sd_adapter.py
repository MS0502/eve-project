"""
SDAdapter

v40 SelfDoubt ↔ v41 System2 강화.

v40 SD의 강점:
- doubt_level 0~1 (5종 신호 합성)
- recommended_action: accept / question / defer / reject
- 호르몬 5종 변조 (cort/NE↑ doubt↑, OT/5HT/DA↑ doubt↓)

System2 통합:
- 각 후보에 대해 evaluate_claim 호출
- recommended_action == "reject" 면 후보 제거
- "question"이면 점수 -0.2
- "defer"면 점수 -0.1
- "accept"면 점수 +0.05
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from self_doubt import SelfDoubt


ACTION_PENALTY = {
    "accept":   +0.05,
    "question": -0.20,
    "defer":    -0.10,
    "reject":   None,    # 제거
}


class SDAdapter:

    def __init__(self,
                 hormone_adapter=None,
                 activation_adapter=None,
                 sd: SelfDoubt | None = None):
        if sd is not None:
            self.sd = sd
        else:
            from spreading_activation import SpreadingActivation
            from hormone_system import HormoneSystem
            sa = activation_adapter.sa if activation_adapter else SpreadingActivation()
            hs = hormone_adapter.hs if hormone_adapter else HormoneSystem()
            self.sd = SelfDoubt(hs, sa)

    def filter_candidates(self, candidates: list[dict]) -> list[dict]:
        """후보들에 대해 신념 검증. reject는 제거, 나머지는 점수 조정."""
        out: list[dict] = []
        for c in candidates:
            try:
                ev = self.sd.evaluate_claim(c.get("text", ""))
            except Exception:
                out.append(c)
                continue

            action = ev.get("recommended_action", "accept")
            penalty = ACTION_PENALTY.get(action, 0.0)
            if penalty is None:
                # reject — 후보 제거
                continue
            new_c = dict(c)
            new_c["score"] = max(0.0, new_c.get("score", 0.0) + penalty)
            new_c["sd_action"] = action
            new_c["sd_doubt"] = ev.get("doubt_level", 0.0)
            out.append(new_c)
        return out
