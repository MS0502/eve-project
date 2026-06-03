"""
IntegratedSelfAdapter

v40 IntegratedSelf (분산 자아 emergent view) ↔ v41.

핵심 원칙: read-only. 어느 모듈도 수정하지 않음.
- snapshot() — 17 차원 자아 view
- thought_stream(n) — 최근 사고 흐름
- goal_alignment_score(action, target) — 행동의 의도 정렬도

학계: Damasio 1999 Protoself, Dennett 1991 Center of Narrative Gravity.

v41 통합:
- /me 에서 활용 (Narrative와 함께)
- decide_action에 goal_alignment_score 가중 (W=0.50, Active Inference EFE)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from integrated_self import IntegratedSelf

from utils.mock_eve import MockEve


class IntegratedSelfAdapter:

    def __init__(self, engine):
        self.engine = engine
        self._mock = MockEve(engine)
        self.is_ = IntegratedSelf(self._mock)

    def snapshot(self):
        try:
            return self.is_.snapshot()
        except Exception:
            return None

    def thought_stream(self, n: int = 10) -> list[dict]:
        try:
            return self.is_.thought_stream(n=n) or []
        except Exception:
            return []

    def goal_alignment_score(self, action: str, target: str | None = None) -> float:
        try:
            return float(self.is_.goal_alignment_score(action, target=target))
        except Exception:
            return 0.0

    def snapshot_phrase(self) -> str:
        snap = self.snapshot()
        if snap is None:
            return "지금 상태를 잘 모르겠어"
        # SelfSnapshot 필드 사용 (있는 만큼)
        feeling = getattr(snap, "feeling", "")
        focus = getattr(snap, "focus", None)
        intimacy = getattr(snap, "user_intimacy", 0.0)
        parts: list[str] = []
        if feeling:
            parts.append(f"기분은 {feeling}")
        if focus:
            parts.append(f"지금 {focus} 생각 중")
        if intimacy and intimacy > 0:
            parts.append(f"민석이랑 {intimacy:.2f}만큼 가까운 느낌")
        if not parts:
            return "그냥 그래"
        return ", ".join(parts)
