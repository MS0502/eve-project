"""
DMNAdapter

v40 DefaultModeNetwork ↔ v41 자발 발화.

v40 DMN의 강점:
- 외부 입력 없을 때 카테고리 자발 활성
- 4모드: mind_wandering / memory_recall / self_referential / self_intent
- 호르몬 기반 강도 변조

흐름:
1. tick(dt) — DMN 진행 (호출 측이 시간 흘림 결정)
2. try_spontaneous() → DMN이 카테고리 활성했으면 가짜 Meaning 반환
3. StreamingEngine.proactive_stream() — 그 Meaning으로 일반 응답 흐름 탐
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from dmn import DefaultModeNetwork

from utils.types import Meaning


class DMNAdapter:

    def __init__(self,
                 hormone_adapter,                   # 필수 (DMN이 호르몬 의존)
                 activation_adapter,                 # 필수 (sa, wm 공유)
                 seed_commonsense: bool = True,
                 dmn: DefaultModeNetwork | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter

        if dmn is not None:
            self.dmn = dmn
        else:
            self.dmn = DefaultModeNetwork(
                activation_adapter.sa,
                activation_adapter.wm,
                hormone_adapter.hs,
            )

        # DMN이 카테고리 그래프 기반이라 사전 연결 시드 필요
        if seed_commonsense:
            self._seed_commonsense()

    def _seed_commonsense(self):
        """COMMONSENSE_TRIPLES → SA Hebbian pair (DMN이 후보 만들 그래프)."""
        try:
            from commonsense_seed import COMMONSENSE_TRIPLES, HEBBIAN_RELATIONS
        except Exception:
            return
        sa = self.activation_adapter.sa
        for subj, rel, obj, weight in COMMONSENSE_TRIPLES:
            if rel in HEBBIAN_RELATIONS:
                try:
                    sa.learn_pair(subj, obj, strength=weight)
                except Exception:
                    pass

    # ===== 시간 흘림 =====

    def tick(self, dt: float = 1.0):
        """DMN 시간 흐름. 자발 활성이 일어나면 결과 반환 가능."""
        return self.dmn.tick(dt=dt)

    def force_spontaneous(self) -> dict | None:
        """테스트/proactive 트리거 — 강제로 한 번 자발 활성."""
        return self.dmn.activate_one(force=True)

    # ===== 자발 활성 → v41 Meaning 변환 =====

    def try_spontaneous_meaning(self, force: bool = False) -> Meaning | None:
        """
        DMN이 활성한 카테고리를 v41 Meaning으로 변환.
        proactive 응답 흐름에 일반 chat_stream처럼 흘려보내기 위함.
        """
        result = self.dmn.activate_one(force=force)
        if not result:
            return None

        cat = result.get("category")
        mode = result.get("mode", "mind_wandering")
        if not cat:
            return None

        # 가짜 Meaning 만듦 (raw_text는 비움 — proactive 표시)
        meaning = Meaning(
            intent="proactive",
            entities=[cat],
            actions=[],
            emotions={},
            context={
                "proactive": True,
                "dmn_mode": mode,
                "dmn_category": cat,
            },
            raw_text=f"[자발: {cat}]",
        )
        return meaning

    def inner_voice(self) -> str | None:
        try:
            return self.dmn.inner_voice()
        except Exception:
            return None
