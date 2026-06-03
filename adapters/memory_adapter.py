"""
MemoryAdapter

v40 EpisodicMemory ↔ v41 일화 회상.

흐름:
1. 대화 끝날 때마다 자동 encode (SAWM의 활성 카테고리를 카테고리로)
2. 새 입력 들어오면 같은 활성 카테고리로 recall → planner 가 sub_points 앞에 추가

v40 EM은 sa, wm, hs 의존 → adapter 들도 같이 받음.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from episodic import EpisodicMemory, Episode

from utils.types import Meaning


class MemoryAdapter:

    def __init__(self,
                 activation_adapter=None,
                 hormone_adapter=None,
                 em: EpisodicMemory | None = None):
        # v40 EM은 sa, wm, hs 다 필요. 어댑터에서 빌려옴.
        if em is not None:
            self.em = em
        else:
            from spreading_activation import SpreadingActivation
            from working_memory import WorkingMemory
            from hormone_system import HormoneSystem
            sa = activation_adapter.sa if activation_adapter else SpreadingActivation()
            wm = activation_adapter.wm if activation_adapter else WorkingMemory()
            hs = hormone_adapter.hs if hormone_adapter else HormoneSystem()
            self.em = EpisodicMemory(sa, wm, hs)
        self.activation_adapter = activation_adapter

    # ===== 회상 (planner가 부름) =====

    def recall_for(self, meaning: Meaning, top_n: int = 2) -> list[Episode]:
        """현재 의미와 관련된 과거 일화."""
        cue = self._build_cue(meaning)
        if not cue:
            return []
        try:
            return self.em.recall(cue, top_n=top_n, min_overlap=1)
        except Exception:
            return []

    def _build_cue(self, meaning: Meaning) -> set[str]:
        # 직접 입력에서만 cue 뽑음. 잔존 활성에 끌려가지 않게.
        cue: set[str] = set(meaning.entities or [])
        cue |= set((meaning.emotions or {}).keys())
        return cue

    # ===== 인코딩 (engine이 대화 끝에 호출) =====

    def encode_turn(self, meaning: Meaning, response: str):
        """이번 턴을 일화로 저장 (카테고리는 직접 입력만)."""
        cats: set[str] = set(meaning.entities or [])
        cats |= set((meaning.emotions or {}).keys())
        if not cats:
            return None

        summary = f"{meaning.raw_text} → {response[:60]}"
        try:
            return self.em.encode(
                categories=cats,
                summary=summary,
                tuple_data={"input": meaning.raw_text, "response": response},
                salience=0.5,
            )
        except Exception:
            return None

    # ===== 회상 → sub_points 변환 (planner 헬퍼) =====

    def recall_phrases(self, meaning: Meaning, max_n: int = 1) -> list[str]:
        """
        과거 일화 → 자연어 sub_point 한 줄로.
        '전에 ~ 얘기 했었잖아' 형태.
        """
        episodes = self.recall_for(meaning, top_n=max_n)
        out: list[str] = []
        for ep in episodes:
            # 가장 핵심 카테고리 하나 뽑음 (현재 cue와 겹치는 것 우선)
            cats = list(ep.categories)
            if not cats:
                continue
            # 현재 입력의 entity와 겹치는 거 우선
            overlap = [c for c in cats if c in (meaning.entities or [])]
            key = overlap[0] if overlap else cats[0]
            out.append(f"전에 {key} 얘기 했던 거 떠오르네")
        return out
