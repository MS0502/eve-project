"""
ActivationAdapter

v40 SpreadingActivation + WorkingMemory ↔ v41 System1 후보 확장.

흐름:
1. v41 LanguageUnderstanding이 Meaning 뽑음 (entities/emotions)
2. 어댑터가 그 카테고리들을 SA.activate + WM.add
3. SA.spread()로 연관 카테고리까지 활성 퍼짐
4. System1이 어댑터의 top_active() 호출 → 후보 풀 확장

v40 코드는 안 건드림.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from spreading_activation import SpreadingActivation
from working_memory import WorkingMemory

from utils.types import Meaning


class ActivationAdapter:

    def __init__(self, sa: SpreadingActivation | None = None,
                 wm: WorkingMemory | None = None,
                 hormone_adapter=None):
        self.sa = sa if sa is not None else SpreadingActivation()
        self.wm = wm if wm is not None else WorkingMemory()
        self.hormone_adapter = hormone_adapter

    # ===== Meaning 들어올 때 호출 =====

    def ingest(self, meaning: Meaning):
        """v41 의미 → SA/WM 활성화."""
        # 0) 직전 턴 잔존 활성 자연 감쇠 (시간 흐름 반영)
        try:
            self.sa.decay(dt=1.0)
        except Exception:
            pass
        try:
            self.wm.decay(dt=1.0)
        except Exception:
            pass

        # 호르몬 변조 적용 (있으면)
        if self.hormone_adapter is not None:
            try:
                self.sa.apply_hormone_modulation(self.hormone_adapter.hs)
                self.wm.apply_hormone_state(self.hormone_adapter.hs)
            except Exception:
                pass

        # entity 카테고리화
        for ent in meaning.entities:
            self.sa.activate(ent, strength=0.5)
            self.wm.add(ent, salience=0.5)

        # 감정도 카테고리 (강도 = 감정 강도)
        for emo, strength in meaning.emotions.items():
            self.sa.activate(emo, strength=strength)
            self.wm.add(emo, salience=strength)

        # action도
        for act in meaning.actions:
            self.sa.activate(act, strength=0.4)

        # 활성 전파
        self.sa.spread(steps=1)

    # ===== System1이 호출 =====

    def top_active(self, n: int = 8) -> list[tuple[str, float]]:
        # VSA 합성 키 ('+', '@' 포함)는 사용자에게 노출 안 함
        out: list[tuple[str, float]] = []
        for cat, lvl in self.sa.get_top_active(n=n * 2):
            if "+" in cat or "@" in cat:
                continue
            out.append((cat, lvl))
            if len(out) >= n:
                break
        return out

    def focus_category(self) -> str | None:
        try:
            return self.wm.get_focus()
        except Exception:
            return None

    def focus_set(self) -> set[str]:
        try:
            return self.wm.get_focus_set()
        except Exception:
            return set()

    def all_active_set(self, threshold: float = 0.2) -> set[str]:
        """현재 활성 카테고리 전체 (cue로 EM에 던지기용)."""
        return {cat for cat, lvl in self.sa.get_top_active(n=20) if lvl >= threshold}

    # ===== 학습 =====

    def learn_pair(self, a: str, b: str, strength: float = 0.4):
        if hasattr(self.sa, "learn_pair"):
            self.sa.learn_pair(a, b, strength=strength)

    def tick(self, dt: float = 1.0):
        if hasattr(self.sa, "decay"):
            self.sa.decay(dt)
        if hasattr(self.wm, "decay"):
            self.wm.decay(dt)
