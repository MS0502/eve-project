"""
HormoneAdapter

v40 HormoneSystem(26종 호르몬) ↔ v41 응답 계획 연결.

매핑:
- mood = compute_mood() = {valence, arousal, dominance}
- (valence, arousal) 사분면 → tone 추천
- 에너지(dopamine + norepinephrine) → sub_points 길이
- oxytocin/cortisol → stage1 망설임 표현

v40 코드는 한 줄도 안 건드림. 인스턴스 받아서 query만.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from hormone_system import HormoneSystem  # legacy/eve_modules/


class HormoneAdapter:

    def __init__(self, hs: HormoneSystem | None = None):
        # None이면 자동 생성 (v41 단독 사용 시)
        self.hs = hs if hs is not None else HormoneSystem()

    # ===== v41 GlobalWorkspace 호환 =====

    def as_dict(self) -> dict[str, float]:
        """v41 hormone dict 호환 (stress/energy/curiosity 외에 26종 다 포함)."""
        d = {name: h.level for name, h in self.hs.hormones.items()}
        # v41 표준 키 보강
        d["stress"] = self.hs.hormones["cortisol"].level
        d["energy"] = (self.hs.hormones["dopamine"].level
                       + self.hs.hormones["norepinephrine"].level) / 2
        d["curiosity"] = self.hs.hormones["dopamine"].level
        return d

    # ===== Planner가 부르는 핵심 헬퍼 =====

    def mood(self) -> dict[str, float]:
        return self.hs.compute_mood()

    def recommend_tone(self, base_emotion: str | None) -> str:
        """
        감정에서 정해진 base tone을 호르몬으로 보정.

        규칙:
        - cortisol 매우 높음 + ot 낮음 → calm 으로 강제 (진정시킴)
        - oxytocin 높음 → warm 으로 끌어당김
        - valence 양 + arousal 낮음 + 감정 없음 → warm
        - valence 양 + arousal 높음 → curious
        - 그 외 → base 유지
        """
        m = self.mood()
        cort = self.hs.hormones["cortisol"].level
        ot = self.hs.hormones["oxytocin"].level

        # 강한 변조 (감정 base 무시)
        if cort > 0.7 and ot < 0.3:
            return "calm"
        if ot > 0.7 and cort < 0.4:
            return "warm"

        # base 있으면 우선
        if base_emotion:
            from language.planner import EMOTION_PLAN
            spec = EMOTION_PLAN.get(base_emotion)
            if spec and "tone" in spec:
                return spec["tone"]

        # base 없을 때 mood로 결정
        v, a = m["valence"], m["arousal"]
        if v > 0.2 and a < 0.5:
            return "warm"
        if v > 0.2 and a >= 0.55:
            return "curious"
        if v < -0.1:
            return "calm"
        return "neutral"

    def stage1_word(self, base_word: str) -> str:
        """망설임 표현을 호르몬으로 살짝 변조."""
        cort = self.hs.hormones["cortisol"].level
        ot = self.hs.hormones["oxytocin"].level
        if cort > 0.7:
            return "잠깐,"
        if ot > 0.7:
            return "어..."
        return base_word

    def sub_point_budget(self) -> int:
        """에너지 낮으면 말 짧아짐 (sub_points 개수 상한)."""
        energy = (self.hs.hormones["dopamine"].level
                  + self.hs.hormones["norepinephrine"].level) / 2
        if energy < 0.25:
            return 1
        if energy < 0.45:
            return 2
        return 5  # 사실상 무제한 (planner가 더 적게 만들면 그게 우선)

    # ===== Workspace에서 호르몬 갱신할 때 호출 =====

    def tick(self, dt: float = 1.0):
        self.hs.update(dt)

    def stimulate(self, hormone_name: str, delta: float):
        """단순 자극 (테스트용)."""
        if hormone_name in self.hs.hormones:
            h = self.hs.hormones[hormone_name]
            h.level = max(0.0, min(1.0, h.level + delta))
