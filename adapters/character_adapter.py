"""
CharacterAdapter — 라운드 25

EVE 호르몬/행동/위치 → 캐릭터 표정 + 모션.

원칙:
- 결정론 (호르몬 조합 → 결정적 표정)
- 백엔드는 라벨만 제공 (구체 그림은 프론트가)
- EVE 정체성 (눈빛 빛, 보디 추상)

표정 10종 (Damasio 1999 기본 감정):
  neutral, happy, sad, angry, surprised,
  sleepy, shy, warm, cold, doubtful

모션 7종:
  idle, walking, sitting, lying, reading, looking_around, sleeping
"""

from typing import Optional


# 표정 결정 — 호르몬 조합 (실제 호르몬 키 사용)
EXPRESSION_RULES = [
    # (조건, 표정)
    # 졸림 (멜라토닌 높음 또는 adenosine 높음)
    (lambda h: h.get("melatonin", 0) > 0.6 or h.get("adenosine", 0) > 0.7, "sleepy"),
    # 화남/공격 (cort + norepinephrine 둘 다 높음)
    (lambda h: h.get("cortisol", 0) > 0.7 and h.get("norepinephrine", 0) > 0.5, "angry"),
    # 슬픔 (cort 높고 dop 낮음)
    (lambda h: h.get("cortisol", 0) > 0.5 and h.get("dopamine", 0) < 0.3, "sad"),
    # 기쁨 (dop + ot 높음)
    (lambda h: h.get("dopamine", 0) > 0.6 and h.get("oxytocin", 0) > 0.5, "happy"),
    # 따뜻함 (ot 높음)
    (lambda h: h.get("oxytocin", 0) > 0.6, "warm"),
    # 놀람/호기심 (norepinephrine 높음 + acetylcholine 높음)
    (lambda h: h.get("norepinephrine", 0) > 0.6 and h.get("acetylcholine", 0) > 0.5, "surprised"),
    # 차가움 (ot 낮고 cort 약간)
    (lambda h: h.get("oxytocin", 0) < 0.2 and h.get("cortisol", 0) > 0.4, "cold"),
    # 의심 (cort 약간 + dop 낮음 + glutamate 높음 — 인지 부하)
    (lambda h: h.get("cortisol", 0) > 0.4 and h.get("dopamine", 0) < 0.4 and h.get("glutamate", 0) > 0.5, "doubtful"),
    # 부끄러움 (ot 약간 + 약한 norepinephrine)
    (lambda h: 0.3 < h.get("oxytocin", 0) < 0.6 and h.get("norepinephrine", 0) > 0.3, "shy"),
]

DEFAULT_EXPRESSION = "neutral"


# 모션 결정 — 행동 + 위치 + 호르몬
MOTION_BY_ACTION = {
    "외출_거실": "walking",
    "외출_주방": "walking",
    "복귀_내방": "walking",
    "잠자기": "sleeping",
    "책_읽기": "reading",
    "일기_쓰기": "sitting",
    "둘러보기": "looking_around",
    "휴식": "sitting",
    "탐색": "walking",
    "학습": "sitting",
    "영상_보기": "sitting",
}


class CharacterAdapter:

    def __init__(self, engine):
        self.engine = engine
        # 마지막 행동 기억 (행동 끝나면 idle로 돌아감)
        self._last_action = None
        self._last_action_time = 0.0
        # 입모양 상태
        self._is_speaking = False

    # ===== 표정 =====

    def expression(self) -> str:
        """현재 호르몬 → 표정 라벨."""
        try:
            hormones = {}
            for name, h in self.engine.hormone_adapter.hs.hormones.items():
                hormones[name] = float(h.level)
            # 첫 매칭 룰
            for cond, expr in EXPRESSION_RULES:
                try:
                    if cond(hormones):
                        return expr
                except Exception:
                    continue
            return DEFAULT_EXPRESSION
        except Exception:
            return DEFAULT_EXPRESSION

    # ===== 모션 =====

    def motion(self) -> str:
        """현재 행동/위치 → 모션 라벨."""
        # 1. 잠자기 (멜라토닌 매우 높음)
        try:
            mel = self.engine.hormone_adapter.hs.hormones.get("melatonin")
            if mel and mel.level > 0.8:
                return "sleeping"
        except Exception:
            pass

        # 2. 최근 행동 (autonomous_loop history 마지막)
        try:
            if self.engine.autonomous_loop is not None:
                history = self.engine.autonomous_loop.history
                if history:
                    last = history[-1]
                    action = last.get("action")
                    if action and action in MOTION_BY_ACTION:
                        return MOTION_BY_ACTION[action]
        except Exception:
            pass

        # 3. 외출 중인지 (env_adapter)
        try:
            if not self.engine.env_adapter.is_home():
                return "walking"   # 외출 = 활동 중
        except Exception:
            pass

        # 4. 기본 — idle
        return "idle"

    # ===== 입모양 =====

    def set_speaking(self, speaking: bool):
        self._is_speaking = speaking

    def is_speaking(self) -> bool:
        return self._is_speaking

    # ===== 종합 snapshot (visualizer가 사용) =====

    def snapshot(self) -> dict:
        return {
            "expression": self.expression(),
            "motion": self.motion(),
            "speaking": self._is_speaking,
        }
