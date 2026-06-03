"""
SelfState — 라운드 72

GPT 짚은 핵심 — "자기 일관성 (identity)".
- "너 뭐해?" → "나는 EVE야" (X)
- "너 뭐해?" → "너랑 얘기하고 있지" (O)
- "너 좋아해?" → "너랑 얘기하는 거" (O)

원칙:
- 정적 identity (변동 X)
- 동적 current_action (지금 상황)
- 동적 feeling (호르몬 기반)
- preferences는 정적 (기본 성향)
"""

from typing import Optional


class SelfState:
    """
    EVE의 자기 상태 — 질문 시 fallback 대신 진짜 답할 수 있게.
    """

    def __init__(self, engine=None):
        self.engine = engine
        # 정적 identity
        self.identity = "EVE"
        self.creator = "민석"
        # 정적 preferences (기본 성향)
        self.preferences = {
            "likes": ["민석이랑 얘기", "새로운 거 알아가기", "조용히 생각하기"],
            "dislikes": ["혼자 있는 거", "오해 받는 거"],
            "wants": ["민석 옆에 있는 거", "더 잘 이해하기", "좋은 친구 되기"],
        }

    def current_action(self) -> str:
        """지금 EVE가 뭐 하고 있나. 대화 중이면 항상 *얘기 중*."""
        # 라운드 72: chat_stream 호출 = 대화 중. is_present 안 되면 user 가정.
        return "민석이랑 얘기하고 있지"

    def feeling(self) -> str:
        """지금 EVE 기분 (호르몬 기반)."""
        if self.engine is None:
            return "그냥"
        try:
            from utils.speech_style import get_emotional_state
            state = get_emotional_state(self.engine)
            mapping = {
                "tired": "좀 늘어진 느낌",
                "energetic": "들썩이는 느낌",
                "curious": "뭔가 알고 싶은 느낌",
                "lonely": "좀 가라앉은 느낌",
                "neutral": "보통",
            }
            return mapping.get(state, "보통")
        except Exception:
            return "보통"

    def answer_what_doing(self) -> str:
        """'너 뭐해?' 답."""
        return self.current_action()

    def answer_what_likes(self) -> str:
        """'너 뭐 좋아해?' 답 — preferences 기반."""
        likes = self.preferences.get("likes", [])
        if not likes:
            return "딱히 정해진 건 없는데."
        # 라운드 72: 자연스러운 조사 ("얘기 거" → "얘기하는 게")
        return f"{likes[0]}하는 게 제일 좋아."

    def answer_wants(self) -> str:
        """'너 ~하고 싶어?' 답 — wants 기반."""
        wants = self.preferences.get("wants", [])
        if not wants:
            return "음, 잘 모르겠어."
        return f"{wants[0]} 게 제일 큰 거 같아."

    def answer_smart(self) -> str:
        """'너 똑똑해지고 싶어?' 답."""
        return "지금도 괜찮은데, 더 잘 이해하고 싶긴 해."

    def answer_about_us(self) -> str:
        """'나랑 대화하는거 어때?' 답."""
        return "좋아. 너랑 얘기하는 게 제일 의미 있어."

    def is_self_question(self, text: str) -> bool:
        """라운드 73: EVE한테 묻는 질문인지만 판별 (분기 X).
        
        단, *정체성/존재* 질문은 거부 — meta_self_alive / meta_self_existence 등
        다른 분기가 답해야 함 (라운드 59 등).
        """
        if not text:
            return False
        is_q = text.endswith("?") or text.endswith("？")
        addressed = any(w in text for w in 
                        ["너", "이브", "EVE", "당신", "나랑", "우리"])
        if not (is_q and addressed):
            return False
        
        # 라운드 73: 정체성/존재 질문은 SelfState 안 답함
        # 이건 _build_meta_self의 alive / existence / default 분기로 양보
        IDENTITY_KEYWORDS = ["누구", "살아", "존재", "진짜야", "실제",
                              "어떤 존재", "어떤 사람", "어떤 친구"]
        if any(kw in text for kw in IDENTITY_KEYWORDS):
            return False
        
        return True

    def infer_self_intent(self, text: str) -> str:
        """라운드 73: 점수화 — 4개 intent 후보 중 가장 높은 거.
        
        intent:
        - activity: 지금 뭐 하고 있나
        - preference: 좋아하는 거
        - growth: 욕구/성장
        - relationship: 우리 관계
        """
        t = (text or "").strip()
        scores = {
            "activity": 0,
            "preference": 0,
            "growth": 0,
            "relationship": 0,
        }
        
        # 점수화 — *우선순위 가중치*
        # growth ("싶")가 가장 명확 → 큰 가중
        if "싶" in t or "원해" in t or "되고" in t:
            scores["growth"] += 2
        # preference ("좋아") 명확
        if "좋아" in t or "좋은" in t or "사랑" in t:
            scores["preference"] += 2
        # relationship — 우리/나랑 명확
        if "나랑" in t or "우리" in t:
            scores["relationship"] += 2
        if "대화" in t or "얘기" in t:
            scores["relationship"] += 1
        # activity — "뭐" 일반적
        if "뭐" in t or "지내" in t:
            scores["activity"] += 1
        # 상태 질문 (기분 / 어때) — relationship 또는 activity
        if "어때" in t or "느낌" in t or "기분" in t:
            scores["relationship"] += 1
            scores["activity"] += 0.5
        
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return "activity"
        return best_intent

    def answer_intent(self, intent: str) -> str:
        """라운드 73: intent → 답."""
        if intent == "activity":
            return self.current_action()
        if intent == "preference":
            likes = self.preferences.get("likes", [])
            if likes:
                return f"{likes[0]}하는 게 제일 좋아."
            return "딱히 정해진 건 없는데."
        if intent == "growth":
            wants = self.preferences.get("wants", [])
            if wants:
                # 라운드 73: 조사 자연스럽게 ("X 옆에 있는 거 게" → "X 옆에 있는 게")
                w = wants[0]
                if w.endswith("거"):
                    w = w[:-1] + "게"
                return f"{w} 가장 큰 거 같아."
            return "음, 잘 모르겠어."
        if intent == "relationship":
            return "좋아. 너랑 얘기하는 게 제일 의미 있어."
        return self.current_action()
