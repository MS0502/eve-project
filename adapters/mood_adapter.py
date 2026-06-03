"""
MoodAdapter — 라운드 31 (B-3)

호르몬 + 시간대 + 대화 흐름 → 분위기 라벨 + 컨텍스트.
SituationResponder가 이걸 받아서 응답 톤 조절.

학계:
- Russell 1980 — Circumplex (valence × arousal)
- Mehrabian 1996 — PAD (pleasure × arousal × dominance)
- Bar 2007 — predictive brain, 분위기 = 주변 상태 통합 추정
- Ekman 1992 — basic emotions

원칙:
- 결정론
- 호르몬 mood는 이미 hormone_system이 계산 (재사용)
- 대화 흐름은 마지막 5턴
- 분위기 = 7가지 라벨 + 강도

분위기 9종:
  warm        — 따뜻함 (oxytocin 높음)
  excited     — 신남 (dopamine + arousal 높음)
  calm        — 평온 (serotonin, valence 중립)
  tense       — 긴장 (cortisol 높음)
  heavy       — 무거움 (cortisol 높고 dopamine 낮음)
  tired       — 피곤 (adenosine/melatonin 높음)
  curious     — 호기심 (norepinephrine + acetylcholine)
  playful     — 장난 (낮은 cort + 높은 dop, 깨어있는 시간)
  neutral     — 평소
"""

import time
from collections import deque
from typing import Optional


# 분위기 9종
MOODS = ["warm", "excited", "calm", "tense", "heavy", "tired",
         "curious", "playful", "neutral"]


def time_label(hour: float) -> str:
    """sim_hour → 시간대 라벨."""
    if 0 <= hour < 6:
        return "deep_night"
    if 6 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


class MoodAdapter:

    def __init__(self, engine, history_size: int = 5):
        self.engine = engine
        # 최근 대화 흐름 (최대 5턴)
        self.history: deque = deque(maxlen=history_size)
        # 마지막 분위기 (변화 추적용)
        self._last_mood: Optional[str] = None
        self._last_intensity: float = 0.0

    # ===== 분위기 추론 =====

    def infer_mood(self) -> dict:
        """
        현재 분위기 추론.
        Returns:
          {"mood": str, "intensity": float, "reasons": [str, ...],
           "valence": float, "arousal": float, "time_label": str}
        """
        result = {
            "mood": "neutral",
            "intensity": 0.0,
            "reasons": [],
            "valence": 0.0,
            "arousal": 0.5,
            "time_label": "midday",
        }

        # 호르몬 mood 가져오기
        try:
            hs = self.engine.hormone_adapter.hs
            mood_pad = hs.compute_mood()
            result["valence"] = mood_pad["valence"]
            result["arousal"] = mood_pad["arousal"]
            result["time_label"] = time_label(getattr(hs, "sim_hour", 12.0))

            h = hs.hormones

            # 분위기 결정 (우선순위 순)
            cort = h.get("cortisol")
            dop = h.get("dopamine")
            ot = h.get("oxytocin")
            ser = h.get("serotonin")
            mel = h.get("melatonin")
            ade = h.get("adenosine")
            ne = h.get("norepinephrine")
            ach = h.get("acetylcholine")

            cort_lvl = cort.level if cort else 0.5
            dop_lvl = dop.level if dop else 0.5
            ot_lvl = ot.level if ot else 0.5
            ser_lvl = ser.level if ser else 0.5
            mel_lvl = mel.level if mel else 0.0
            ade_lvl = ade.level if ade else 0.5
            ne_lvl = ne.level if ne else 0.5
            ach_lvl = ach.level if ach else 0.5

            # 1. 피곤 (mel/ade 높음 또는 늦은 시간)
            if mel_lvl > 0.6 or ade_lvl > 0.7 or result["time_label"] in ("deep_night", "night"):
                if mel_lvl > 0.6 or ade_lvl > 0.7:
                    result["mood"] = "tired"
                    result["intensity"] = max(mel_lvl, ade_lvl)
                    result["reasons"].append(f"melatonin={mel_lvl:.2f}, adenosine={ade_lvl:.2f}")
                    return result

            # 2. 긴장 (cort 매우 높음)
            if cort_lvl > 0.7:
                result["mood"] = "tense"
                result["intensity"] = cort_lvl
                result["reasons"].append(f"cortisol high={cort_lvl:.2f}")
                return result

            # 3. 무거움 (cort 높고 dop 낮음)
            if cort_lvl > 0.55 and dop_lvl < 0.35:
                result["mood"] = "heavy"
                result["intensity"] = cort_lvl - dop_lvl
                result["reasons"].append(f"cort {cort_lvl:.2f} > dop {dop_lvl:.2f}")
                return result

            # 4. 따뜻함 (ot 높음 + cort 낮음)
            if ot_lvl > 0.6 and cort_lvl < 0.5:
                result["mood"] = "warm"
                result["intensity"] = ot_lvl
                result["reasons"].append(f"oxytocin={ot_lvl:.2f}")
                return result

            # 5. 신남 (dop + arousal 높음)
            if dop_lvl > 0.6 and result["arousal"] > 0.6:
                result["mood"] = "excited"
                result["intensity"] = dop_lvl
                result["reasons"].append(f"dopamine={dop_lvl:.2f}, arousal={result['arousal']:.2f}")
                return result

            # 6. 호기심 (ne + ach 높음)
            if ne_lvl > 0.55 and ach_lvl > 0.55:
                result["mood"] = "curious"
                result["intensity"] = (ne_lvl + ach_lvl) / 2
                result["reasons"].append(f"ne+ach 높음")
                return result

            # 7. 장난기 (낮 + 낮은 cort + 적당한 dop)
            if cort_lvl < 0.35 and dop_lvl > 0.4 and result["time_label"] in ("midday", "afternoon"):
                result["mood"] = "playful"
                result["intensity"] = dop_lvl - cort_lvl
                result["reasons"].append("낮 시간 + 낮은 스트레스")
                return result

            # 8. 평온 (ser 높음 + valence 중립~양)
            if ser_lvl > 0.6 and result["valence"] >= 0:
                result["mood"] = "calm"
                result["intensity"] = ser_lvl
                result["reasons"].append(f"serotonin={ser_lvl:.2f}")
                return result

            # 기본
            result["mood"] = "neutral"
            result["intensity"] = 0.5

        except Exception:
            pass

        return result

    # ===== 대화 흐름 추적 =====

    def record_turn(self, user_text: str, eve_response: str, situation: Optional[str] = None):
        """매 turn 기록."""
        self.history.append({
            "time": time.time(),
            "user": user_text,
            "eve": eve_response,
            "situation": situation,
        })

    def conversation_flow(self) -> dict:
        """최근 대화 흐름 분석."""
        if not self.history:
            return {"length": 0, "dominant_situations": [], "is_continuation": False}

        situations = [h.get("situation") for h in self.history if h.get("situation")]
        # 같은 상황이 연속되면 = 깊이 들어가는 대화
        is_continuation = False
        if len(situations) >= 2 and situations[-1] == situations[-2]:
            is_continuation = True

        # 우세한 상황 (최근 5턴 중 가장 많은 거)
        from collections import Counter
        cnt = Counter(situations)
        dominant = [s for s, c in cnt.most_common(3) if c >= 2]

        return {
            "length": len(self.history),
            "dominant_situations": dominant,
            "is_continuation": is_continuation,
            "last_situation": situations[-1] if situations else None,
        }

    # ===== 모호 입력 처리 =====

    AMBIGUOUS_PATTERNS = [
        "그러게", "그래", "그냥", "음", "그치", "맞아",
        "좀 그래", "그저 그래", "뭐랄까", "어쩌지", "글쎄",
        "별거 아냐", "별로", "흠", "허",
    ]

    def is_ambiguous_input(self, text: str) -> bool:
        """모호 입력?"""
        # 구두점/공백 제거 후 비교
        cleaned = text.strip()
        for ch in [".", "?", "!", "…", " "]:
            cleaned = cleaned.replace(ch, "")
        if not cleaned:
            return False
        # 정확 매칭
        if cleaned in self.AMBIGUOUS_PATTERNS:
            return True
        # 너무 짧고 한글
        if len(cleaned) <= 3 and all('\uac00' <= c <= '\ud7af' for c in cleaned):
            # 알려진 모호 키워드 시작
            for kw in self.AMBIGUOUS_PATTERNS:
                if cleaned == kw or cleaned.startswith(kw):
                    return True
        return False

    def interpret_ambiguous(self, text: str, mood: dict) -> str:
        """모호 입력을 분위기로 해석 → EVE가 어떻게 받을지."""
        m = mood.get("mood", "neutral")
        flow = self.conversation_flow()
        last_sit = flow.get("last_situation")

        # 직전이 감정 공유면 → 공감 깊게
        if last_sit == "emotional_share":
            if m in ("heavy", "tense", "tired"):
                return "wait_silence"           # 옆에 있어줌
            return "follow_up_emotion"          # "뭐 일 있었어?"

        # 직전이 인사 또는 small_talk면 → 안부
        if m == "warm":
            return "warm_continue"              # 따뜻하게 이어가기
        if m in ("heavy", "tense"):
            return "concerned_check"            # "괜찮아?"

        return "open_question"                  # "어떻게 지내?"

    # ===== 응답 톤 힌트 (SituationResponder가 사용) =====

    def get_response_hint(self) -> dict:
        """
        SituationResponder에 줄 힌트.
        - tone: "soft" | "warm" | "calm" | "playful" | "concerned" | "neutral"
        - quietness: 0~1 (높을수록 짧고 조용한 응답)
        - empathy_depth: 0~1
        """
        mood = self.infer_mood()
        m = mood["mood"]

        if m == "tired":
            return {"tone": "soft", "quietness": 0.8, "empathy_depth": 0.3, "mood": m}
        if m == "tense":
            return {"tone": "calm", "quietness": 0.6, "empathy_depth": 0.7, "mood": m}
        if m == "heavy":
            return {"tone": "concerned", "quietness": 0.5, "empathy_depth": 0.9, "mood": m}
        if m == "warm":
            return {"tone": "warm", "quietness": 0.2, "empathy_depth": 0.6, "mood": m}
        if m == "excited":
            return {"tone": "playful", "quietness": 0.1, "empathy_depth": 0.4, "mood": m}
        if m == "curious":
            return {"tone": "warm", "quietness": 0.3, "empathy_depth": 0.5, "mood": m}
        if m == "playful":
            return {"tone": "playful", "quietness": 0.2, "empathy_depth": 0.3, "mood": m}
        if m == "calm":
            return {"tone": "calm", "quietness": 0.4, "empathy_depth": 0.5, "mood": m}

        return {"tone": "neutral", "quietness": 0.4, "empathy_depth": 0.5, "mood": m}

    # ===== 영속화 =====

    def to_dict(self) -> dict:
        return {
            "history": list(self.history),
            "_last_mood": self._last_mood,
            "_last_intensity": self._last_intensity,
        }

    def from_dict(self, data: dict):
        from collections import deque
        self.history = deque(data.get("history", []), maxlen=5)
        self._last_mood = data.get("_last_mood")
        self._last_intensity = data.get("_last_intensity", 0.0)
