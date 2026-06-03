"""
EmotionCauseTracker — 라운드 39a

GPT 분석: 지금 EVE는 "감정 표현"은 되는데 "감정 이유"가 없다.
인간은 "외로워" 하면서 "왜 외로운가?" 도 안다.

학계:
- Weiner 1985, 1986 — 귀인 이론 (감정은 사건+해석에서 나옴)
- Heider 1958 — naive scientist (사람은 항상 원인 찾는다)
- Neumann 2000 — 귀인이 감정에 영향을 준다 (causal influence)
- Gross 2002 — emotion regulation (이유를 알면 조절 가능)

원칙:
- 호르몬이 *변할 때*마다 그 시점의 *문맥/사건* 기록
- 큰 변화 (spike) 일수록 더 강한 cause로 기록
- 발화 시 cause를 자연스럽게 짜넣기 ("X 때문에 그런가...")

EVE의 cause 자료:
- 마지막 사용자 입력 시점/내용
- 마지막 자율 발화 시점
- 마지막 학습/가르침 사건
- 시간 경과 (idle gap)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CauseEvent:
    """원인 사건 한 개. intensity는 시간 따라 자연 감소."""
    timestamp: float
    cause_type: str        # "user_input" / "long_silence" / "teaching" / "learn_fail" / "interaction_gap"
    description: str       # 자연어 설명 ("아까 너랑 얘기 끊겼을 때")
    hormone_change: dict   # {"oxytocin": -0.1, ...} 순간 변화
    intensity: float = 1.0  # 0~1 (얼마나 강한 사건)
    
    def current_intensity(self, now: Optional[float] = None,
                          decay_per_sec: float = 0.001) -> float:
        """
        시간 따라 자연 감소한 현재 intensity.
        decay_per_sec=0.001 → 1000초 (16분) 후 약 37%로.
        오래된 사건은 자연스럽게 fade.
        """
        import math
        if now is None:
            now = time.time()
        elapsed = max(0, now - self.timestamp)
        return self.intensity * math.exp(-decay_per_sec * elapsed)


# 호르몬 변화 → 감정 매핑
HORMONE_TO_EMOTION = {
    "cortisol_up":   "stress",      # 코르티솔 증가 = 스트레스
    "cortisol_down": "relief",
    "oxytocin_up":   "bonding",     # 옥시토신 증가 = 유대
    "oxytocin_down": "lonely",      # 감소 = 외로움
    "dopamine_up":   "excited",
    "dopamine_down": "flat",
    "norepi_up":     "alert",
    "norepi_down":   "tired",
    "melatonin_up":  "sleepy",
}


class EmotionCauseTracker:
    """
    감정 변화의 원인 추적 + 발화 짜넣기.
    """

    def __init__(self, engine=None, max_history: int = 50):
        self.engine = engine
        self.history: deque = deque(maxlen=max_history)
        # 마지막 호르몬 스냅샷 (변화 감지용)
        self._last_hormone_snapshot: dict = {}
        self._last_user_input_ts: float = 0.0
        self._last_user_input_text: str = ""
        # 통계
        self.causes_recorded = 0
        self.causes_used_in_speech = 0

    # ===== 사건 기록 =====

    def record_user_input(self, text: str, timestamp: Optional[float] = None):
        """사용자 입력 받았을 때."""
        ts = timestamp or time.time()
        self._last_user_input_ts = ts
        self._last_user_input_text = text
        # 호르몬 스냅샷 갱신
        self._refresh_hormone_snapshot()

    def record_event(self, cause_type: str, description: str,
                     intensity: float = 1.0):
        """일반 사건."""
        ts = time.time()
        # 현재 호르몬 변화량 측정
        change = self._compute_hormone_change()
        evt = CauseEvent(
            timestamp=ts,
            cause_type=cause_type,
            description=description,
            hormone_change=change,
            intensity=intensity,
        )
        self.history.append(evt)
        self.causes_recorded += 1
        self._refresh_hormone_snapshot()

    def detect_passive_cause(self) -> Optional[CauseEvent]:
        """
        외부 사건 X일 때, 시간 경과 자체를 cause로 인식.
        예: 사용자 입력 1분 X → "interaction_gap" 사건.
        """
        now = time.time()
        if self._last_user_input_ts > 0:
            gap = now - self._last_user_input_ts
            if gap > 60.0:  # 1분 이상 침묵
                desc = self._gap_description(gap)
                # 같은 종류 사건 최근에 있으면 갱신만
                if self.history and self.history[-1].cause_type == "interaction_gap":
                    self.history[-1].description = desc
                    self.history[-1].timestamp = now
                else:
                    self.record_event("interaction_gap", desc, intensity=min(1.0, gap / 600.0))
                return self.history[-1] if self.history else None
        return None

    # ===== 호르몬 변화 추적 =====

    def _refresh_hormone_snapshot(self):
        if self.engine is None or self.engine.hormone_adapter is None:
            return
        h = self.engine.hormone_adapter.hs.hormones
        self._last_hormone_snapshot = {
            name: hormone.level for name, hormone in h.items()
        }

    def _compute_hormone_change(self) -> dict:
        """현재 호르몬 vs 마지막 스냅샷 차이."""
        if self.engine is None or self.engine.hormone_adapter is None:
            return {}
        h = self.engine.hormone_adapter.hs.hormones
        change = {}
        for name, hormone in h.items():
            current = hormone.level
            prev = self._last_hormone_snapshot.get(name, current)
            diff = current - prev
            if abs(diff) > 0.05:    # 의미있는 변화만
                change[name] = round(diff, 3)
        return change

    # ===== cause 표현 (발화에 짜넣기) =====

    def get_relevant_cause(self, current_emotion: str,
                            min_intensity: float = 0.15) -> Optional[CauseEvent]:
        """
        현재 감정과 관련 있는 가장 강한 cause 찾기.
        시간 따라 fade — 너무 약해진 cause는 무시 (min_intensity).
        """
        if not self.history:
            return self.detect_passive_cause()
        
        now = time.time()
        # 최근 사건부터 역순 — 매칭하는 거 중 *현재 강도* 충분한 것
        for evt in reversed(self.history):
            current = evt.current_intensity(now)
            if current < min_intensity:
                continue
            
            # 외로움이면 oxytocin 감소 사건
            if current_emotion == "lonely":
                ot_change = evt.hormone_change.get("oxytocin", 0)
                if ot_change < -0.05:
                    return evt
                if evt.cause_type == "interaction_gap":
                    return evt
            elif current_emotion == "tired":
                if evt.hormone_change.get("norepinephrine", 0) < -0.05:
                    return evt
                if evt.hormone_change.get("cortisol", 0) > 0.05:
                    return evt
            elif current_emotion == "energetic":
                if evt.hormone_change.get("dopamine", 0) > 0.05:
                    return evt
            elif current_emotion == "curious":
                if evt.cause_type in ("user_input", "teaching"):
                    return evt
            elif current_emotion == "anxious":
                if evt.hormone_change.get("cortisol", 0) > 0.05:
                    return evt
        
        # 매칭 없으면 — passive cause 시도 (interaction_gap)
        passive = self.detect_passive_cause()
        if passive is not None:
            return passive
        # 그래도 없으면 가장 최근 사건 (강도 무관)
        return self.history[-1] if self.history else None

    def _gap_description(self, seconds: float) -> str:
        """침묵 길이 → 자연어 설명."""
        if seconds < 120:
            return "잠깐 조용했던 사이"
        elif seconds < 600:
            return "한참 말 안 한 사이"
        elif seconds < 3600:
            return "오래 조용했던 사이"
        else:
            return "긴 침묵"

    def speech_clause_for_cause(self, evt: CauseEvent, emotion: str) -> str:
        """
        cause를 발화에 짜넣을 절 만들기.
        예: "아까 대화 끊겨서 그런가..."
            "아까 너 PT 얘기 들어서 그런가..."
        """
        if evt is None:
            return ""
        
        # 사건 유형별 자연어
        if evt.cause_type == "user_input":
            text_short = (evt.description or "")[:20]
            if emotion == "curious":
                return f"아까 {text_short} 얘기 떠올라서 그런가..."
            elif emotion == "lonely":
                return "아까 얘기하다 끊겨서 그런가..."
            else:
                return f"아까 너 한 말 때문에 그런가..."
        
        elif evt.cause_type == "interaction_gap":
            if emotion == "lonely":
                return f"{evt.description} 동안 좀 그래서..."
            elif emotion == "tired":
                return "오래 조용해서 좀 늘어졌나..."
            else:
                return f"{evt.description} 때문인지..."
        
        elif evt.cause_type == "teaching":
            return "아까 너가 가르쳐준 거 곱씹다 보니..."
        
        elif evt.cause_type == "learn_fail":
            return "뭔가 제대로 못 알아들은 게 신경 쓰여서..."
        
        # 일반
        return f"{evt.description} 때문인지..."

    # ===== 통계 =====

    def stats(self) -> dict:
        recent = list(self.history)[-5:] if self.history else []
        return {
            "causes_recorded": self.causes_recorded,
            "causes_used": self.causes_used_in_speech,
            "history_size": len(self.history),
            "last_user_input_ts": self._last_user_input_ts,
            "recent_5": [(e.cause_type, e.description) for e in recent],
        }
