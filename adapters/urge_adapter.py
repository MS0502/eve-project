"""
UrgeAdapter — 라운드 36

발화 충동 (urge) 계산.

원칙: 인간은 *말하고 싶을 때* 말한다.
- 호기심 (도파민) 높음 → 표현 욕구
- 외로움 (낮은 oxytocin + 시간) → 말 걸고 싶음
- 흥분 (norepinephrine) → 입이 들썩
- 평온 (낮은 cortisol) → 자연스러운 흐름
- 피곤 (낮은 energy) → 침묵

학계:
- Damasio Somatic Marker — 신체상태가 결정 유도
- Vygotsky inner→outer speech — 사고가 임계 넘으면 외부화
- Harlow & Maslow — 사회적 욕구 (loneliness drive)
"""


# 발화 임계 — 이 값 넘어야 발화
# 0.45 = 호르몬 중립 + 시간 압박 30초 정도면 통과 (자연스러운 빈도)
URGE_THRESHOLD = 0.45

# 가중치 (학계: 사회적 + 호기심 동기 우세)
W_DOPAMINE = 0.30      # 호기심 → 표현
W_NOREPI = 0.20        # 흥분 → 말 들썩
W_LONELINESS = 0.30    # 외로움 → 말 걸고 싶음
W_TIME_SINCE = 0.20    # 시간 누적 (안 말한 지 오래)


class UrgeAdapter:
    """발화 충동 계산기."""

    def __init__(self, engine=None):
        self.engine = engine
        self.last_emit_ts = 0.0
        # 통계
        self.urge_checks = 0
        self.urge_above_threshold = 0
        self.urge_below_threshold = 0
        # 최근 urge 추적 (디버깅)
        self._last_urge = 0.0
        self._last_components = {}

    def compute_urge(self, current_time: float = 0.0) -> float:
        """
        발화 충동 계산. 0~1+ 범위.
        ≥ URGE_THRESHOLD 면 발화.
        """
        self.urge_checks += 1
        
        if self.engine is None or self.engine.hormone_adapter is None:
            return 0.5  # 중립 (호르몬 없으면 50%)
        
        h = self.engine.hormone_adapter
        
        # 1. 도파민 (호기심)
        dopamine = h.hs.hormones["dopamine"].level if "dopamine" in h.hs.hormones else 0.5
        
        # 2. 노르에피네프린 (흥분)
        norepi = h.hs.hormones["norepinephrine"].level if "norepinephrine" in h.hs.hormones else 0.5
        
        # 3. 외로움 = (1 - oxytocin) — 옥시토신 낮을수록 외로움
        oxy = h.hs.hormones["oxytocin"].level if "oxytocin" in h.hs.hormones else 0.5
        loneliness = 1.0 - oxy
        
        # 4. 시간 누적 — 마지막 발화 후 오래
        # last_emit_ts=0 (시작 직후) 이면 current_time을 시간으로
        time_since = current_time - self.last_emit_ts if self.last_emit_ts > 0 else current_time
        time_pressure = min(1.0, time_since / 60.0)  # 60초 누적되면 max
        
        # 가중 합
        urge = (
            W_DOPAMINE * dopamine +
            W_NOREPI * norepi +
            W_LONELINESS * loneliness +
            W_TIME_SINCE * time_pressure
        )
        
        self._last_urge = urge
        self._last_components = {
            "dopamine": dopamine,
            "norepi": norepi,
            "loneliness": loneliness,
            "time_pressure": time_pressure,
        }
        
        if urge >= URGE_THRESHOLD:
            self.urge_above_threshold += 1
        else:
            self.urge_below_threshold += 1
        
        return urge

    def should_emit(self, current_time: float = 0.0) -> bool:
        """이 순간 발화해야 하는가."""
        return self.compute_urge(current_time) >= URGE_THRESHOLD

    def mark_emitted(self, current_time: float = 0.0):
        """발화했음 — 시간 기록 (다음 urge 계산용)."""
        self.last_emit_ts = current_time
    
    def stats(self) -> dict:
        return {
            "urge_checks": self.urge_checks,
            "above_threshold": self.urge_above_threshold,
            "below_threshold": self.urge_below_threshold,
            "last_urge": self._last_urge,
            "last_components": self._last_components,
            "last_emit_ts": self.last_emit_ts,
        }
