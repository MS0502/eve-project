"""
UserPresenceAdapter

v40 UserPresence ↔ v41 사용자 인식.

핵심: 김민석 = 진짜 친구 (시뮬 NPC와 명확히 구분).

흐름:
1. 첫 chat — first_visit_setup (강한 카테고리 연결: 민석 ↔ 진짜)
2. 매 chat — visit_arrive (호르몬 약자극: ot+, cort-)
3. 매 chat 끝 — visit_leave (방문 카운트)
4. is_user_real("민석") → True

학계:
- Damon & Hart 1988 — 자아 정체성에서 의미 있는 타자
- Bowlby 1969 — 애착 (진짜 관계 vs 일시적)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

import time

from user_presence import UserPresence

from utils.mock_eve import MockEve


class UserPresenceAdapter:

    def __init__(self,
                 engine,
                 user_name: str = "민석"):
        self.engine = engine
        self.user_name = user_name
        self._mock = MockEve(engine)
        self.up = UserPresence(self._mock, user_name=user_name)
        self._first_done = False
        # 시간 추적
        self.last_seen_time: float = 0.0
        self.first_seen_time: float = 0.0
        self._cached_gap: float = 0.0      # 마지막 visit 직전의 gap

    def ensure_first_visit(self):
        """최초 chat 전에 한 번 호출."""
        if self._first_done:
            return
        try:
            self.up.first_visit_setup()
            self._first_done = True
            self.first_seen_time = time.time()
        except Exception:
            pass

    def on_user_arrive(self):
        """사용자 메시지 받았을 때."""
        self.ensure_first_visit()
        # 시간 기록 — gap 계산 → cache → 갱신
        now = time.time()
        if self.last_seen_time > 0:
            self._cached_gap = now - self.last_seen_time
        else:
            self._cached_gap = 0.0
        self.last_seen_time = now
        try:
            self.up.visit_arrive()
        except Exception:
            pass

    def gap_since_last_seen(self) -> float:
        """마지막 visit 직전의 gap (캐시됨, 인사 멘트용)."""
        return self._cached_gap

    def gap_phrase(self) -> str:
        """시간 간격 → 한국어 문구. EVE가 자연스럽게 사용."""
        gap = self._cached_gap
        if gap < 60:
            return ""
        if gap < 600:
            return "방금 전"
        if gap < 3600:
            return "조금 전"
        if gap < 7200:
            return "한 시간 전쯤"
        if gap < 21600:
            return "몇 시간 전"
        if gap < 86400:
            return "오늘 아침"
        if gap < 172800:
            return "어제"
        if gap < 604800:
            return "며칠 전"
        return "오랜만"

    def on_user_leave(self, message: str = "", response: str = ""):
        try:
            self.up.visit_leave(message=message, response=response)
        except Exception:
            pass

    def get_intimacy(self) -> float:
        try:
            return float(self.up.get_intimacy())
        except Exception:
            return 0.0

    def is_real(self, name: str) -> bool:
        try:
            return bool(self.up.is_user_real(name))
        except Exception:
            return name == self.user_name

    def stats(self) -> dict:
        try:
            return self.up.stats() or {}
        except Exception:
            return {}
