"""
ContextSeed — 라운드 68

GPT 짚은 "결정론 유지하면서 사람처럼 변동":
- random.random() X
- hash(text + hormone + turn + history) → 결정론 seed
- 같은 입력이라도 시간/호르몬/대화 흐름 따라 *다른 응답*

원칙:
- 완전 결정론 (재현 가능)
- 외부에서 보면 *자연 변동*
- 추적 가능 (어떤 요소가 변동에 영향했는지)
"""

from typing import Optional


class ContextSeed:
    """
    결정론적 다양성 통합 seed.
    
    같은 입력 + 같은 호르몬 + 같은 turn = 같은 결과.
    Turn이 진행되면서 자연 변동.
    """

    BIG_PRIME = 2_147_483_647   # Mersenne prime

    def __init__(self, engine=None):
        self.engine = engine
        self.turn_count = 0   # 사용자 입력 받을 때마다 +1
        # 최근 응답 keyword (간단 history)
        self.recent_keywords: list = []
        self.max_keywords = 5

    def increment_turn(self):
        """사용자 입력 받을 때 호출."""
        self.turn_count += 1

    def remember_keyword(self, kw: str):
        """직전 응답의 핵심 키워드 기록 (다음 turn에 영향)."""
        if not kw:
            return
        self.recent_keywords.append(kw)
        if len(self.recent_keywords) > self.max_keywords:
            self.recent_keywords.pop(0)

    def compute(self, text: str, ctx: Optional[dict] = None) -> int:
        """
        text + 호르몬 + turn + history → 결정론 int seed.
        
        외부에서 보면 *변동*, 내부는 *완전 결정론*.
        """
        ctx = ctx or {}
        
        # 1. 입력 텍스트 — 가장 큰 영향
        text_hash = sum(ord(c) for c in (text or ""))
        
        # 2. 호르몬 상태 — 시간에 따라 변하는 *내부 상태*
        hormone_hash = 0
        if self.engine is not None:
            try:
                h = self.engine.hormone_adapter.hs.hormones
                # 주요 호르몬 level을 정수로 (소수 둘째자리)
                for name in ("dopamine", "oxytocin", "cortisol", "melatonin"):
                    horm = h.get(name)
                    if horm:
                        hormone_hash += int(horm.level * 100)
            except Exception:
                pass
        
        # 3. Turn count — 같은 입력이라도 turn 진행하면 다름
        turn_hash = self.turn_count * 13
        
        # 4. 최근 keyword — 이전 응답이 다음 응답에 영향
        history_hash = sum(sum(ord(c) for c in kw) for kw in self.recent_keywords)
        
        # 통합 (큰 prime 곱해서 충돌 줄임)
        combined = (
            text_hash * 31 +
            hormone_hash * 17 +
            turn_hash * 7 +
            history_hash * 3
        ) % self.BIG_PRIME
        
        return abs(combined)

    def stats(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "recent_keywords": list(self.recent_keywords),
        }
