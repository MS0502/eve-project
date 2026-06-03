"""
CuriosityAdapter — 라운드 23

EVE가 모르는 카테고리 만나면 자율로 인터넷 검색해서 학습.

원칙:
- 사용자 trigger 또는 autonomous_loop tick에서 발동
- 하루 검색 한도 (rate limit, 의도치 않은 폭주 방지)
- 화이트리스트 도메인만 (web_learning_adapter 활용)
- 결과 corpus_adapter로 흘러서 학습
- "모름" 판단 = SA 활성 0 + CG 인과 0 + HG 관계 0 + EM 일화 0

EVE 결정론 안 깸 (검색 자체는 외부, 사고는 EVE가).

학계 근거:
- Loewenstein 1994 — Information Gap Theory of Curiosity
- Friston 2010 — Active Inference (epistemic value)
- Kidd & Hayden 2015 — Psychology and neuroscience of curiosity
"""

import time
from typing import Optional


class CuriosityAdapter:

    def __init__(self, engine,
                 daily_limit: int = 20,
                 cooldown_seconds: float = 300.0):
        self.engine = engine
        self.daily_limit = daily_limit
        self.cooldown_seconds = cooldown_seconds
        # 상태
        self.searches_today: list[dict] = []
        self.day_start = self._today_start()
        self.last_search_time = 0.0
        # 사용자 입력에서 검색했는지 / 자율 검색했는지 분리 추적
        self._known_unknowns: set[str] = set()    # 이미 시도한 카테고리

    def _today_start(self) -> float:
        t = time.time()
        return t - (t % 86400)

    def _reset_if_new_day(self):
        today = self._today_start()
        if today > self.day_start:
            self.searches_today = []
            self.day_start = today
            self._known_unknowns = set()

    # ===== 모름 판정 =====

    def is_unknown_category(self, category: str) -> bool:
        """카테고리에 대해 EVE가 충분히 알고 있는지."""
        if not category or len(category) < 2:
            return False

        # SA 활성 있나?
        try:
            sa = self.engine.activation_adapter.sa
            if sa.activations.get(category, 0.0) > 0.0:
                return False     # 알고 있음
            # 페어 관계 있나?
            for other in sa.activations:
                if other == category:
                    continue
                w = sa.get_weight(category, other)
                if w > 0.0:
                    return False
        except Exception:
            pass

        # CG 인과 있나?
        try:
            if self.engine.simulation_engine.cg is not None:
                cg = self.engine.simulation_engine.cg
                children = cg.children(category)
                if children:
                    return False
        except Exception:
            pass

        # HG 관계 있나?
        try:
            if self.engine.hypergraph_adapter is not None:
                neighbors = self.engine.hypergraph_adapter.neighbors(category)
                if neighbors:
                    return False
        except Exception:
            pass

        # EM 일화 있나?
        try:
            if self.engine.memory_adapter is not None:
                results = self.engine.memory_adapter.em.recall({category}, top_n=1, min_overlap=1)
                if results:
                    return False
        except Exception:
            pass

        return True

    def find_unknowns_in_meaning(self, meaning) -> list[str]:
        """현재 입력의 카테고리 중 EVE가 모르는 것 찾기."""
        unknowns = []
        for cat in (meaning.entities or []):
            if cat in self._known_unknowns:
                continue       # 이미 시도함, skip
            if self.is_unknown_category(cat):
                unknowns.append(cat)
        return unknowns

    # ===== 한도 체크 =====

    def can_search(self) -> tuple[bool, str]:
        """검색 가능한 상태인지."""
        self._reset_if_new_day()
        # 일 한도
        if len(self.searches_today) >= self.daily_limit:
            return (False, "daily_limit")
        # 쿨다운
        elapsed = time.time() - self.last_search_time
        if elapsed < self.cooldown_seconds:
            return (False, "cooldown")
        # web 어댑터 사용 가능?
        if self.engine.web_learning_adapter is None:
            return (False, "no_web")
        return (True, "ok")

    # ===== 자율 검색 실행 =====

    def search_unknown(self, category: str) -> dict:
        """모르는 카테고리 1개 검색 + 학습."""
        ok, reason = self.can_search()
        if not ok:
            return {"category": category, "searched": False, "reason": reason}

        # wiki 우선
        web = self.engine.web_learning_adapter
        result = web.wiki_and_learn(category)
        # 시도 기록
        self._known_unknowns.add(category)
        self.last_search_time = time.time()
        self.searches_today.append({
            "category": category,
            "time": self.last_search_time,
            "learned": result.get("learned", False),
            "text_length": result.get("text_length", 0),
        })
        return {
            "category": category,
            "searched": True,
            "learned": result.get("learned", False),
            "text_length": result.get("text_length", 0),
        }

    def explore_meaning(self, meaning) -> list[dict]:
        """meaning에서 모름 찾고 1~2개 자율 검색."""
        unknowns = self.find_unknowns_in_meaning(meaning)
        if not unknowns:
            return []
        # 한 turn에 최대 1개 (과부하 방지)
        results = []
        for cat in unknowns[:1]:
            r = self.search_unknown(cat)
            results.append(r)
            if not r["searched"]:
                break       # 한도 초과면 중단
        return results

    # ===== 자율 사이클 통합 (autonomous_loop에서 호출) =====

    def explore_in_autonomous_step(self) -> Optional[dict]:
        """
        autonomous_loop step에서 호출.
        호르몬 curiosity 높을 때 + 활성 카테고리 중 모름 있으면 검색.
        """
        # curiosity 호르몬 체크 (있으면)
        try:
            hs = self.engine.hormone_adapter.hs
            curiosity = hs.hormones.get("curiosity")
            if curiosity is None or curiosity.level < 0.5:
                return None
        except Exception:
            return None

        # 활성 카테고리 중 모름 찾기
        try:
            sa = self.engine.activation_adapter.sa
            active = [(cat, lvl) for cat, lvl in sa.activations.items() if lvl > 0.1]
            active.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            return None

        for cat, _ in active[:5]:
            if cat in self._known_unknowns:
                continue
            if self.is_unknown_category(cat):
                return self.search_unknown(cat)

        return None

    # ===== 상태 =====

    def status(self) -> dict:
        self._reset_if_new_day()
        return {
            "searches_today": len(self.searches_today),
            "daily_limit": self.daily_limit,
            "remaining": self.daily_limit - len(self.searches_today),
            "cooldown_seconds": self.cooldown_seconds,
            "since_last": time.time() - self.last_search_time if self.last_search_time > 0 else None,
            "tried_categories": len(self._known_unknowns),
        }

    def history(self, n: int = 10) -> list[dict]:
        return list(self.searches_today[-n:])
