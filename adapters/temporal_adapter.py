"""
TemporalAdapter

v40 Temporal (Tulving 1985) ↔ v41 시간 추론.

흐름:
1. LU에서 시간 키워드 인식 ("내일", "어제", "전에", "다음에")
2. meaning.context["tm_query"] = {"direction": "past"|"future", "category": ...}
3. Planner: past → recall_past, future → imagine_future
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from temporal import Temporal

from utils.types import Meaning


PAST_KEYWORDS = ("어제", "전에", "예전", "지난번")
FUTURE_KEYWORDS = ("내일", "다음", "나중에", "앞으로")


class TemporalAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 memory_adapter=None,
                 tm: Temporal | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if tm is not None:
            self.tm = tm
        else:
            from causal_graph import CausalGraph
            em = memory_adapter.em if memory_adapter is not None else None
            cg = CausalGraph(activation_adapter.sa)
            self.tm = Temporal(em, activation_adapter.sa, cg, hormone_adapter.hs)

    def enrich_meaning(self, meaning: Meaning):
        text = meaning.raw_text or ""
        direction: str | None = None
        if any(k in text for k in PAST_KEYWORDS):
            direction = "past"
        elif any(k in text for k in FUTURE_KEYWORDS):
            direction = "future"
        if not direction:
            return
        # 첫 entity를 대상 카테고리로
        cat = meaning.entities[0] if meaning.entities else None
        meaning.context = meaning.context or {}
        meaning.context["tm_query"] = {"direction": direction, "category": cat}

    def recall_past_phrase(self, category: str | None) -> str | None:
        if not category:
            return "예전에 비슷한 거 있었던 거 같은데 뭐였더라"
        try:
            episodes = self.tm.recall_past(category, n=1)
        except Exception:
            return None
        if not episodes:
            return f"{category} 관련해서 예전 거는 잘 떠오르지 않아"
        # 첫 일화의 카테고리 중 하나로 표현
        ep = episodes[0]
        if isinstance(ep, dict):
            summary = ep.get("summary") or ""
            if summary:
                return f"전에 {category} 얘기 했던 거 떠오르네 ({summary[:30]})"
        return f"전에 {category} 얘기 했던 거 떠오르네"

    def imagine_future_phrase(self, seed: str | None) -> str | None:
        if not seed:
            return "앞으로 어떨지 같이 생각해볼까"
        try:
            result = self.tm.imagine_future(seed, depth=1)
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        # 결과에서 핵심 카테고리 뽑음
        next_cats = result.get("predicted_categories") or result.get("paths") or []
        if isinstance(next_cats, list) and next_cats:
            first = next_cats[0]
            if isinstance(first, dict):
                first = first.get("category", "")
            elif isinstance(first, (list, tuple)) and first:
                first = first[0] if isinstance(first[0], str) else str(first[0])
            if first:
                return f"{seed} 다음엔 {first} 올 수도 있겠네"
        return f"{seed} 다음이 어떨지 잘 모르겠네"
