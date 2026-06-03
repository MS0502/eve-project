"""
CounterfactualAdapter

v40 Counterfactual (Lewis 1973, Pearl 3단계) ↔ v41 "만약?" 응답.

흐름:
1. LU가 입력에서 "만약 X 안 했으면" / "X 아니었다면" 패턴 추출
2. meaning.context["cf_query"] = X 박음
3. Planner가 분기 → cf.what_if_not(X) → core_message
"""

import re

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from counterfactual import Counterfactual

from utils.types import Meaning


# 만약 X 안 했으면 / X 없었으면 / X 아니었다면 / X 못 했으면
_CF_PATTERNS = [
    re.compile(r"만약\s+(\S+?)\s*(?:이|가)?\s*(?:안\s*했|없었|아니었|못\s*했)"),
    re.compile(r"(\S+?)\s+(?:안\s*했|없었|아니었)으면"),
]


def extract_cf_query(text: str) -> str | None:
    for p in _CF_PATTERNS:
        m = p.search(text)
        if m:
            target = m.group(1).strip()
            # 조사 떼기 단순 처리
            for suf in ("을", "를", "이", "가", "은", "는"):
                if target.endswith(suf) and len(target) > 1:
                    target = target[:-1]
                    break
            if target:
                return target
    return None


class CounterfactualAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 cf: Counterfactual | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if cf is not None:
            self.cf = cf
        else:
            from causal_graph import CausalGraph
            cg = CausalGraph(activation_adapter.sa)
            self.cf = Counterfactual(cg, activation_adapter.sa, hormone_adapter.hs)

    def enrich_meaning(self, meaning: Meaning):
        target = extract_cf_query(meaning.raw_text or "")
        if target:
            meaning.context = meaning.context or {}
            meaning.context["cf_query"] = target

    def what_if_not(self, category: str) -> str | None:
        """무슨 일 일어날지 짧은 한국어 한 줄로."""
        try:
            result = self.cf.what_if_not(category)
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        # affected categories 보고 짧게 표현
        affected = result.get("affected") or result.get("removed_descendants") or []
        if not affected:
            return f"{category} 없었으면 별로 달라지진 않았을 것 같아"
        if isinstance(affected, dict):
            affected = list(affected.keys())
        affected_str = ", ".join(list(affected)[:2])
        return f"{category} 없었으면 {affected_str}도 안 일어났을 텐데"
