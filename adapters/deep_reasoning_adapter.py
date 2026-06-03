"""
DeepReasoningAdapter

v40 DeepReasoning ↔ v41 multi-hop 추론.
- find_path(start, end) — 두 카테고리 사이 인과 경로
- multi_hop_reason — N-hop 추론
- why_chain — 원인 체인
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from deep_reasoning import DeepReasoning


class DeepReasoningAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 nl_adapter=None,
                 dr: DeepReasoning | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if dr is not None:
            self.dr = dr
        else:
            from causal_graph import CausalGraph
            cg = CausalGraph(activation_adapter.sa)
            self.dr = DeepReasoning(
                activation_adapter.sa,
                cg,
                hormone_system=hormone_adapter.hs,
                natural_lang=nl_adapter.nl if nl_adapter else None,
            )

    def why_chain(self, target: str, max_depth: int = 3) -> list:
        try:
            return self.dr.why_chain(target, max_depth=max_depth) or []
        except Exception:
            return []

    def predict_chain(self, start: str, max_depth: int = 3) -> list:
        try:
            return self.dr.predict_chain(start, max_depth=max_depth) or []
        except Exception:
            return []

    def reason_phrase(self, target: str) -> str | None:
        """target의 원인 체인을 한 줄로."""
        chain = self.why_chain(target, max_depth=3)
        if not chain:
            return None
        # chain은 list of (node, weight) tuple 일 수도, dict일 수도
        nodes: list[str] = []
        for item in chain[:3]:
            if isinstance(item, (list, tuple)) and item:
                nodes.append(str(item[0]))
            elif isinstance(item, dict):
                n = item.get("category") or item.get("node") or item.get("from")
                if n:
                    nodes.append(str(n))
            elif isinstance(item, str):
                nodes.append(item)
        if len(nodes) < 2:
            return None
        return f"{nodes[0]}이 {target}로 이어지는 거 같아"
