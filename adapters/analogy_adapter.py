"""
AnalogyAdapter

v40 Analogy (Gentner 1983 SME) ↔ v41 유추 응답.

직접 입력 패턴 매칭은 한국어로 빡세다. 일단 메서드만 노출,
외부에서 명시적으로 호출 가능하게.

향후 확장:
- "X는 Y처럼 ___" 패턴 인식
- planner sub_points에 유추 추가
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from analogy import Analogy


class AnalogyAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 an: Analogy | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if an is not None:
            self.an = an
        else:
            from causal_graph import CausalGraph
            cg = CausalGraph(activation_adapter.sa)
            self.an = Analogy(cg, activation_adapter.sa, hormone_adapter.hs)

    def find(self, a: str, b: str, c: str, top_n: int = 1) -> list[tuple[str, float]]:
        """A:B :: C:? — D 후보 찾기."""
        try:
            return self.an.find_analogy(a, b, c, top_n=top_n)
        except Exception:
            return []

    def complete(self, a: str, b: str, c: str) -> str | None:
        """A:B :: C:? — D 한 개."""
        try:
            return self.an.complete_proportion(a, b, c)
        except Exception:
            return None

    def explain(self, source_pair: tuple[str, str], target_anchor: str) -> dict | None:
        try:
            return self.an.map_relation(source_pair, target_anchor)
        except Exception:
            return None
