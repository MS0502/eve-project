"""
HypergraphAdapter

v40 HyperGraph ↔ v41 n-ary 관계 저장.

일반 그래프 = 두 노드 사이 edge. HyperGraph = 여러 노드를 묶는 hyperedge.
예: {agent:민석, patient:PT, action:하다} 한 묶음을 한 edge로 저장.

v41 통합:
- FrameAdapter가 파싱한 프레임을 자동 저장 (장기 기억)
- query — "민석이 한 일들" 같은 role-based 검색
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from hypergraph import HyperGraph


class HypergraphAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 hg: HyperGraph | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if hg is not None:
            self.hg = hg
        else:
            self.hg = HyperGraph(
                spreading_activation=activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
            )

    def add_edge(self, nodes: set[str], roles: dict | None = None,
                 weight: float = 0.5) -> str | None:
        try:
            return self.hg.add_edge(nodes, roles=roles, weight=weight)
        except Exception:
            return None

    def query(self, role_filler: dict[str, str], top_k: int = 5) -> list:
        try:
            return self.hg.query(role_filler, top_k=top_k) or []
        except Exception:
            return []

    def find_with(self, node: str) -> list:
        try:
            return self.hg.find_edges_with(node) or []
        except Exception:
            return []

    def neighbors(self, node: str) -> set[str]:
        try:
            return self.hg.neighbors_via_edges(node) or set()
        except Exception:
            return set()

    def stats(self) -> dict:
        try:
            return self.hg.stats() or {}
        except Exception:
            return {}
