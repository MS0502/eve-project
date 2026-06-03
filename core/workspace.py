"""
Global Workspace (Baars 1988, Dehaene 2011)

eve2.md 3.3:
- 최신 입력 항상 attention(focus)
- working_memory 최대 10 유지

v41 첫 버전: 단순 리스트.
나중에 v40의 working_memory.py(Cowan, GNW broadcast)를 어댑터로 끼움.
"""

from utils.types import Meaning, WorkspaceState
from cognition.meaning_graph import MeaningGraph


WM_MAX = 10


class GlobalWorkspace:

    def __init__(self, hormone_adapter=None):
        self.working_memory: list[MeaningGraph] = []
        self.focus_graph: MeaningGraph | None = None
        self.last_meaning: Meaning | None = None
        self.hormone_adapter = hormone_adapter           # optional, v40 hormone_system
        self._fallback_hormone: dict[str, float] = {
            "stress": 0.0, "energy": 0.5, "curiosity": 0.4,
        }
        self.active_goals: list[str] = []

    @property
    def hormone(self) -> dict[str, float]:
        if self.hormone_adapter is not None:
            return self.hormone_adapter.as_dict()
        return self._fallback_hormone

    def update(self, graph: MeaningGraph, meaning: Meaning):
        self.working_memory.append(graph)
        if len(self.working_memory) > WM_MAX:
            self.working_memory.pop(0)
        self.focus_graph = graph
        self.last_meaning = meaning

    def get_state(self) -> WorkspaceState:
        focus_id = None
        if self.focus_graph and self.focus_graph.nodes:
            # 첫 emotion 노드를 focus로 (없으면 첫 entity)
            for nid, n in self.focus_graph.nodes.items():
                if n.type == "emotion":
                    focus_id = nid
                    break
            if focus_id is None:
                focus_id = next(iter(self.focus_graph.nodes))
        return WorkspaceState(
            focus=focus_id,
            working_memory=list(self.working_memory),
            hormone=dict(self.hormone),
            active_goals=list(self.active_goals),
            meaning=self.last_meaning,
        )
