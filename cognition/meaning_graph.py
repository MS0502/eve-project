"""
Meaning Graph: Meaning -> Graph (노드 + 관계)

eve2.md 3.2:
- entity / action / emotion 모두 노드화
- 관계는 반드시 edge로 연결
"""

from utils.types import Meaning, MeaningNode, MeaningEdge


class MeaningGraph:

    def __init__(self):
        self.nodes: dict[str, MeaningNode] = {}
        self.edges: list[MeaningEdge] = []

    def add_node(self, node: MeaningNode):
        self.nodes[node.id] = node

    def connect(self, source_id: str, relation: str, target_id: str):
        self.edges.append(MeaningEdge(source_id, relation, target_id))

    def has(self, node_id: str) -> bool:
        return node_id in self.nodes

    def neighbors(self, node_id: str) -> list[tuple[str, str]]:
        return [(e.relation, e.target) for e in self.edges if e.source == node_id]

    def __repr__(self):
        return f"MeaningGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


def build_graph(meaning: Meaning) -> MeaningGraph:
    """Meaning -> MeaningGraph"""
    g = MeaningGraph()

    # entity 노드
    for ent in meaning.entities:
        g.add_node(MeaningNode(id=f"e:{ent}", type="entity", attributes={"text": ent}))

    # action 노드
    for act in meaning.actions:
        g.add_node(MeaningNode(id=f"a:{act}", type="action", attributes={"text": act}))

    # emotion 노드 + 관계 (감정은 가장 강한 entity에 has 관계)
    primary_entity = next(iter(meaning.entities), None)
    for emo, strength in meaning.emotions.items():
        emo_id = f"emo:{emo}"
        g.add_node(MeaningNode(
            id=emo_id, type="emotion",
            attributes={"name": emo, "strength": strength}
        ))
        if primary_entity:
            g.connect(f"e:{primary_entity}", "feels", emo_id)

    return g
