"""
SemanticDistanceAdapter

v40 SemanticDistance (Tversky 1977 + path-based + neighbor overlap)
↔ v41 카테고리 유사도.

- similarity(a, b) → 0~1
- find_similar(x, k=5) → top-k 비슷한 것
- antonym_candidates → 반대 후보
- cluster_by_similarity → 자동 클러스터링

v41 통합:
- "X랑 비슷한 게 뭐야?" 질문 → find_similar
- VSA와 보완 (VSA는 합성, SD는 거리)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from semantic_distance import SemanticDistance


class SemanticDistanceAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 vsa_adapter=None,
                 hypergraph_adapter=None,
                 sd: SemanticDistance | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if sd is not None:
            self.sd = sd
        else:
            self.sd = SemanticDistance(
                activation_adapter.sa,
                vsa_binding=vsa_adapter.vsa if vsa_adapter else None,
                hypergraph=hypergraph_adapter.hg if hypergraph_adapter else None,
                hormone_system=hormone_adapter.hs,
            )

    def similarity(self, a: str, b: str) -> float:
        try:
            return float(self.sd.similarity(a, b))
        except Exception:
            return 0.0

    def distance(self, a: str, b: str) -> float:
        try:
            return float(self.sd.distance(a, b))
        except Exception:
            return 1.0

    def find_similar(self, target: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            return self.sd.find_similar(target, k=top_k) or []
        except Exception:
            return []

    def find_dissimilar(self, target: str, top_k: int = 5) -> list[tuple[str, float]]:
        try:
            return self.sd.find_dissimilar(target, k=top_k) or []
        except Exception:
            return []

    def antonyms(self, target: str, top_k: int = 3) -> list[tuple[str, float]]:
        try:
            return self.sd.antonym_candidates(target, k=top_k) or []
        except Exception:
            return []

    def cluster(self, threshold: float = 0.5) -> dict:
        try:
            return self.sd.cluster_by_similarity(threshold=threshold) or {}
        except Exception:
            return {}

    def similar_phrase(self, target: str) -> str | None:
        """비슷한 카테고리 한 줄 표현."""
        items = self.find_similar(target, top_k=3)
        if not items:
            return None
        names = [name for name, _ in items if name != target][:3]
        if not names:
            return None
        if len(names) == 1:
            return f"{target}이랑 비슷한 건 {names[0]} 정도"
        return f"{target}이랑 비슷한 건 {', '.join(names[:-1])}, {names[-1]} 같은 거"
