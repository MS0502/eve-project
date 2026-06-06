"""Artifact-free fastText observation test double.

The stub represents an explicitly operator-authorized, already-loaded adapter
without creating or loading seed/vector artifacts. It exposes only the scalar
observation APIs used by round35-39 trace surfaces and returns no word vectors,
so tests preserve the no-default-vector invariant.
"""

from __future__ import annotations


class ExplicitLoadedFasttextObservationStub:
    """Loaded-adapter stub for observation-only tests without vector artifacts."""

    subset_name = "artifact_free_explicit_loaded_stub"
    dim = 300

    def is_loaded(self) -> bool:
        return True

    def stats(self) -> dict:
        return {
            "loaded": True,
            "subset_name": self.subset_name,
            "vocab_size": 0,
            "dimension": self.dim,
            "artifact_free_stub": True,
        }

    def get_vector(self, _word):
        return None

    def similarity(self, left, right) -> float:
        return self.text_similarity(str(left or ""), str(right or ""))

    def text_similarity(self, left, right) -> float:
        left_tokens = set(str(left or "").split())
        right_tokens = set(str(right or "").split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return float(len(overlap) / len(union))


def install_explicit_loaded_fasttext_stub(engine) -> ExplicitLoadedFasttextObservationStub:
    """Install a loaded fastText stand-in without touching seed/vector files."""
    stub = ExplicitLoadedFasttextObservationStub()
    engine.fasttext_embedding = stub
    return stub
