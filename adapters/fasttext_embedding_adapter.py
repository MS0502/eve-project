"""FasttextEmbeddingAdapter scaffold — EVE v3 round32.

This module defines the 300d fastText subset boundary without switching any
runtime module away from the existing PMI+SVD ``SelfEmbeddingAdapter``.

Round32 scope:
- expose a future adapter interface
- keep the default subset as the registered medium 30k production lexical seed
- provide read-only migration/audit data
- do not load ``vectors.npy``
- do not mark any subset as used
- do not rewrite ``self_embedding_adapter.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:  # numpy is already a core project dependency, but keep import soft here.
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import boundary
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    SUBSET_STATE_EXTRACTED,
    audit_subset_artifact,
    assess_self_embedding_rewrite_readiness,
    compute_seed_checksum,
    load_manifest_file,
    subset_state,
)

FASTTEXT_EMBEDDING_ADAPTER_VERSION = "v3_round33_actual_load"
FASTTEXT_EMBEDDING_METHOD = "fasttext_extracted_subset"
FASTTEXT_EMBEDDING_DEFAULT_DIMENSION = FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM
FASTTEXT_EMBEDDING_DEFAULT_SUBSET = FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME

SELF_EMBEDDING_PUBLIC_METHODS = [
    "observe",
    "bootstrap_from_engine",
    "ensure_ready",
    "train",
    "get_embedding",
    "similarity",
    "most_similar",
    "text_embedding",
    "text_similarity",
    "stats",
]

FASTTEXT_EMBEDDING_PUBLIC_METHODS = [
    "load",
    "get_vector",
    "get_embedding",
    "similarity",
    "most_similar",
    "text_embedding",
    "text_similarity",
    "stats",
    "is_loaded",
    "get_dimension",
]

COMPATIBLE_EMBEDDING_METHODS = [
    "get_embedding",
    "similarity",
    "most_similar",
    "text_embedding",
    "text_similarity",
    "stats",
]

DIRECT_SELF_EMBEDDING_USAGE_HINTS = {
    "adapters/concept_memory_adapter.py": "observe newly defined concepts into self_embedding",
    "adapters/compositor_adapter.py": "uses get_embedding for semantic composition similarity",
    "adapters/attention_analyzer.py": "reads self_embedding for attention/semantic signals",
    "adapters/situation_responder.py": "uses self_embedding as learned-concept fallback",
    "adapters/state_debug_adapter.py": "reports self_embedding stats in debug dumps",
    "language/streaming.py": "observes input text into self_embedding during streaming",
    "main.py": "wires SelfEmbeddingAdapter into engine.self_embedding",
}

INDIRECT_SELF_EMBEDDING_AFFECTED_MODULES = [
    "agp_adapter",
    "activation_adapter",
    "concept_memory_adapter",
    "speech_hub",
    "compositor_adapter",
]


class FasttextEmbeddingAdapter:
    """300d fastText subset adapter.

    Round51 uses the committed medium 30k subset as the runtime primary, but still
    keeps the adapter separate from ``engine.self_embedding``. Loading is an
    explicit per-instance action and never rewrites the existing PMI+SVD path.
    """

    def __init__(
        self,
        engine: Any = None,
        subset_name: str = FASTTEXT_EMBEDDING_DEFAULT_SUBSET,
        subset_dir: str | Path = FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
        dimension: int = FASTTEXT_EMBEDDING_DEFAULT_DIMENSION,
    ):
        self.engine = engine
        self.subset_name = subset_name
        self.subset_dir = str(subset_dir)
        self.dimension = int(dimension)
        self._loaded = False
        self._vocab: Optional[list[str]] = None
        self._vocab_index: dict[str, int] = {}
        self._vectors: Any = None
        self._manifest_entry: Optional[dict[str, Any]] = None

    def load(self) -> bool:
        """Load the extracted subset files into this adapter instance.

        The load path is deterministic and fail-closed:
        1. find the subset entry in ``seeds/MANIFEST.yaml``
        2. re-audit files/checksums/shape/dtype
        3. read ``vocab.txt`` and ``vectors.npy``
        4. populate an in-memory index

        It does not mutate the manifest, does not import fastText, and does not
        replace ``engine.self_embedding``.
        """
        if self._loaded:
            return True
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy_unavailable")

        manifest = load_manifest_file()
        audit = audit_subset_artifact(
            manifest,
            subset_name=self.subset_name,
            subset_dir=self.subset_dir,
        )
        if not audit.get("valid"):
            raise ValueError(f"subset_audit_failed:{audit.get('errors', [])}")

        entries = [
            entry
            for entry in manifest.get("seeds", [])
            if isinstance(entry, dict) and entry.get("name") == self.subset_name
        ]
        if not entries:
            raise ValueError("subset_not_registered")
        entry = dict(entries[0])
        subset_path = Path(self.subset_dir)
        vocab_path = subset_path / str(entry.get("vocab_file", "vocab.txt"))
        vectors_path = subset_path / str(entry.get("vectors_file", "vectors.npy"))

        if compute_seed_checksum(vocab_path) != entry.get("vocab_checksum"):
            raise ValueError("vocab_checksum_mismatch")
        if compute_seed_checksum(vectors_path) != entry.get("vectors_checksum"):
            raise ValueError("vectors_checksum_mismatch")

        vocab = vocab_path.read_text(encoding="utf-8").splitlines()
        vectors = np.load(vectors_path)
        expected_shape = (int(entry.get("vocab_size")), int(entry.get("vector_dim")))
        if tuple(int(x) for x in vectors.shape) != expected_shape:
            raise ValueError("vectors_shape_mismatch")
        if str(vectors.dtype) != "float32":
            raise ValueError("vectors_dtype_mismatch")
        if len(vocab) != expected_shape[0]:
            raise ValueError("vocab_size_mismatch")
        if expected_shape[1] != self.dimension:
            raise ValueError("dimension_mismatch")

        self._vocab = list(vocab)
        self._vocab_index = {word: index for index, word in enumerate(self._vocab)}
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._manifest_entry = entry
        self._loaded = True
        return True

    def _require_loaded(self) -> None:
        if not self._loaded or self._vocab is None or self._vectors is None:
            raise RuntimeError("FasttextEmbeddingAdapter is not loaded")

    def get_vector(self, word: str):
        """Return a copy of the 300d vector for ``word`` or ``None`` if OOV."""
        self._require_loaded()
        index = self._vocab_index.get(str(word))
        if index is None:
            return None
        return self._vectors[index].astype(np.float32, copy=True)

    def get_embedding(self, word: str):
        """Compatibility alias for SelfEmbeddingAdapter.get_embedding."""
        return self.get_vector(word)

    def _cosine(self, vec_a, vec_b) -> float:
        if not NUMPY_AVAILABLE:
            return 0.0
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def similarity(self, word_a: str, word_b: str) -> float:
        """Deterministic cosine similarity. Unknown words return 0.0."""
        vec_a = self.get_vector(word_a)
        vec_b = self.get_vector(word_b)
        if vec_a is None or vec_b is None:
            return 0.0
        return self._cosine(vec_a, vec_b)

    def most_similar(self, word: str, top_n: int = 5, min_sim: float = 0.0) -> list:
        """Return deterministic nearest words by cosine similarity."""
        self._require_loaded()
        vec = self.get_vector(word)
        if vec is None:
            return []
        sims: list[tuple[str, float]] = []
        for other in self._vocab or []:
            if other == word:
                continue
            other_vec = self.get_vector(other)
            if other_vec is None:
                continue
            score = self._cosine(vec, other_vec)
            if score > min_sim:
                sims.append((other, score))
        sims.sort(key=lambda item: (-item[1], item[0]))
        return sims[: max(0, int(top_n))]

    def text_embedding(self, text: str):
        """Return a 300d mean embedding for whitespace tokens.

        Round33 intentionally uses only ``str.split()``. Korean morphology and
        fastText subword inference are later explicit rounds. Unknown tokens are
        skipped; an empty known-token set returns a deterministic zero vector.
        """
        self._require_loaded()
        vectors = []
        for token in str(text or "").split():
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return np.zeros(self.dimension, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def text_similarity(self, text_a: str, text_b: str) -> float:
        """Return deterministic cosine similarity between mean text vectors."""
        vec_a = self.text_embedding(text_a)
        vec_b = self.text_embedding(text_b)
        return self._cosine(vec_a, vec_b)

    def is_loaded(self) -> bool:
        return bool(self._loaded)

    def get_dimension(self) -> int:
        return self.dimension

    def stats(self) -> dict[str, Any]:
        return {
            "adapter": "FasttextEmbeddingAdapter",
            "version": FASTTEXT_EMBEDDING_ADAPTER_VERSION,
            "method": FASTTEXT_EMBEDDING_METHOD,
            "subset_name": self.subset_name,
            "dimension": self.dimension,
            "loaded": self.is_loaded(),
            "vocab_size": 0 if self._vocab is None else len(self._vocab),
            "vectors_shape": None if self._vectors is None else tuple(int(x) for x in self._vectors.shape),
            "runtime_used": False,
            "self_embedding_rewrite": False,
        }


def audit_interface_compatibility(engine: Any = None) -> dict[str, Any]:
    """Return read-only interface compatibility data for a future swap.

    This function is data-only. It does not instantiate the active adapter into
    the engine, does not load subset files, and does not mutate thresholds or
    state ladders.
    """
    manifest = load_manifest_file()
    readiness = assess_self_embedding_rewrite_readiness(engine, manifest)
    self_embedding = getattr(engine, "self_embedding", None) if engine is not None else None
    current_dim = getattr(self_embedding, "dim", 50) if self_embedding is not None else 50
    current_vocab_size = len(getattr(self_embedding, "word_counts", {}) or {}) if self_embedding is not None else 0
    current_trained = len(getattr(self_embedding, "embeddings", {}) or {}) if self_embedding is not None else 0
    adapter = FasttextEmbeddingAdapter(engine=engine)

    return {
        "version": FASTTEXT_EMBEDDING_ADAPTER_VERSION,
        "audit_layer": "self_embedding_interface_compatibility",
        "self_embedding_adapter": {
            "module": "adapters/self_embedding_adapter.py",
            "class": "SelfEmbeddingAdapter",
            "method": "PMI+SVD",
            "dimension": current_dim,
            "vocab_size": current_vocab_size,
            "trained_embeddings_count": current_trained,
            "public_methods": list(SELF_EMBEDDING_PUBLIC_METHODS),
        },
        "fasttext_embedding_adapter": {
            "module": "adapters/fasttext_embedding_adapter.py",
            "class": "FasttextEmbeddingAdapter",
            "method": FASTTEXT_EMBEDDING_METHOD,
            "dimension": adapter.get_dimension(),
            "default_subset": adapter.subset_name,
            "subset_state": subset_state(manifest, adapter.subset_name),
            "public_methods": list(FASTTEXT_EMBEDDING_PUBLIC_METHODS),
            "loaded": adapter.is_loaded(),
        },
        "compatible_methods": list(COMPATIBLE_EMBEDDING_METHODS),
        "incompatible_changes": {
            "dimension": f"{current_dim} -> {adapter.get_dimension()}",
            "vocab_size": f"{current_vocab_size} -> {FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE}",
            "training_method": "PMI+SVD local observations -> external-seed subset vectors",
        },
        "swap_strategy": "separate_adapter_then_module_by_module_wrapper",
        "swap_ready": False,
        "readiness": readiness.get("readiness"),
        "runtime_load": False,
        "self_embedding_rewrite": False,
        "state_transition": False,
    }


def audit_affected_modules(project_root: str | Path = ".") -> dict[str, Any]:
    """Return a read-only affected-module matrix for the 50d -> 300d migration."""
    root = Path(project_root)
    usage_sites: list[dict[str, Any]] = []
    for relative, usage in sorted(DIRECT_SELF_EMBEDDING_USAGE_HINTS.items()):
        path = root / relative
        occurrences = 0
        methods: list[str] = []
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            occurrences = text.count("self_embedding")
            for method in ("get_embedding", "similarity", "most_similar", "text_embedding", "text_similarity", "observe", "stats"):
                if method in text:
                    methods.append(method)
        risk = "medium" if relative in {"adapters/compositor_adapter.py", "adapters/attention_analyzer.py", "language/streaming.py"} else "low"
        usage_sites.append(
            {
                "module": relative,
                "usage": usage,
                "occurrences": occurrences,
                "methods": sorted(methods),
                "risk": risk,
            }
        )

    direct_modules = [site["module"] for site in usage_sites]
    return {
        "version": FASTTEXT_EMBEDDING_ADAPTER_VERSION,
        "audit_layer": "self_embedding_affected_modules",
        "directly_uses_self_embedding": direct_modules,
        "usage_sites": usage_sites,
        "indirectly_affected_through_categories": list(INDIRECT_SELF_EMBEDDING_AFFECTED_MODULES),
        "dimension_change_impact": {
            "current_dimension": 50,
            "target_dimension": FASTTEXT_EMBEDDING_DEFAULT_DIMENSION,
            "current_50d_usage_sites": direct_modules,
            "target_300d_compatibility": "wrapper_or_explicit_adapter_swap_required",
            "risk_per_site": {site["module"]: site["risk"] for site in usage_sites},
        },
        "migration_strategy": "module_by_module_with_wrapper",
        "runtime_load": False,
        "self_embedding_rewrite": False,
        "state_transition": False,
    }


def build_round32_embedding_boundary_report(engine: Any = None) -> dict[str, Any]:
    """Return a compact read-only report used by ROUND_V3_R32_REPORT."""
    return {
        "version": FASTTEXT_EMBEDDING_ADAPTER_VERSION,
        "round": "v3_round32",
        "scope": "self_embedding_rewrite_scaffold_and_300d_boundary_audit",
        "interface_compatibility": audit_interface_compatibility(engine),
        "affected_modules": audit_affected_modules(),
        "not_doing": [
            "FasttextEmbeddingAdapter actual load",
            "self_embedding_adapter rewrite",
            "engine.self_embedding swap",
            "subset runtime load",
            "AGP runtime change",
        ],
    }
