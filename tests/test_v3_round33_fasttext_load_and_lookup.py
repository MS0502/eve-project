"""EVE v3 round33 — FasttextEmbeddingAdapter fail-closed no-vector boundary.

The adapter may load an explicitly provided, checksum-valid operator artifact.
This repository intentionally does not commit ``vectors.npy`` fixtures, so the
base tests assert deterministic no-load behavior and honest load refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    SUBSET_STATE_EXTRACTED,
    audit_subset_artifact,
    load_manifest_file,
    subset_state,
)
from adapters.fasttext_embedding_adapter import (
    FASTTEXT_EMBEDDING_DEFAULT_SUBSET,
    FASTTEXT_EMBEDDING_METHOD,
    FasttextEmbeddingAdapter,
    audit_affected_modules,
    audit_interface_compatibility,
)
from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
SMALL_VECTORS = Path(FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH) / "vectors.npy"


def test_adapter_initial_state_is_unloaded_and_medium_by_default():
    adapter = FasttextEmbeddingAdapter()

    assert adapter.is_loaded() is False
    assert adapter.subset_name == FASTTEXT_EMBEDDING_DEFAULT_SUBSET == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert adapter.get_dimension() == 300
    assert adapter.stats()["method"] == FASTTEXT_EMBEDDING_METHOD
    assert adapter.stats()["vocab_size"] == 0
    assert adapter.stats()["vectors_shape"] is None


def test_small_subset_explicit_load_fails_closed_without_vectors():
    adapter = FasttextEmbeddingAdapter(subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME, subset_dir=FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH)

    assert not SMALL_VECTORS.exists()
    with pytest.raises(ValueError, match="missing_vectors_file"):
        adapter.load()
    assert adapter.is_loaded() is False
    assert adapter.stats()["vocab_size"] == 0


def test_lookup_methods_require_loaded_operator_artifact():
    adapter = FasttextEmbeddingAdapter()

    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.get_vector("사랑")
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.similarity("사랑", "사랑")
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.most_similar("사랑")
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.text_embedding("친구 사랑")
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.text_similarity("친구", "사랑")


def test_no_fake_zero_vectors_are_created_for_missing_operator_artifacts():
    adapter = FasttextEmbeddingAdapter()

    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.text_embedding("없는단어xyz 또없는단어")
    assert adapter.stats()["vectors_shape"] is None


def test_artifact_readiness_reports_missing_vectors_without_mutation():
    adapter = FasttextEmbeddingAdapter(subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME, subset_dir=FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH)
    before = adapter.stats()

    readiness = adapter.artifact_readiness()
    audit = audit_subset_artifact(subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME)

    assert readiness["ready"] is False
    assert "missing_vectors_file" in str(readiness)
    assert audit["valid"] is False
    assert "missing_vectors_file" in audit["errors"]
    assert adapter.stats() == before


def test_subset_state_remains_extracted_without_global_transition():
    manifest = load_manifest_file()
    adapter = FasttextEmbeddingAdapter(subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME, subset_dir=FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH)

    assert adapter.is_loaded() is False
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED


def test_engine_self_embedding_wrapper_exists_but_primary_is_unloaded_by_default():
    before_fasttext_imported = "fasttext" in sys.modules
    engine = build_full_engine()
    text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.fasttext_embedding.stats()["vocab_size"] == 0
    assert "FasttextEmbeddingAdapter" not in text
    assert "cc.ko.300.subset" not in text
    assert "vectors.npy" not in text
    assert ("fasttext" in sys.modules) == before_fasttext_imported


def test_interface_audit_remains_read_only_and_no_load():
    engine = build_full_engine()
    before_loaded = engine.fasttext_embedding.is_loaded()

    audit = audit_interface_compatibility(engine)
    affected = audit_affected_modules()

    assert audit["fasttext_embedding_adapter"]["loaded"] is False
    assert audit["runtime_load"] is False
    assert audit["self_embedding_rewrite"] is False
    assert affected["self_embedding_rewrite"] is False
    assert engine.fasttext_embedding.is_loaded() is before_loaded


def test_numpy_available_but_no_numpy_vector_file_committed():
    assert np.zeros(300, dtype=np.float32).shape == (300,)
    assert not SMALL_VECTORS.exists()
