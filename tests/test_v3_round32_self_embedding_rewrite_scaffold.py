"""EVE v3 round32/33 — self_embedding rewrite boundary audit.

Round32 added the separate FasttextEmbeddingAdapter scaffold. Round33 implements
explicit instance-level loading, but this older governance test still verifies
that the engine is not globally swapped and that normal runtime behavior remains
on the existing PMI+SVD adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    SEED_MANIFEST_PATH,
    SUBSET_STATE_EXTRACTED,
    load_manifest_file,
    subset_state,
)
from adapters.fasttext_embedding_adapter import (
    FASTTEXT_EMBEDDING_DEFAULT_DIMENSION,
    FASTTEXT_EMBEDDING_DEFAULT_SUBSET,
    FASTTEXT_EMBEDDING_METHOD,
    FasttextEmbeddingAdapter,
    audit_affected_modules,
    audit_interface_compatibility,
    build_round32_embedding_boundary_report,
)
from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")
FASTTEXT_ADAPTER = Path("adapters/fasttext_embedding_adapter.py")
REPORT = Path("ROUND_V3_R32_REPORT.md")


def test_fasttext_embedding_adapter_module_and_class_exist():
    assert FASTTEXT_ADAPTER.is_file()
    adapter = FasttextEmbeddingAdapter()

    assert adapter.__class__.__name__ == "FasttextEmbeddingAdapter"
    assert adapter.stats()["adapter"] == "FasttextEmbeddingAdapter"


def test_fasttext_embedding_adapter_dimension_and_default_subset():
    adapter = FasttextEmbeddingAdapter()

    assert adapter.get_dimension() == 300 == FASTTEXT_EMBEDDING_DEFAULT_DIMENSION
    assert adapter.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME == FASTTEXT_EMBEDDING_DEFAULT_SUBSET
    assert adapter.stats()["method"] == FASTTEXT_EMBEDDING_METHOD
    assert adapter.is_loaded() is False


def test_fasttext_embedding_adapter_methods_are_explicit_load_only():
    adapter = FasttextEmbeddingAdapter()

    assert adapter.is_loaded() is False
    with pytest.raises(RuntimeError):
        adapter.get_vector("친구")
    with pytest.raises(RuntimeError):
        adapter.get_embedding("친구")
    with pytest.raises(RuntimeError):
        adapter.similarity("친구", "사랑")
    with pytest.raises(RuntimeError):
        adapter.text_embedding("친구 사랑")
    with pytest.raises(RuntimeError):
        adapter.text_similarity("친구", "사랑")


def test_self_embedding_adapter_unchanged_and_not_swapped():
    engine = build_full_engine()
    text = SELF_EMBEDDING.read_text(encoding="utf-8")

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert "FasttextEmbeddingAdapter" not in text
    assert "cc.ko.300.subset" not in text
    assert "vectors.npy" not in text


def test_interface_compatibility_audit_returns_data_dict():
    engine = build_full_engine()
    audit = audit_interface_compatibility(engine)

    assert isinstance(audit, dict)
    assert audit["audit_layer"] == "self_embedding_interface_compatibility"
    assert audit["self_embedding_adapter"]["method"] == "PMI+SVD"
    assert audit["self_embedding_adapter"]["dimension"] == 300
    assert audit["fasttext_embedding_adapter"]["dimension"] == 300
    assert audit["fasttext_embedding_adapter"]["default_subset"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert audit["fasttext_embedding_adapter"]["subset_state"] == SUBSET_STATE_EXTRACTED
    assert "get_embedding" in audit["compatible_methods"]
    assert "similarity" in audit["compatible_methods"]
    assert audit["swap_ready"] is False
    assert audit["runtime_load"] is False
    assert audit["self_embedding_rewrite"] is False


def test_affected_modules_matrix_documents_known_usage_sites():
    matrix = audit_affected_modules()

    assert isinstance(matrix, dict)
    assert matrix["audit_layer"] == "self_embedding_affected_modules"
    direct = matrix["directly_uses_self_embedding"]
    assert "adapters/compositor_adapter.py" in direct
    assert "adapters/concept_memory_adapter.py" in direct
    assert "language/streaming.py" in direct
    assert matrix["dimension_change_impact"]["current_dimension"] == 50
    assert matrix["dimension_change_impact"]["target_dimension"] == 300
    assert matrix["migration_strategy"] == "module_by_module_with_wrapper"
    assert matrix["runtime_load"] is False
    assert matrix["self_embedding_rewrite"] is False


def test_small_subset_state_still_extracted_and_not_loaded():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    adapter = FasttextEmbeddingAdapter()

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED
    assert adapter.is_loaded() is False
    assert adapter.stats()["runtime_used"] is False
    assert adapter.stats()["self_embedding_rewrite"] is False


def test_readiness_acknowledges_scaffold_without_auto_apply():
    engine = build_full_engine()
    report = build_round32_embedding_boundary_report(engine)

    assert report["round"] in {"v3_round32", "v3_round33"}
    assert report["interface_compatibility"]["readiness"] == "needs_more_audit"
    assert report["interface_compatibility"]["swap_ready"] is False
    assert report["affected_modules"]["self_embedding_rewrite"] is False
    assert "self_embedding_adapter rewrite" in report["not_doing"]


def test_round32_audits_are_read_only_for_engine_manifest_and_imports():
    manifest_before = load_manifest_file(SEED_MANIFEST_PATH)
    engine = build_full_engine()
    before = {
        "self_embedding_class": engine.self_embedding.__class__.__name__,
        "self_embedding_dim": engine.self_embedding.dim,
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "fasttext_imported": "fasttext" in sys.modules,
    }

    _ = audit_interface_compatibility(engine)
    _ = audit_affected_modules()
    _ = build_round32_embedding_boundary_report(engine)
    manifest_after = load_manifest_file(SEED_MANIFEST_PATH)

    assert manifest_after == manifest_before
    assert engine.self_embedding.__class__.__name__ == before["self_embedding_class"]
    assert engine.self_embedding.dim == before["self_embedding_dim"]
    assert engine.agp_adapter.mode == before["agp_mode"]
    assert engine.compositor.agp_mode == before["compositor_mode"]
    assert engine.speech_hub.agp_mode == before["speech_hub_mode"]
    assert ("fasttext" in sys.modules) == before["fasttext_imported"]


def test_round32_report_documents_boundary_audit_and_non_goals():
    text = REPORT.read_text(encoding="utf-8")

    assert "self_embedding rewrite scaffold" in text
    assert "FasttextEmbeddingAdapter" in text
    assert "affected modules" in text
    assert "module_by_module_with_wrapper" in text
    assert "not doing" in text
