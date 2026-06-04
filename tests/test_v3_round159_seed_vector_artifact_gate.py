from adapters.fasttext_embedding_adapter import FasttextEmbeddingAdapter
from adapters.seed_vector_artifact_readiness import (
    ROUND159_ARTIFACT_GATE_VERSION,
    build_seed_vector_artifact_readiness_gate,
)
from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME


def test_round159_medium_artifact_gate_blocks_missing_real_vectors_without_dummy_data():
    gate = build_seed_vector_artifact_readiness_gate(subset_names=(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,))

    assert gate["version"] == ROUND159_ARTIFACT_GATE_VERSION
    assert gate["status"] == "blocked_operator_artifact_required"
    assert gate["ready"] is False
    assert gate["load_should_be_attempted"] is False
    assert "operator_artifact_files_missing" in gate["readiness_blockers"]
    assert any(path.endswith("vectors.npy") for path in gate["missing_files"])
    assert gate["dummy_vectors_created"] is False
    assert gate["vectors_npy_written"] is False
    assert gate["manifest_mutated"] is False
    assert gate["fasttext_runtime_imported"] is False
    assert gate["skip_or_xfail_used"] is False


def test_round159_adapter_readiness_is_skip_free_and_does_not_load_adapter():
    adapter = FasttextEmbeddingAdapter()
    gate = adapter.artifact_readiness()

    assert adapter.is_loaded() is False
    assert gate["selected_subsets"][0]["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert gate["selected_subsets"][0]["vectors_file_present"] is False
    assert gate["selected_subsets"][0]["load_should_be_attempted"] is False
    assert gate["production_persistence_enabled"] is False
    assert gate["runtime_mapping_enabled_default"] is False
    assert gate["enforcement_enabled_default"] is False
