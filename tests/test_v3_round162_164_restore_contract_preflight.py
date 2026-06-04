import hashlib
from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
from adapters.seed_vector_restore_contract import (
    ROUND162_RESTORE_CONTRACT_VERSION,
    ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION,
    ROUND164_LOAD_DEPENDENT_REPAIR_PREFLIGHT_VERSION,
    build_round162_operator_artifact_restore_contract,
    build_round163_post_restore_validation_schema,
    build_round164_load_dependent_repair_preflight,
)


def _sha256(path: Path) -> str:
    return "SHA256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_for_subset(tmp_path: Path, vectors_checksum: str = "SHA256:" + "0" * 64) -> dict:
    subset_dir = tmp_path / "seed_subset"
    return {
        "seeds": [
            {
                "name": "cc.ko.300.bin",
                "source": "https://fasttext.cc/docs/en/crawl-vectors.html",
                "license": "CC-BY-SA-3.0",
                "version": "2017-12",
                "downloaded_at": "2026-05-11",
                "checksum": "SHA256:" + "1" * 64,
                "imported_at_round": 24,
                "imported_at_patch": "v3_round24",
            },
            {
                "name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
                "file_location": str(subset_dir),
                "entry_type": "subset",
                "parent_seed": "cc.ko.300.bin",
                "parent_checksum": "SHA256:" + "1" * 64,
                "selection_method": "fasttext_frequency_order_top_k",
                "filter_rule": "remove_words_containing_unicode_replacement_char",
                "vocab_size": 2,
                "vector_dim": 3,
                "vocab_file": "vocab.txt",
                "vectors_file": "vectors.npy",
                "subset_manifest_file": "subset_manifest.json",
                "vocab_checksum": "SHA256:" + "0" * 64,
                "vectors_checksum": vectors_checksum,
                "subset_manifest_checksum": "SHA256:" + "0" * 64,
                "extracted_at": "2026-05-12",
                "status": "extracted",
                "imported_at_round": 50,
                "imported_at_patch": "v3_round50",
            },
        ]
    }


def test_round162_restore_contract_lists_exact_registered_paths_and_safety_boundary():
    contract = build_round162_operator_artifact_restore_contract(subset_names=(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,))

    assert contract["version"] == ROUND162_RESTORE_CONTRACT_VERSION
    assert contract["artifacts_included"] is False
    assert contract["production_persistence_enabled"] is False
    assert contract["runtime_mapping_enabled_default"] is False
    assert contract["enforcement_enabled"] is False
    assert contract["agp_bypass_used"] is False
    row = contract["restore_contract_rows"][0]
    assert row["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert row["expected_paths"]["vectors"].endswith("seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy")
    assert row["expected_manifest_fields"]["vectors_checksum"] == "SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05"
    assert row["shape_dtype_verification"]["vectors_expected_shape"] == [30000, 300]
    assert row["shape_dtype_verification"]["vectors_expected_dtype"] == "float32"
    assert "seeds/subsets/**/vectors.npy" in contract["no_commit_safety_list"]


def test_round163_post_restore_validation_schema_is_artifact_free_and_policy_locked():
    schema = build_round163_post_restore_validation_schema()

    assert schema["version"] == ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION
    assert schema["artifacts_included"] is False
    assert schema["vectors_written"] is False
    assert any(item["id"] == "sha256_match" and item["required"] for item in schema["checklist"])
    assert any(item["id"] == "shape_dtype_match" and item["required"] for item in schema["checklist"])
    assert "blocked_operator_artifact_required" in schema["json_schema"]["properties"]["overall_status"]["enum"]


def test_round164_preflight_hard_blocks_when_readiness_gate_is_red():
    preflight = build_round164_load_dependent_repair_preflight(
        readiness_gate={
            "status": "blocked_operator_artifact_required",
            "ready": False,
            "load_should_be_attempted": False,
            "readiness_blockers": ["operator_artifact_files_missing"],
        }
    )

    assert preflight["version"] == ROUND164_LOAD_DEPENDENT_REPAIR_PREFLIGHT_VERSION
    assert preflight["status"] == "hard_block_load_dependent_repair_until_artifacts_ready"
    assert preflight["hard_block"] is True
    assert preflight["ready"] is False
    assert preflight["allowed_when_green"] == []
    assert "load-dependent repair" in preflight["blocked_when_red"]
    assert preflight["production_persistence_enabled"] is False
    assert preflight["runtime_mapping_enabled_default"] is False
    assert preflight["enforcement_enabled"] is False


def test_round164_preflight_allows_load_dependent_work_only_after_real_gate_green(tmp_path: Path):
    subset_dir = tmp_path / "seed_subset"
    subset_dir.mkdir()
    (subset_dir / "vocab.txt").write_text("민석\n오늘\n", encoding="utf-8")
    (subset_dir / "subset_manifest.json").write_text('{"subset":"test"}\n', encoding="utf-8")
    np.save(subset_dir / "vectors.npy", np.zeros((2, 3), dtype=np.float32))
    manifest = _manifest_for_subset(tmp_path, vectors_checksum=_sha256(subset_dir / "vectors.npy"))
    manifest["seeds"][1]["vocab_checksum"] = _sha256(subset_dir / "vocab.txt")
    manifest["seeds"][1]["subset_manifest_checksum"] = _sha256(subset_dir / "subset_manifest.json")

    preflight = build_round164_load_dependent_repair_preflight(
        manifest_data=manifest,
        subset_names=(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,),
    )

    assert preflight["status"] == "load_dependent_repair_preflight_green"
    assert preflight["hard_block"] is False
    assert preflight["ready"] is True
    assert preflight["readiness_gate_status"] == "ready_for_explicit_operator_artifact_load"
    assert "explicit FasttextEmbeddingAdapter.load() with registered artifact path" in preflight["allowed_when_green"]
    assert preflight["vectors_written"] is False
