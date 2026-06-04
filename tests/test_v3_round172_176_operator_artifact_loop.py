from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
from adapters.operator_artifact_verification import (
    ROUND172_OPERATOR_ARTIFACT_VERIFICATION_VERSION,
    verify_operator_subset_artifact,
)
from adapters.seed_vector_restore_contract import build_round164_load_dependent_repair_preflight


def _sha256(path: Path) -> str:
    from adapters.external_seed_manifest import compute_seed_checksum

    return compute_seed_checksum(path)


def _manifest(tmp_path: Path, *, vectors_checksum: str = "SHA256:" + "0" * 64) -> dict:
    subset_dir = tmp_path / "operator_subset"
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


def test_round172_operator_artifact_verification_blocks_when_local_artifacts_missing(tmp_path: Path):
    missing_dir = tmp_path / "missing_operator_subset"
    result = verify_operator_subset_artifact(
        subset_dir=missing_dir,
        manifest_data=_manifest(tmp_path),
        include_git_status=False,
    )

    assert result["version"] == ROUND172_OPERATOR_ARTIFACT_VERIFICATION_VERSION
    assert result["status"] == "blocked_operator_artifact_required"
    assert result["ready"] is False
    assert result["load_should_be_attempted"] is False
    assert result["files"]["vectors"]["exists"] is False
    assert result["vectors"]["shape"] is None
    assert "operator_artifact_files_missing" in result["blockers"]
    assert result["artifacts_written"] is False
    assert result["production_persistence_enabled"] is False
    assert result["runtime_mapping_enabled_default"] is False
    assert result["enforcement_enabled"] is False


def test_round172_operator_artifact_verification_green_for_consistent_local_artifacts(tmp_path: Path):
    subset_dir = tmp_path / "operator_subset"
    subset_dir.mkdir()
    vocab_path = subset_dir / "vocab.txt"
    manifest_path = subset_dir / "subset_manifest.json"
    vectors_path = subset_dir / "vectors.npy"
    vocab_path.write_text("민석\n오늘\n", encoding="utf-8")
    manifest_path.write_text('{"subset":"test"}\n', encoding="utf-8")
    np.save(vectors_path, np.zeros((2, 3), dtype=np.float32))
    manifest = _manifest(tmp_path, vectors_checksum=_sha256(vectors_path))
    manifest["seeds"][1]["vocab_checksum"] = _sha256(vocab_path)
    manifest["seeds"][1]["subset_manifest_checksum"] = _sha256(manifest_path)

    result = verify_operator_subset_artifact(
        subset_dir=subset_dir,
        manifest_data=manifest,
        include_git_status=False,
    )

    assert result["status"] == "operator_artifact_verified_green"
    assert result["ready"] is True
    assert result["load_should_be_attempted"] is True
    assert result["files"]["vectors"]["sha256_match"] is True
    assert result["vectors"]["shape"] == [2, 3]
    assert result["vectors"]["dtype"] == "float32"
    assert result["manifest_consistency"]["consistent"] is True


def test_round173_preflight_accepts_real_local_operator_path_when_gate_green(tmp_path: Path):
    subset_dir = tmp_path / "operator_subset"
    subset_dir.mkdir()
    vocab_path = subset_dir / "vocab.txt"
    manifest_path = subset_dir / "subset_manifest.json"
    vectors_path = subset_dir / "vectors.npy"
    vocab_path.write_text("민석\n오늘\n", encoding="utf-8")
    manifest_path.write_text('{"subset":"test"}\n', encoding="utf-8")
    np.save(vectors_path, np.ones((2, 3), dtype=np.float32))
    manifest = _manifest(tmp_path, vectors_checksum=_sha256(vectors_path))
    manifest["seeds"][1]["vocab_checksum"] = _sha256(vocab_path)
    manifest["seeds"][1]["subset_manifest_checksum"] = _sha256(manifest_path)

    gate = {
        "status": "ready_for_explicit_operator_artifact_load",
        "ready": True,
        "load_should_be_attempted": True,
        "readiness_blockers": [],
    }
    preflight = build_round164_load_dependent_repair_preflight(readiness_gate=gate)

    assert preflight["status"] == "load_dependent_repair_preflight_green"
    assert preflight["ready"] is True
    assert preflight["hard_block"] is False
    assert preflight["production_persistence_enabled"] is False
    assert preflight["runtime_mapping_enabled_default"] is False
    assert preflight["enforcement_enabled"] is False
