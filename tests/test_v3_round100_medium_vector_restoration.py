"""EVE v3 Round100 — medium vector restoration plan tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH
from adapters.medium_vector_restoration import (
    ROUND100_RESTORATION_VERSION,
    audit_operator_supplied_medium_vectors,
    build_round100_restoration_plan,
    scan_subset_artifact_paths,
    write_round100_restoration_status,
)


def test_round100_scan_separates_medium_and_small_artifact_status() -> None:
    scan = scan_subset_artifact_paths()

    assert scan["version"] == ROUND100_RESTORATION_VERSION
    assert scan["read_only"] is True
    assert scan["no_artifact_created"] is True
    rows = {row["subset_name"]: row for row in scan["subset_rows"]}
    medium = rows["cc.ko.300.subset.medium.30k"]
    small = rows["cc.ko.300.subset.small.5k"]
    assert medium["paths"]["vectors"].endswith(f"{FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH}/vectors.npy")
    assert medium["validation_tier"] == "medium_full_validation"
    assert small["validation_tier"] == "small_focused_validation_only"
    assert scan["medium_validation_status"] in {"available", "blocked_missing_or_invalid_vectors"}
    assert scan["small_focused_validation_status"] in {"available", "blocked_missing_or_invalid_vectors"}


def test_round100_missing_operator_candidate_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "vectors.npy"
    audit = audit_operator_supplied_medium_vectors(candidate)

    assert audit["read_only"] is True
    assert audit["acceptable_for_manual_install"] is False
    assert "candidate_vectors_file_missing" in audit["rejection_reasons"]
    assert audit["no_copy_performed"] is True
    assert audit["no_artifact_created"] is True
    assert audit["fake_or_dummy_artifact_allowed"] is False


def test_round100_wrong_shape_candidate_is_rejected_without_install(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.npy"
    np.save(candidate, np.zeros((1, 1), dtype=np.float32))

    audit = audit_operator_supplied_medium_vectors(candidate)

    assert audit["exists"] is True
    assert audit["acceptable_for_manual_install"] is False
    assert "vectors_shape_mismatch" in audit["rejection_reasons"]
    assert "vectors_checksum_mismatch" in audit["rejection_reasons"]
    assert audit["shape"] == [1, 1]
    assert audit["dtype"] == "float32"
    assert audit["no_copy_performed"] is True


def test_round100_plan_preserves_validation_boundaries() -> None:
    plan = build_round100_restoration_plan()

    assert plan["version"] == ROUND100_RESTORATION_VERSION
    assert plan["policy"]["commit_vectors_to_pr_diff"] is False
    assert plan["policy"]["create_dummy_vectors"] is False
    assert plan["policy"]["relax_manifest_checksums"] is False
    assert plan["policy"]["weaken_tests"] is False
    assert plan["policy"]["agp_bypass"] is False
    assert plan["validation_classification"]["round97_98_direct_cause"] == "missing_known_fasttext_context_vectors_when_subset_vectors_absent"
    assert plan["validation_classification"]["round92_98_direct_cause"] == "shared_setup_requires_round97_98_prerequisite_eve_specific_vector_commit"
    assert any("tests/test_v3_round50_subset_medium_30k.py" in step for step in plan["operator_restoration_steps"])


def test_round100_status_writer_writes_json_only(tmp_path: Path) -> None:
    output = tmp_path / "status.json"
    payload = build_round100_restoration_plan()
    result = write_round100_restoration_status(output, payload)

    assert result["json_written"] is True
    assert result["vectors_written"] is False
    assert result["no_artifact_created"] is True
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["version"] == ROUND100_RESTORATION_VERSION


def test_round101_workflow_prefers_release_or_operator_artifact_without_binary_commit() -> None:
    from adapters.medium_vector_restoration import (
        ROUND101_ARTIFACT_WORKFLOW_VERSION,
        build_round101_artifact_workflow,
    )

    workflow = build_round101_artifact_workflow(release_asset_url="https://github.com/MS0502/eve-project/releases/download/eve-v3-medium-vector-artifacts/cc.ko.300.subset.medium.30k.vectors.npy")

    assert workflow["version"] == ROUND101_ARTIFACT_WORKFLOW_VERSION
    assert workflow["preferred_distribution"]["github_release_asset_preferred"] is True
    assert workflow["preferred_distribution"]["repo_commit_binary_artifact"] is False
    assert workflow["preferred_distribution"]["split_binary_artifact_in_repo"] is False
    assert workflow["preferred_distribution"]["download_performed_by_helper"] is False
    assert workflow["safety_policy"]["dummy_vectors_allowed"] is False
    assert workflow["safety_policy"]["runtime_persistence_allowed"] is False
    assert workflow["expected_medium_artifact"]["shape"] == [30000, 300]
    assert workflow["expected_medium_artifact"]["dtype"] == "float32"
    assert workflow["validation_gate"]["runtime_mapping_persistence"].startswith("disabled")


def test_round101_workflow_rejects_missing_candidate_fail_closed(tmp_path: Path) -> None:
    from adapters.medium_vector_restoration import build_round101_artifact_workflow

    workflow = build_round101_artifact_workflow(candidate_path=tmp_path / "missing_vectors.npy")

    assert workflow["workflow_status"] == "blocked_waiting_for_operator_supplied_artifact"
    assert workflow["candidate_audit"]["acceptable_for_manual_install"] is False
    assert "candidate_vectors_file_missing" in workflow["candidate_audit"]["rejection_reasons"]
    assert workflow["validation_gate"]["medium_full_validation"] in {"available", "blocked"}
    assert workflow["no_artifact_created"] is True
