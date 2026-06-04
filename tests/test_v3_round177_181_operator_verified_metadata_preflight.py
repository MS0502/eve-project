from pathlib import Path

from adapters.operator_verified_artifact_evidence import (
    EXPECTED_MEDIUM_30K_VECTORS_SHA256,
    ROUND177_OPERATOR_VERIFIED_EVIDENCE_VERSION,
    ROUND178_LOAD_DEPENDENT_CLUSTER_SELECTION_VERSION,
    ROUND179_METADATA_ONLY_LOAD_PREFLIGHT_VERSION,
    build_round177_operator_verified_artifact_evidence,
    build_round179_metadata_only_load_preflight,
    select_round178_load_dependent_repair_cluster,
)


OPERATOR_CODESPACES_EVIDENCE = {
    "local_path": "_operator_artifacts/subset_medium_30k/",
    "files_exist": {
        "vocab.txt": True,
        "vectors.npy": True,
        "subset_manifest.json": True,
    },
    "vector_shape": [30000, 300],
    "vector_dtype": "float32",
    "vectors_sha256": EXPECTED_MEDIUM_30K_VECTORS_SHA256,
    "manifest_vectors_checksum": EXPECTED_MEDIUM_30K_VECTORS_SHA256,
    "manifest_vocab_size": 30000,
    "manifest_vector_dim": 300,
    "git_status_clean": True,
}


def test_round177_records_operator_verified_metadata_without_runtime_load() -> None:
    evidence = build_round177_operator_verified_artifact_evidence(evidence=OPERATOR_CODESPACES_EVIDENCE)

    assert evidence["version"] == ROUND177_OPERATOR_VERIFIED_EVIDENCE_VERSION
    assert evidence["status"] == "operator_verified_artifact_evidence_recorded"
    assert evidence["accepted_for_planning"] is True
    assert evidence["accepted_for_runtime_load"] is False
    assert evidence["files_exist"]["vectors.npy"] is True
    assert evidence["vector_shape"] == [30000, 300]
    assert evidence["vector_dtype"] == "float32"
    assert evidence["vectors_sha256"] == EXPECTED_MEDIUM_30K_VECTORS_SHA256
    assert evidence["artifact_files_included"] is False
    assert evidence["artifacts_written"] is False
    assert evidence["vectors_loaded"] is False
    assert evidence["production_persistence_enabled"] is False
    assert evidence["runtime_mapping_enabled_default"] is False
    assert evidence["enforcement_enabled"] is False
    assert evidence["agp_bypass_used"] is False


def test_round178_selects_one_metadata_only_load_dependent_cluster() -> None:
    evidence = build_round177_operator_verified_artifact_evidence(evidence=OPERATOR_CODESPACES_EVIDENCE)
    selection = select_round178_load_dependent_repair_cluster(operator_evidence=evidence)

    assert selection["version"] == ROUND178_LOAD_DEPENDENT_CLUSTER_SELECTION_VERSION
    assert selection["status"] == "load_dependent_cluster_selected_metadata_only"
    assert selection["selected_cluster"]["selected"] is True
    assert selection["selected_cluster"]["actual_load_allowed_by_selection"] is False
    assert selection["requires_local_artifact_access_before_load"] is True
    assert selection["artifacts_included"] is False
    assert selection["vectors_loaded"] is False
    assert selection["production_persistence_enabled"] is False
    assert selection["runtime_mapping_enabled_default"] is False
    assert selection["enforcement_enabled"] is False


def test_round179_hard_blocks_actual_load_when_artifacts_are_inaccessible(tmp_path: Path) -> None:
    evidence = build_round177_operator_verified_artifact_evidence(evidence=OPERATOR_CODESPACES_EVIDENCE)
    selection = select_round178_load_dependent_repair_cluster(operator_evidence=evidence)
    missing_dir = tmp_path / "missing_operator_artifacts"

    preflight = build_round179_metadata_only_load_preflight(
        operator_evidence=evidence,
        selected_cluster=selection,
        artifact_dir=missing_dir,
    )

    assert preflight["version"] == ROUND179_METADATA_ONLY_LOAD_PREFLIGHT_VERSION
    assert preflight["status"] == "hard_block_actual_load_artifacts_inaccessible"
    assert preflight["planning_ready_from_operator_evidence"] is True
    assert preflight["local_artifacts_accessible"] is False
    assert preflight["actual_load_allowed"] is False
    assert "operator_artifact_files_inaccessible_in_this_environment" in preflight["blockers"]
    assert preflight["vectors_loaded"] is False
    assert preflight["artifacts_written"] is False
    assert preflight["production_persistence_enabled"] is False
    assert preflight["runtime_mapping_enabled_default"] is False
    assert preflight["enforcement_enabled"] is False
    assert preflight["agp_bypass_used"] is False


def test_round179_can_defer_to_existing_readiness_gate_when_files_are_accessible(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "subset_medium_30k"
    artifact_dir.mkdir()
    for filename in ("vocab.txt", "vectors.npy", "subset_manifest.json"):
        (artifact_dir / filename).write_text("placeholder only for path-access preflight\n", encoding="utf-8")
    evidence = build_round177_operator_verified_artifact_evidence(evidence=OPERATOR_CODESPACES_EVIDENCE)
    selection = select_round178_load_dependent_repair_cluster(operator_evidence=evidence)

    preflight = build_round179_metadata_only_load_preflight(
        operator_evidence=evidence,
        selected_cluster=selection,
        artifact_dir=artifact_dir,
    )

    assert preflight["status"] == "metadata_preflight_ready_for_existing_readiness_gate"
    assert preflight["local_artifacts_accessible"] is True
    assert preflight["actual_load_allowed"] is False
    assert preflight["next_allowed_step_if_green"] == "run existing seed_vector_artifact_readiness gate, then explicit load in a separate focused repair"
    assert preflight["vectors_loaded"] is False
