"""Rounds177-179 metadata-only operator artifact evidence and load preflight.

These helpers record operator-side verification evidence for the medium 30k
fastText subset without requiring this execution environment to contain the
operator artifacts. They never create, copy, download, checksum, or load vector
artifacts. Actual runtime load remains hard-blocked unless the real files are
accessible to the local process and the existing readiness/load gates can be run
against them.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
from adapters.operator_artifact_verification import DEFAULT_OPERATOR_MEDIUM_30K_DIR

ROUND177_OPERATOR_VERIFIED_EVIDENCE_VERSION = "v3_round177_operator_verified_artifact_evidence"
ROUND178_LOAD_DEPENDENT_CLUSTER_SELECTION_VERSION = "v3_round178_load_dependent_cluster_selection"
ROUND179_METADATA_ONLY_LOAD_PREFLIGHT_VERSION = "v3_round179_metadata_only_load_preflight"

EXPECTED_MEDIUM_30K_VECTOR_SHAPE = [30000, 300]
EXPECTED_MEDIUM_30K_VECTOR_DTYPE = "float32"
EXPECTED_MEDIUM_30K_VECTORS_SHA256 = "SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05"
REQUIRED_OPERATOR_FILES = ("vocab.txt", "vectors.npy", "subset_manifest.json")
SELECTED_CLUSTER_ID = "fasttext_medium_30k_explicit_load_access_preflight"


def _normalize_sha256(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith("SHA256:") else f"SHA256:{text}"


def _bool(value: Any) -> bool:
    return bool(value is True)


def build_round177_operator_verified_artifact_evidence(
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only record of operator-side artifact verification.

    The input is operator-provided metadata, not the artifact files themselves.
    This function deliberately avoids filesystem reads beyond no-op metadata
    normalization, so it is safe in Codex Cloud where ``_operator_artifacts`` may
    be absent. Evidence can unblock planning, but it cannot authorize runtime
    load without local file accessibility and the existing artifact readiness
    gate.
    """

    source = deepcopy(evidence or {})
    local_path = str(source.get("local_path") or DEFAULT_OPERATOR_MEDIUM_30K_DIR)
    files_exist = {
        name: _bool(source.get(f"{name}_exists"))
        for name in REQUIRED_OPERATOR_FILES
    }
    if "files_exist" in source and isinstance(source["files_exist"], dict):
        files_exist.update({name: _bool(source["files_exist"].get(name)) for name in REQUIRED_OPERATOR_FILES})

    vector_shape = list(source.get("vector_shape") or [])
    vector_dtype = source.get("vector_dtype")
    vectors_sha256 = _normalize_sha256(source.get("vectors_sha256"))
    manifest_vectors_checksum = _normalize_sha256(source.get("manifest_vectors_checksum"))
    manifest_vocab_size = source.get("manifest_vocab_size")
    manifest_vector_dim = source.get("manifest_vector_dim")
    git_status_clean = _bool(source.get("git_status_clean"))

    checks = {
        "required_files_reported_present": all(files_exist.values()),
        "shape_matches_expected": vector_shape == EXPECTED_MEDIUM_30K_VECTOR_SHAPE,
        "dtype_matches_expected": vector_dtype == EXPECTED_MEDIUM_30K_VECTOR_DTYPE,
        "vectors_checksum_matches_expected": vectors_sha256 == EXPECTED_MEDIUM_30K_VECTORS_SHA256,
        "manifest_checksum_matches_vectors": manifest_vectors_checksum == vectors_sha256,
        "manifest_size_matches_expected": manifest_vocab_size == EXPECTED_MEDIUM_30K_VECTOR_SHAPE[0],
        "manifest_dim_matches_expected": manifest_vector_dim == EXPECTED_MEDIUM_30K_VECTOR_SHAPE[1],
        "operator_artifacts_git_status_clean": git_status_clean,
    }
    accepted = all(checks.values())
    blockers = sorted(name for name, passed in checks.items() if not passed)

    return {
        "version": ROUND177_OPERATOR_VERIFIED_EVIDENCE_VERSION,
        "round": 177,
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "evidence_layer": "operator_verified_metadata_only",
        "status": "operator_verified_artifact_evidence_recorded" if accepted else "operator_verified_artifact_evidence_incomplete",
        "accepted_for_planning": accepted,
        "accepted_for_runtime_load": False,
        "local_path": local_path,
        "files_exist": files_exist,
        "vector_shape": vector_shape,
        "expected_vector_shape": list(EXPECTED_MEDIUM_30K_VECTOR_SHAPE),
        "vector_dtype": vector_dtype,
        "expected_vector_dtype": EXPECTED_MEDIUM_30K_VECTOR_DTYPE,
        "vectors_sha256": vectors_sha256,
        "expected_vectors_sha256": EXPECTED_MEDIUM_30K_VECTORS_SHA256,
        "manifest_vectors_checksum": manifest_vectors_checksum,
        "manifest_vocab_size": manifest_vocab_size,
        "manifest_vector_dim": manifest_vector_dim,
        "git_status_clean": git_status_clean,
        "checks": checks,
        "blockers": blockers,
        "artifact_files_included": False,
        "artifacts_written": False,
        "vectors_loaded": False,
        "vectors_committed": False,
        "read_only": True,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


def select_round178_load_dependent_repair_cluster(
    *,
    operator_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Select one narrow load-dependent repair cluster from metadata evidence."""

    evidence = deepcopy(operator_evidence)
    accepted = bool(evidence.get("accepted_for_planning"))
    return {
        "version": ROUND178_LOAD_DEPENDENT_CLUSTER_SELECTION_VERSION,
        "round": 178,
        "selection_layer": "metadata_only_load_dependent_cluster_selection",
        "status": "load_dependent_cluster_selected_metadata_only" if accepted else "hard_block_no_load_dependent_cluster_selected",
        "selected_cluster": {
            "id": SELECTED_CLUSTER_ID,
            "name": "FasttextEmbeddingAdapter medium 30k explicit-load accessibility preflight",
            "selected": accepted,
            "repair_scope": [
                "verify operator evidence can drive planning without embedding vector files",
                "check local artifact accessibility before any explicit load attempt",
                "fail closed when _operator_artifacts/subset_medium_30k is unavailable in Codex Cloud",
            ],
            "load_dependent_behavior_prepared": accepted,
            "actual_load_allowed_by_selection": False,
        },
        "hard_block_reason": None if accepted else "operator_verified_metadata_not_accepted_for_planning",
        "operator_evidence_status": evidence.get("status"),
        "requires_local_artifact_access_before_load": True,
        "artifacts_included": False,
        "vectors_loaded": False,
        "read_only": True,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


def build_round179_metadata_only_load_preflight(
    *,
    operator_evidence: dict[str, Any],
    selected_cluster: dict[str, Any],
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed before load if operator artifacts are inaccessible locally.

    The function performs only path presence checks. It does not hash, mmap,
    ``numpy.load``, or instantiate ``FasttextEmbeddingAdapter.load()``. If all
    files are locally accessible, the result merely says a later explicit
    readiness gate may run; it still does not load vectors.
    """

    evidence = deepcopy(operator_evidence)
    cluster = deepcopy(selected_cluster)
    base_dir = Path(artifact_dir or evidence.get("local_path") or DEFAULT_OPERATOR_MEDIUM_30K_DIR)
    file_access = {name: (base_dir / name).is_file() for name in REQUIRED_OPERATOR_FILES}
    locally_accessible = all(file_access.values())
    planning_ready = bool(evidence.get("accepted_for_planning")) and bool(cluster.get("selected_cluster", {}).get("selected"))
    preflight_green = planning_ready and locally_accessible

    blockers: list[str] = []
    if not planning_ready:
        blockers.append("metadata_or_cluster_not_ready")
    if not locally_accessible:
        blockers.append("operator_artifact_files_inaccessible_in_this_environment")

    return {
        "version": ROUND179_METADATA_ONLY_LOAD_PREFLIGHT_VERSION,
        "round": 179,
        "preflight_layer": "metadata_only_actual_load_access_guard",
        "status": "metadata_preflight_ready_for_existing_readiness_gate" if preflight_green else "hard_block_actual_load_artifacts_inaccessible",
        "selected_cluster_id": SELECTED_CLUSTER_ID,
        "planning_ready_from_operator_evidence": planning_ready,
        "artifact_dir": str(base_dir),
        "local_file_access": file_access,
        "local_artifacts_accessible": locally_accessible,
        "actual_load_allowed": False,
        "next_allowed_step_if_green": "run existing seed_vector_artifact_readiness gate, then explicit load in a separate focused repair",
        "blockers": sorted(set(blockers)),
        "artifacts_included": False,
        "artifacts_written": False,
        "vectors_loaded": False,
        "vectors_committed": False,
        "read_only": True,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


__all__ = [
    "EXPECTED_MEDIUM_30K_VECTOR_DTYPE",
    "EXPECTED_MEDIUM_30K_VECTOR_SHAPE",
    "EXPECTED_MEDIUM_30K_VECTORS_SHA256",
    "REQUIRED_OPERATOR_FILES",
    "ROUND177_OPERATOR_VERIFIED_EVIDENCE_VERSION",
    "ROUND178_LOAD_DEPENDENT_CLUSTER_SELECTION_VERSION",
    "ROUND179_METADATA_ONLY_LOAD_PREFLIGHT_VERSION",
    "SELECTED_CLUSTER_ID",
    "build_round177_operator_verified_artifact_evidence",
    "build_round179_metadata_only_load_preflight",
    "select_round178_load_dependent_repair_cluster",
]
