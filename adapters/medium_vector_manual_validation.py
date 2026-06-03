"""Round103 manual medium vector validation checkpoint.

The Round103 surface is a deterministic audit layer over the Round100/102
medium-vector restoration helpers.  It does not create vectors, copy operator
artifacts, weaken manifest checksums, or claim that runtime validation passed
unless the supplied status and vector audit prove the medium 30k asset is
present and checksum-valid.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.medium_vector_restoration import audit_operator_supplied_medium_vectors

ROUND103_MANUAL_VALIDATION_VERSION = "v3_round103_manual_medium_vector_validation"


def _load_json_payload(payload_or_path: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if payload_or_path is None:
        return {}
    if isinstance(payload_or_path, dict):
        return deepcopy(payload_or_path)
    path = Path(payload_or_path)
    if not path.is_file():
        return {"status_path": str(path), "status_file_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("status_path", str(path))
        data.setdefault("status_file_present", True)
        return data
    return {"status_path": str(path), "status_file_present": True, "status_payload_type": type(data).__name__}


def validate_round103_manual_medium_vectors(
    candidate_path: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    release_restore_status: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed manual validation decision for the medium vectors.

    ``candidate_path`` may point at an operator-supplied ``vectors.npy``.  When it
    is omitted the function audits the repository's ignored medium-vector target.
    The function never installs or rewrites the candidate; mutation remains an
    explicit operator action outside this read-only validator.
    """
    repo = Path(repo_root)
    candidate = Path(candidate_path) if candidate_path is not None else repo / "seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy"
    release_status = _load_json_payload(release_restore_status)
    vector_audit = audit_operator_supplied_medium_vectors(candidate, repo_root=repo)
    release_claims_restored = bool(release_status.get("hard_stop_released", False))
    vector_accepted = bool(vector_audit.get("acceptable_for_manual_install", False))
    release_status_name = str(release_status.get("status", "release_status_not_supplied"))

    blocking_reasons: list[str] = []
    if not release_status:
        blocking_reasons.append("release_restore_status_not_supplied")
    elif not release_claims_restored:
        blocking_reasons.append("release_restore_not_complete")
    if not vector_accepted:
        blocking_reasons.extend(str(r) for r in vector_audit.get("rejection_reasons", []) or ["medium_vectors_audit_failed"])

    validation_passed = bool(release_claims_restored and vector_accepted and not blocking_reasons)
    return {
        "validation_version": ROUND103_MANUAL_VALIDATION_VERSION,
        "round": 103,
        "candidate_path": str(candidate),
        "repo_root": str(repo),
        "release_restore_status": release_status_name,
        "release_restore_hard_stop_released": release_claims_restored,
        "vectors_audit": vector_audit,
        "validation_passed": validation_passed,
        "status": "manual_medium_vector_validation_passed" if validation_passed else "blocked_manual_medium_vector_validation_incomplete",
        "blocking_reasons": sorted(set(blocking_reasons)),
        "operator_next_step": "rerun_runtime_mapping_and_agp_validation" if validation_passed else "restore_verified_medium_vectors_before_runtime_mapping_persistence",
        "policy": {
            "read_only": True,
            "commit_vectors_npy_to_pr": False,
            "create_dummy_vectors": False,
            "relax_manifest_checksums": False,
            "weaken_tests": False,
            "agp_bypass": False,
        },
        "read_only": True,
    }


def write_round103_manual_medium_vector_validation(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("validation_version", ROUND103_MANUAL_VALIDATION_VERSION),
        "path": str(out),
        "json_written": True,
        "vectors_written": False,
        "binary_artifact_written_to_pr": False,
        "read_only": True,
    }


__all__ = [
    "ROUND103_MANUAL_VALIDATION_VERSION",
    "validate_round103_manual_medium_vectors",
    "write_round103_manual_medium_vector_validation",
]
