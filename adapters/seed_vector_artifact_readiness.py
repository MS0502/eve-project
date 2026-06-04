"""Seed/vector operator-artifact readiness gate for EVE v3 rounds157-159.

The gate is intentionally read-only. It diagnoses whether registered fastText
subset artifacts are present and checksum-ready before any runtime load is
attempted. Missing operator-owned vectors are reported as an honest blocked
state instead of being replaced with dummy vectors or test skips.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    audit_subset_artifact,
    load_manifest_file,
)

ROUND159_ARTIFACT_GATE_VERSION = "v3_round159_seed_vector_artifact_readiness_gate"
ROUND159_DEFAULT_SUBSET_PATHS = {
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME: FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME: FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME: FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
}
ROUND159_FORBIDDEN_SYNTHETIC_ACTIONS = (
    "create_dummy_vectors",
    "fabricate_vector_checksums",
    "commit_vectors_npy",
    "skip_or_xfail_artifact_tests",
)


def _entry_for_subset(manifest: dict[str, Any], subset_name: str) -> dict[str, Any] | None:
    for entry in manifest.get("seeds", []) if isinstance(manifest, dict) else []:
        if isinstance(entry, dict) and entry.get("name") == subset_name:
            return deepcopy(entry)
    return None


def _path_for_subset(subset_name: str, entry: dict[str, Any] | None, subset_dirs: dict[str, str | Path] | None) -> str:
    if subset_dirs and subset_name in subset_dirs:
        return str(subset_dirs[subset_name])
    if entry and entry.get("file_location"):
        return str(entry["file_location"])
    if subset_name in ROUND159_DEFAULT_SUBSET_PATHS:
        return str(ROUND159_DEFAULT_SUBSET_PATHS[subset_name])
    return str(Path("seeds") / "subsets" / subset_name)


def _classify_audit_errors(errors: Iterable[str]) -> tuple[str, list[str]]:
    unique_errors = sorted(set(str(error) for error in errors))
    if not unique_errors:
        return "ready", []

    blockers: list[str] = []
    if any(error.startswith("missing_") for error in unique_errors):
        blockers.append("operator_artifact_files_missing")
    if any("checksum_mismatch" in error for error in unique_errors):
        blockers.append("operator_artifact_checksum_mismatch")
    if any("vectors_shape_mismatch" in error or "vectors_dtype_mismatch" in error for error in unique_errors):
        blockers.append("operator_artifact_vector_format_invalid")
    if any(error.startswith("manifest:") or error.startswith("subset:") or error == "subset_not_registered" for error in unique_errors):
        blockers.append("manifest_or_subset_registration_invalid")
    if any(error.startswith("vectors_load_failed:") for error in unique_errors):
        blockers.append("operator_artifact_vector_load_failed")
    if not blockers:
        blockers.append("operator_artifact_audit_failed")
    return "blocked", sorted(set(blockers))


def build_seed_vector_artifact_readiness_gate(
    *,
    manifest_data: dict[str, Any] | None = None,
    subset_names: Iterable[str] | None = None,
    subset_dirs: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return a read-only readiness gate for registered fastText subset files.

    The returned data is suitable for reports and validation JSON. It never
    creates vectors, never loads fastText, never changes manifest state, never
    mutates runtime mapping/enforcement defaults, and never emits skip/xfail
    instructions. If a real ``vectors.npy`` artifact is absent, the status is
    ``blocked_operator_artifact_required``.
    """

    manifest = deepcopy(manifest_data) if manifest_data is not None else load_manifest_file()
    selected_names = list(subset_names or (FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,))
    subset_results: list[dict[str, Any]] = []
    all_blockers: list[str] = []
    missing_files: list[str] = []

    for subset_name in selected_names:
        entry = _entry_for_subset(manifest, subset_name)
        subset_dir = _path_for_subset(subset_name, entry, subset_dirs)
        audit = audit_subset_artifact(manifest, subset_name=subset_name, subset_dir=subset_dir)
        classification, blockers = _classify_audit_errors(audit.get("errors", []))
        files = audit.get("file_checks", {}).get("files", {}) if isinstance(audit.get("file_checks"), dict) else {}
        subset_missing = [
            str(file_info.get("path"))
            for file_info in files.values()
            if isinstance(file_info, dict) and not file_info.get("exists")
        ]
        missing_files.extend(subset_missing)
        all_blockers.extend(blockers)
        subset_results.append(
            {
                "subset_name": subset_name,
                "subset_dir": subset_dir,
                "registered": entry is not None,
                "audit_valid": bool(audit.get("valid")),
                "readiness": classification,
                "blockers": blockers,
                "audit_errors": list(audit.get("errors", [])),
                "missing_files": sorted(subset_missing),
                "vectors_file_present": bool(files.get("vectors", {}).get("exists")) if isinstance(files.get("vectors"), dict) else False,
                "vocab_file_present": bool(files.get("vocab", {}).get("exists")) if isinstance(files.get("vocab"), dict) else False,
                "subset_manifest_file_present": bool(files.get("subset_manifest", {}).get("exists")) if isinstance(files.get("subset_manifest"), dict) else False,
                "load_should_be_attempted": bool(audit.get("valid")),
            }
        )

    ready = bool(subset_results) and all(item["readiness"] == "ready" for item in subset_results)
    status = "ready_for_explicit_operator_artifact_load" if ready else "blocked_operator_artifact_required"
    return {
        "version": ROUND159_ARTIFACT_GATE_VERSION,
        "rounds": [157, 158, 159],
        "gate_layer": "seed_vector_operator_artifact_readiness",
        "status": status,
        "ready": ready,
        "load_should_be_attempted": ready,
        "selected_subsets": subset_results,
        "readiness_blockers": sorted(set(all_blockers)),
        "missing_files": sorted(set(missing_files)),
        "safe_handling_decision": "require_real_operator_artifacts_or_keep_load_blocked",
        "forbidden_synthetic_actions": list(ROUND159_FORBIDDEN_SYNTHETIC_ACTIONS),
        "dummy_vectors_created": False,
        "vectors_npy_written": False,
        "manifest_mutated": False,
        "fasttext_runtime_imported": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "skip_or_xfail_used": False,
    }
