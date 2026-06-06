"""Round172 read-only operator artifact verification helpers.

These helpers inspect an operator-provided fastText subset directory outside the
tracked seed/subset tree. They never create, copy, download, or commit artifacts;
they only report existence, SHA256, numpy shape/dtype, manifest consistency, and
basic git-status safety data for the supplied path.
"""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    audit_subset_artifact,
    compute_seed_checksum,
    load_manifest_file,
)

ROUND172_OPERATOR_ARTIFACT_VERIFICATION_VERSION = "v3_round172_operator_artifact_verification"
DEFAULT_OPERATOR_MEDIUM_30K_DIR = Path("_operator_artifacts/subset_medium_30k")
FORBIDDEN_TRACKED_ARTIFACT_PATTERNS = (
    "_operator_artifacts/",
    "seeds/subsets/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    ".zip",
    ".part",
)


def _entry_for_subset(manifest: dict[str, Any], subset_name: str) -> dict[str, Any] | None:
    for entry in manifest.get("seeds", []) if isinstance(manifest, dict) else []:
        if isinstance(entry, dict) and entry.get("name") == subset_name:
            return deepcopy(entry)
    return None


def _file_report(path: Path, expected_checksum: str | None = None) -> dict[str, Any]:
    exists = path.is_file()
    actual_checksum = compute_seed_checksum(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "sha256": actual_checksum,
        "expected_sha256": expected_checksum,
        "sha256_match": bool(exists and expected_checksum and actual_checksum == expected_checksum),
    }


def _vector_shape_dtype(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"shape": None, "dtype": None, "load_error": None}
    try:
        vectors = np.load(path, mmap_mode="r")
        return {
            "shape": [int(item) for item in vectors.shape],
            "dtype": str(vectors.dtype),
            "load_error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive read-only diagnostics
        return {"shape": None, "dtype": None, "load_error": type(exc).__name__}


def _git_status_short(paths: tuple[str, ...]) -> dict[str, Any]:
    command = ["git", "status", "--short", "--", *paths]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    forbidden_lines = [
        line
        for line in lines
        if any(pattern in line for pattern in FORBIDDEN_TRACKED_ARTIFACT_PATTERNS)
    ]
    return {
        "command": " ".join(command),
        "returncode": int(result.returncode),
        "lines": lines,
        "forbidden_artifact_lines": forbidden_lines,
        "safe": result.returncode == 0 and not forbidden_lines,
    }


def verify_operator_subset_artifact(
    *,
    subset_dir: str | Path = DEFAULT_OPERATOR_MEDIUM_30K_DIR,
    subset_name: str = FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    manifest_data: dict[str, Any] | None = None,
    include_git_status: bool = True,
) -> dict[str, Any]:
    """Return a read-only verification payload for a local operator subset.

    The supplied directory may be outside the registered manifest path. Manifest
    consistency is evaluated by auditing the supplied files against the
    registered subset entry and by comparing file checksums, vector shape, and
    dtype. Missing artifacts fail closed and do not trigger any load attempt.
    """

    manifest = deepcopy(manifest_data) if manifest_data is not None else load_manifest_file()
    entry = _entry_for_subset(manifest, subset_name)
    base_dir = Path(subset_dir)
    vocab_file = str(entry.get("vocab_file", "vocab.txt")) if entry else "vocab.txt"
    vectors_file = str(entry.get("vectors_file", "vectors.npy")) if entry else "vectors.npy"
    subset_manifest_file = str(entry.get("subset_manifest_file", "subset_manifest.json")) if entry else "subset_manifest.json"
    vocab_path = base_dir / vocab_file
    vectors_path = base_dir / vectors_file
    subset_manifest_path = base_dir / subset_manifest_file

    files = {
        "vocab": _file_report(vocab_path, entry.get("vocab_checksum") if entry else None),
        "vectors": _file_report(vectors_path, entry.get("vectors_checksum") if entry else None),
        "subset_manifest": _file_report(subset_manifest_path, entry.get("subset_manifest_checksum") if entry else None),
    }
    vector_info = _vector_shape_dtype(vectors_path)
    expected_shape = [int(entry.get("vocab_size")), int(entry.get("vector_dim"))] if entry else None
    manifest_audit = audit_subset_artifact(
        manifest,
        subset_name=subset_name,
        subset_dir=base_dir,
        allow_vector_content_read=True,
    )
    git_status = _git_status_short((str(base_dir), "seeds/subsets")) if include_git_status else None

    missing = [name for name, report in files.items() if not report["exists"]]
    checksum_mismatches = [name for name, report in files.items() if report["exists"] and not report["sha256_match"]]
    shape_match = bool(vector_info["shape"] == expected_shape)
    dtype_match = bool(vector_info["dtype"] == "float32")
    manifest_consistent = bool(manifest_audit.get("valid"))
    git_status_safe = True if git_status is None else bool(git_status.get("safe"))
    ready = bool(not missing and not checksum_mismatches and shape_match and dtype_match and manifest_consistent and git_status_safe)

    blockers: list[str] = []
    if missing:
        blockers.append("operator_artifact_files_missing")
    if checksum_mismatches:
        blockers.append("operator_artifact_checksum_mismatch")
    if not shape_match:
        blockers.append("operator_artifact_vector_shape_invalid")
    if not dtype_match:
        blockers.append("operator_artifact_vector_dtype_invalid")
    if not manifest_consistent:
        blockers.append("operator_artifact_manifest_inconsistent")
    if not git_status_safe:
        blockers.append("operator_artifact_git_status_unsafe")

    return {
        "version": ROUND172_OPERATOR_ARTIFACT_VERIFICATION_VERSION,
        "round": 172,
        "subset_name": subset_name,
        "subset_dir": str(base_dir),
        "registered": entry is not None,
        "directory_exists": base_dir.is_dir(),
        "files": files,
        "vectors": {
            "path": str(vectors_path),
            "shape": vector_info["shape"],
            "expected_shape": expected_shape,
            "shape_match": shape_match,
            "dtype": vector_info["dtype"],
            "expected_dtype": "float32",
            "dtype_match": dtype_match,
            "load_error": vector_info["load_error"],
        },
        "manifest_consistency": {
            "consistent": manifest_consistent,
            "audit_valid": bool(manifest_audit.get("valid")),
            "audit_errors": list(manifest_audit.get("errors", [])),
        },
        "git_status_safety": git_status,
        "ready": ready,
        "status": "operator_artifact_verified_green" if ready else "blocked_operator_artifact_required",
        "blockers": sorted(set(blockers)),
        "load_should_be_attempted": ready,
        "read_only": True,
        "artifacts_written": False,
        "vectors_committed": False,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
    }


__all__ = [
    "DEFAULT_OPERATOR_MEDIUM_30K_DIR",
    "ROUND172_OPERATOR_ARTIFACT_VERIFICATION_VERSION",
    "verify_operator_subset_artifact",
]
