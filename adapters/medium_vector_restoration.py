"""Round100 medium vector artifact restoration and validation helpers.

This module is intentionally read-only.  It never creates ``vectors.npy``, never
copies an operator artifact into ``seeds/``, and never relaxes the fastText
manifest checksum/shape/dtype requirements.  Its job is to make the Round99
blocker explicit and give operators a deterministic way to verify a separately
supplied medium 30k vector artifact before any full validation is claimed.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM,
    compute_seed_checksum,
)

ROUND100_RESTORATION_VERSION = "v3_round100_medium_vector_restoration_plan"

SUBSET_ARTIFACT_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "subset_dir": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM,
        "expected_shape": (FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM),
        "runtime_role": "default_runtime_primary",
        "validation_tier": "medium_full_validation",
    },
    {
        "subset_name": FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
        "subset_dir": FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM,
        "expected_shape": (5000, 300),
        "runtime_role": "code_only_focused_fallback_if_vectors_exist",
        "validation_tier": "small_focused_validation_only",
    },
    {
        "subset_name": FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
        "subset_dir": FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_MANIFEST_CHECKSUM,
        "expected_shape": (1000, 300),
        "runtime_role": "fixture_artifact_only_not_runtime_fallback",
        "validation_tier": "fixture_boundary_validation_only",
    },
)


def _repo_path(repo_root: str | Path, relative: str | Path) -> Path:
    return Path(repo_root) / Path(relative)


def _checksum_if_file(path: Path) -> str | None:
    return compute_seed_checksum(path) if path.is_file() else None


def _vector_shape_dtype(path: Path) -> tuple[tuple[int, ...] | None, str | None, str | None]:
    if not path.is_file():
        return None, None, None
    try:
        vectors = np.load(path, mmap_mode="r")
        return tuple(int(x) for x in vectors.shape), str(vectors.dtype), None
    except Exception as exc:  # defensive data only; no import wrapping
        return None, None, f"vectors_load_failed:{type(exc).__name__}"


def scan_subset_artifact_paths(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return read-only path/checksum status for medium, small, and mini subsets."""
    root = Path(repo_root)
    rows: list[dict[str, Any]] = []
    for target in SUBSET_ARTIFACT_TARGETS:
        subset_dir = _repo_path(root, target["subset_dir"])
        vocab_path = subset_dir / "vocab.txt"
        vectors_path = subset_dir / "vectors.npy"
        subset_manifest_path = subset_dir / "subset_manifest.json"
        shape, dtype, load_error = _vector_shape_dtype(vectors_path)
        row = {
            "subset_name": target["subset_name"],
            "subset_dir": str(subset_dir),
            "runtime_role": target["runtime_role"],
            "validation_tier": target["validation_tier"],
            "expected_shape": list(target["expected_shape"]),
            "paths": {
                "vocab": str(vocab_path),
                "vectors": str(vectors_path),
                "subset_manifest": str(subset_manifest_path),
            },
            "exists": {
                "subset_dir": subset_dir.is_dir(),
                "vocab": vocab_path.is_file(),
                "vectors": vectors_path.is_file(),
                "subset_manifest": subset_manifest_path.is_file(),
            },
            "checksums": {
                "vocab_expected": target["vocab_checksum"],
                "vocab_actual": _checksum_if_file(vocab_path),
                "vectors_expected": target["vectors_checksum"],
                "vectors_actual": _checksum_if_file(vectors_path),
                "subset_manifest_expected": target["subset_manifest_checksum"],
                "subset_manifest_actual": _checksum_if_file(subset_manifest_path),
            },
            "vectors": {
                "shape": list(shape) if shape is not None else None,
                "dtype": dtype,
                "load_error": load_error,
            },
        }
        row["checksums"]["vocab_match"] = bool(row["checksums"]["vocab_actual"] == row["checksums"]["vocab_expected"])
        row["checksums"]["vectors_match"] = bool(row["checksums"]["vectors_actual"] == row["checksums"]["vectors_expected"])
        row["checksums"]["subset_manifest_match"] = bool(
            row["checksums"]["subset_manifest_actual"] == row["checksums"]["subset_manifest_expected"]
        )
        row["artifact_complete"] = bool(
            row["exists"]["vocab"]
            and row["exists"]["vectors"]
            and row["exists"]["subset_manifest"]
            and row["checksums"]["vocab_match"]
            and row["checksums"]["vectors_match"]
            and row["checksums"]["subset_manifest_match"]
            and row["vectors"]["shape"] == list(target["expected_shape"])
            and row["vectors"]["dtype"] == "float32"
        )
        rows.append(row)

    by_name = {row["subset_name"]: row for row in rows}
    medium = by_name[FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME]
    small = by_name[FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME]
    return {
        "version": ROUND100_RESTORATION_VERSION,
        "round": 100,
        "read_only": True,
        "repo_root": str(root),
        "subset_rows": rows,
        "medium_validation_status": "available" if medium["artifact_complete"] else "blocked_missing_or_invalid_vectors",
        "small_focused_validation_status": "available" if small["artifact_complete"] else "blocked_missing_or_invalid_vectors",
        "medium_vectors_present": bool(medium["exists"]["vectors"]),
        "small_vectors_present": bool(small["exists"]["vectors"]),
        "no_artifact_created": True,
    }


def audit_operator_supplied_medium_vectors(candidate_path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    """Fail-closed read-only audit for an operator-supplied medium vectors file.

    The candidate may live outside the repository.  A passing result means the
    operator may manually install that exact file at the manifest path; this
    function itself does not copy, move, or create anything.
    """
    path = Path(candidate_path)
    expected_shape = (FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM)
    exists = path.is_file()
    checksum_actual = _checksum_if_file(path)
    shape, dtype, load_error = _vector_shape_dtype(path)
    reasons: list[str] = []
    if not exists:
        reasons.append("candidate_vectors_file_missing")
    if checksum_actual != FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM:
        reasons.append("vectors_checksum_mismatch")
    if shape != expected_shape:
        reasons.append("vectors_shape_mismatch")
    if dtype != "float32":
        reasons.append("vectors_dtype_mismatch")
    if load_error:
        reasons.append(load_error)

    target_path = _repo_path(repo_root, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH) / "vectors.npy"
    return {
        "version": ROUND100_RESTORATION_VERSION,
        "round": 100,
        "read_only": True,
        "candidate_path": str(path),
        "target_path": str(target_path),
        "exists": bool(exists),
        "checksum_expected": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
        "checksum_actual": checksum_actual,
        "checksum_match": bool(checksum_actual == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM),
        "expected_shape": list(expected_shape),
        "shape": list(shape) if shape is not None else None,
        "dtype": dtype,
        "load_error": load_error,
        "acceptable_for_manual_install": not reasons,
        "rejection_reasons": reasons,
        "operator_action_if_acceptable": (
            f"copy candidate to {target_path} and rerun manifest/medium validation"
            if not reasons
            else None
        ),
        "no_copy_performed": True,
        "no_artifact_created": True,
        "fake_or_dummy_artifact_allowed": False,
    }


def build_round100_restoration_plan(
    repo_root: str | Path = ".",
    *,
    candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic restoration/validation plan without mutation."""
    scan = scan_subset_artifact_paths(repo_root)
    candidate_audit = (
        audit_operator_supplied_medium_vectors(candidate_path, repo_root=repo_root)
        if candidate_path is not None
        else None
    )
    medium_available = scan["medium_validation_status"] == "available"
    small_available = scan["small_focused_validation_status"] == "available"
    recommended_next_step = "rerun_medium_and_runtime_mapping_validation" if medium_available else "operator_restore_medium_vectors_artifact"
    if not medium_available and small_available:
        focused_status = "small_focused_validation_possible_medium_full_blocked"
    elif medium_available:
        focused_status = "medium_full_validation_possible"
    else:
        focused_status = "focused_runtime_mapping_validation_blocked_without_known_context_vectors"
    return {
        "version": ROUND100_RESTORATION_VERSION,
        "round": 100,
        "read_only": True,
        "policy": {
            "commit_vectors_to_pr_diff": False,
            "create_dummy_vectors": False,
            "relax_manifest_checksums": False,
            "weaken_tests": False,
            "agp_bypass": False,
            "runtime_mapping_persistence": False,
        },
        "scan": scan,
        "candidate_audit": candidate_audit,
        "validation_classification": {
            "medium_full_validation": "available" if medium_available else "blocked",
            "small_focused_validation": "available" if small_available else "blocked",
            "round97_98_direct_cause": "missing_known_fasttext_context_vectors_when_subset_vectors_absent",
            "round92_98_direct_cause": "shared_setup_requires_round97_98_prerequisite_eve_specific_vector_commit",
            "focused_status": focused_status,
        },
        "operator_restoration_steps": [
            "Obtain the original operator-extracted cc.ko.300.subset.medium.30k/vectors.npy artifact outside the PR diff.",
            "Run python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy to verify checksum, shape, and dtype.",
            f"Only if acceptable_for_manual_install is true, place the file at {FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH}/vectors.npy.",
            "Run tests/test_v3_round50_subset_medium_30k.py and tests/test_v3_round51_wrapper_primary_medium_swap.py for medium artifact validation.",
            "Then rerun Round97/98 and Round92~98 focused runtime-mapping validation.",
        ],
        "recommended_next_step": recommended_next_step,
        "hard_stop_if_vectors_absent": not medium_available,
        "no_artifact_created": True,
    }


def write_round100_restoration_status(path: str | Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a JSON status artifact only when explicitly called."""
    out = Path(path)
    payload = deepcopy(data) if data is not None else build_round100_restoration_plan()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "version": ROUND100_RESTORATION_VERSION,
        "path": str(out),
        "read_only": False,
        "json_written": True,
        "vectors_written": False,
        "no_artifact_created": True,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round100 medium vector restoration audit")
    parser.add_argument("--candidate", help="operator-supplied vectors.npy candidate path", default=None)
    parser.add_argument("--repo-root", help="repository root for target-path reporting", default=".")
    parser.add_argument("--output", help="optional JSON output path", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_round100_restoration_plan(repo_root=args.repo_root, candidate_path=args.candidate)
    text = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        write_round100_restoration_status(args.output, plan)
    print(text, end="")
    return 0 if not plan.get("hard_stop_if_vectors_absent", True) else 2


__all__ = [
    "ROUND100_RESTORATION_VERSION",
    "SUBSET_ARTIFACT_TARGETS",
    "audit_operator_supplied_medium_vectors",
    "build_round100_restoration_plan",
    "scan_subset_artifact_paths",
    "write_round100_restoration_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
