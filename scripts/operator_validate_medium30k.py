#!/usr/bin/env python3
"""Operator-local medium 30k artifact validation for EVE v3 rounds187-191.

This script is intentionally read-only by default. It validates the local
``_operator_artifacts/subset_medium_30k`` directory, runs the current canonical
seed/vector readiness and explicit-load preflight gates, and prints compact JSON.
It never creates, copies, downloads, or commits vector artifacts. Actual adapter
loading is opt-in with ``--attempt-load`` and remains guarded by preflight.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.explicit_load_guard import guarded_explicit_medium30k_load
from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
)
from adapters.operator_artifact_verification import (
    DEFAULT_OPERATOR_MEDIUM_30K_DIR,
    verify_operator_subset_artifact,
)
from adapters.seed_vector_artifact_readiness import build_seed_vector_artifact_readiness_gate
from adapters.seed_vector_restore_contract import build_round164_load_dependent_repair_preflight

ROUND187_191_OPERATOR_VALIDATE_MEDIUM30K_VERSION = "v3_round187_191_operator_validate_medium30k"
EXPECTED_MEDIUM30K_VECTORS_SHA256 = FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM
EXPECTED_MEDIUM30K_VECTORS_SHA256_BARE = EXPECTED_MEDIUM30K_VECTORS_SHA256.removeprefix("SHA256:")
EXPECTED_MEDIUM30K_SHAPE = [30000, 300]
EXPECTED_MEDIUM30K_DTYPE = "float32"

POLICY_FLAGS: dict[str, bool] = {
    "production_persistence_enabled": False,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled": False,
    "agp_bypass_used": False,
    "artifacts_written": False,
    "vectors_committed": False,
    "dummy_vectors_created": False,
}


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_status_safety(repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    """Return git-status safety for operator artifact and seed paths.

    Normal ``git status --short`` omits ignored operator artifacts, which is the
    desired safe state. Any status line for the guarded artifact/seed paths means
    an artifact or seed-subset path is staged, tracked-modified, deleted, or
    otherwise present in the PR diff, so the script fails closed.
    """

    guarded_paths = [
        "_operator_artifacts",
        str(artifact_dir / "vectors.npy"),
        str(artifact_dir / "vocab.txt"),
        str(artifact_dir / "subset_manifest.json"),
        "seeds/subsets",
    ]
    command = ["git", "status", "--short", "--", *guarded_paths]
    result = subprocess.run(command, cwd=repo_root, check=False, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    tracked_command = ["git", "ls-files", "--", "_operator_artifacts", str(artifact_dir / "vectors.npy"), str(artifact_dir / "vocab.txt"), str(artifact_dir / "subset_manifest.json")]
    tracked_result = subprocess.run(tracked_command, cwd=repo_root, check=False, capture_output=True, text=True)
    tracked_artifact_files = [line for line in tracked_result.stdout.splitlines() if line.strip()]
    return {
        "command": " ".join(command),
        "returncode": int(result.returncode),
        "lines": lines,
        "tracked_command": " ".join(tracked_command),
        "tracked_returncode": int(tracked_result.returncode),
        "tracked_artifact_files": tracked_artifact_files,
        "safe": result.returncode == 0 and tracked_result.returncode == 0 and not lines and not tracked_artifact_files,
        "guarded_paths": guarded_paths,
    }


def _blockers_from_report(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    verification = report.get("verification", {}) if isinstance(report.get("verification"), dict) else {}
    readiness = report.get("readiness", {}) if isinstance(report.get("readiness"), dict) else {}
    preflight = report.get("preflight", {}) if isinstance(report.get("preflight"), dict) else {}
    load_report = report.get("explicit_load", {}) if isinstance(report.get("explicit_load"), dict) else {}
    git_safety = report.get("git_status_safety", {}) if isinstance(report.get("git_status_safety"), dict) else {}

    blockers.extend(str(item) for item in verification.get("blockers", []) or [])
    blockers.extend(str(item) for item in readiness.get("readiness_blockers", []) or [])
    blockers.extend(str(item) for item in preflight.get("blockers", []) or [])
    blockers.extend(str(item) for item in load_report.get("blockers", []) or [])
    if not verification.get("ready"):
        blockers.append("operator_artifact_verification_not_green")
    if not readiness.get("ready"):
        blockers.append("seed_vector_readiness_not_green")
    if not preflight.get("ready"):
        blockers.append("round164_preflight_not_green")
    if not git_safety.get("safe"):
        blockers.append("git_status_safety_not_green")
    if report.get("attempt_load_requested") and not load_report.get("vectors_loaded"):
        blockers.append("explicit_load_not_successful")
    return sorted(set(blockers))


def run_validation(
    *,
    artifact_dir: str | Path = DEFAULT_OPERATOR_MEDIUM_30K_DIR,
    attempt_load: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run the read-only operator validation workflow and return JSON data."""

    repo = Path(repo_root)
    artifact_path = Path(artifact_dir)
    if not artifact_path.is_absolute():
        artifact_path = repo / artifact_path
    display_artifact_dir = str(Path(artifact_dir))

    verification = verify_operator_subset_artifact(subset_dir=artifact_path, include_git_status=False)
    git_safety = _git_status_safety(repo, Path(display_artifact_dir))
    readiness = build_seed_vector_artifact_readiness_gate(
        subset_names=(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,),
        subset_dirs={FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME: artifact_path},
    )
    preflight = build_round164_load_dependent_repair_preflight(readiness_gate=readiness)
    explicit_load = guarded_explicit_medium30k_load(
        readiness_gate=readiness,
        artifact_dir=artifact_path,
        attempt_load=attempt_load,
        attach_to_engine=False,
    )

    report: dict[str, Any] = {
        "version": ROUND187_191_OPERATOR_VALIDATE_MEDIUM30K_VERSION,
        "rounds_completed": [187, 188, 189, 190, 191],
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "artifact_dir": display_artifact_dir,
        "expected": {
            "required_files": ["vocab.txt", "vectors.npy", "subset_manifest.json"],
            "vectors_shape": EXPECTED_MEDIUM30K_SHAPE,
            "vectors_dtype": EXPECTED_MEDIUM30K_DTYPE,
            "vectors_sha256": EXPECTED_MEDIUM30K_VECTORS_SHA256,
            "vectors_sha256_bare": EXPECTED_MEDIUM30K_VECTORS_SHA256_BARE,
        },
        "default_read_only_no_load": not attempt_load,
        "attempt_load_requested": bool(attempt_load),
        "verification": verification,
        "git_status_safety": git_safety,
        "readiness": readiness,
        "preflight": preflight,
        "explicit_load": explicit_load,
        "one_command_workflow": "python scripts/operator_validate_medium30k.py",
        **POLICY_FLAGS,
    }
    blockers = _blockers_from_report(report)
    success = not blockers
    report.update(
        {
            "status": "operator_medium30k_validation_green" if success else "blocked_operator_medium30k_validation_failed",
            "success": success,
            "exit_code": 0 if success else 1,
            "blockers": blockers,
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the operator-local medium 30k artifact and guarded explicit-load path.")
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_OPERATOR_MEDIUM_30K_DIR),
        help="Operator-local artifact directory; default: _operator_artifacts/subset_medium_30k",
    )
    parser.add_argument(
        "--attempt-load",
        action="store_true",
        help="Opt in to guarded FasttextEmbeddingAdapter.load() after readiness/preflight are green.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_validation(artifact_dir=args.artifact_dir, attempt_load=args.attempt_load)
    except Exception as exc:  # fail closed while preserving compact machine-readable output
        report = {
            "version": ROUND187_191_OPERATOR_VALIDATE_MEDIUM30K_VERSION,
            "status": "blocked_operator_medium30k_validation_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["operator_validation_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            "attempt_load_requested": bool(getattr(args, "attempt_load", False)),
            **POLICY_FLAGS,
        }
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - exercised by operator command
    raise SystemExit(main())
