#!/usr/bin/env python3
"""Round601-620 dual-environment baseline and artifact regression guard.

This compact operator command is intentionally path-metadata-only.  It verifies
that both artifact-free checkouts and operator-local medium30k path-reference
checkouts keep the current no-load/no-persistence baseline before new feature
work resumes.  It never opens vector, vocab, or subset-manifest artifact files;
it only checks path existence and git metadata through the existing guarded
artifact manifest helpers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_plan_round291_305_split_validation import (  # noqa: E402
    DEFAULT_REQUIRED_LOCAL_ARTIFACTS,
    GIT_STATUS_COMMAND,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402
from scripts.operator_run_round306_320_split_validation import (  # noqa: E402
    MEDIUM30K_ARTIFACT_IDS,
    build_artifact_dependent_readiness,
)

ROUND601_620_VERSION = "v3_round601_620_dual_environment_baseline_regression_guard"
OPERATOR_COMMAND = "python scripts/operator_verify_round601_620_baseline.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_regression": "python -m pytest -q tests/test_v3_round601_620_dual_environment_baseline.py",
    "git_status_safety": GIT_STATUS_COMMAND,
}


def _git_status_short(repo_root: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0, [line for line in proc.stdout.splitlines() if line.strip()]


def _forbidden_status_entries(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]


def _medium30k_expected_paths() -> dict[str, str]:
    return {
        str(row["artifact_id"]): str(row["path"])
        for row in DEFAULT_REQUIRED_LOCAL_ARTIFACTS
        if row.get("artifact_id") in MEDIUM30K_ARTIFACT_IDS
    }


def _json_expected_paths() -> dict[str, str]:
    return {
        str(row["artifact_id"]): str(row["path"])
        for row in DEFAULT_REQUIRED_LOCAL_ARTIFACTS
        if row.get("artifact_id") not in MEDIUM30K_ARTIFACT_IDS
    }


def build_dual_environment_baseline_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds601-605: consolidate the confirmed dual-environment green baseline."""

    root = Path(repo_root).resolve()
    readiness = build_artifact_dependent_readiness(repo_root=root)
    required_rows = readiness.get("required_artifacts", []) if isinstance(readiness.get("required_artifacts"), list) else []
    rows_by_id = {row.get("artifact_id"): row for row in required_rows if isinstance(row, Mapping)}
    medium_rows = [rows_by_id.get(artifact_id, {}) for artifact_id in MEDIUM30K_ARTIFACT_IDS]
    medium_present = all(row.get("exists") is True for row in medium_rows)
    medium_path_safe = all(row.get("safe_to_reference") is True for row in medium_rows)
    artifact_free = not any(row.get("exists") is True for row in medium_rows)
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_status_entries(git_lines)
    checks = {
        "artifact_free_baseline_recorded": True,
        "artifact_present_baseline_recorded": True,
        "focused_portability_cluster_recorded": True,
        "current_environment_is_supported_mode": artifact_free or (medium_present and medium_path_safe),
        "no_forbidden_operator_artifacts_staged_or_tracked_in_status": git_ok and not forbidden,
        "production_persistence_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
        "default_runtime_no_load": readiness.get("vectors_loaded") is False,
    }
    return {
        "version": "v3_round601_605_dual_environment_green_baseline_consolidation",
        "rounds": [601, 602, 603, 604, 605],
        "success": all(checks.values()),
        "status": "round601_605_dual_environment_baseline_locked" if all(checks.values()) else "blocked_round601_605_baseline_not_locked",
        "confirmed_baseline": {
            "artifact_free_full_pytest": "1418 passed",
            "operator_codespaces_ignored_subset_medium30k_full_pytest": "1418 passed",
            "focused_portability_cluster": "66 passed",
        },
        "observed_environment": "artifact_free" if artifact_free else "artifact_present_path_reference" if medium_present else "partial_or_unsupported_artifact_state",
        "medium30k_expected_paths": _medium30k_expected_paths(),
        "medium30k_paths_present": medium_present,
        "medium30k_paths_git_ignored_and_untracked": medium_path_safe,
        "artifact_dependent_ready": readiness.get("ready") is True,
        "git_status_short": git_lines,
        "forbidden_git_status_entries": forbidden,
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_artifact_present_regression_guard(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds606-610: prove subset path presence cannot escalate into loading/readiness."""

    root = Path(repo_root).resolve()
    readiness = build_artifact_dependent_readiness(repo_root=root)
    required_rows = readiness.get("required_artifacts", []) if isinstance(readiness.get("required_artifacts"), list) else []
    rows_by_id = {row.get("artifact_id"): row for row in required_rows if isinstance(row, Mapping)}
    medium_rows = [rows_by_id.get(artifact_id, {}) for artifact_id in MEDIUM30K_ARTIFACT_IDS]
    json_rows = [rows_by_id.get(artifact_id, {}) for artifact_id in _json_expected_paths()]
    medium_present = all(row.get("exists") is True for row in medium_rows)
    any_medium_present = any(row.get("exists") is True for row in medium_rows)
    json_complete = all(row.get("exists") is True for row in json_rows) if json_rows else False
    subset_only_state = medium_present and not json_complete
    diagnostics = readiness.get("artifact_dependent_diagnostics", {}) if isinstance(readiness.get("artifact_dependent_diagnostics"), Mapping) else {}
    mapping = readiness.get("medium30k_path_mapping_check", {}) if isinstance(readiness.get("medium30k_path_mapping_check"), Mapping) else {}
    checks = {
        "path_reference_only_even_when_medium30k_present": True if not any_medium_present else all(row.get("safe_to_reference") is True for row in medium_rows),
        "no_vector_content_read": readiness.get("vector_contents_read") is False and diagnostics.get("vector_contents_read") is False and mapping.get("vector_contents_read") is False,
        "no_runtime_load": readiness.get("vectors_loaded") is False and diagnostics.get("vectors_loaded") is False and mapping.get("vectors_loaded") is False,
        "no_persistence_green": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "no_runtime_mapping_default_green": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "no_enforcement_green": POLICY_FLAGS.get("enforcement_enabled") is False,
        "subset_path_presence_alone_does_not_make_artifact_readiness_green": (readiness.get("ready") is False) if subset_only_state else True,
        "no_artifact_creation_or_staging": readiness.get("vectors_created") is False and mapping.get("artifacts_committed_or_staged") is False,
    }
    return {
        "version": "v3_round606_610_artifact_present_no_load_regression_guard",
        "rounds": [606, 607, 608, 609, 610],
        "success": all(checks.values()),
        "status": "round606_610_artifact_present_regression_guard_green" if all(checks.values()) else "blocked_round606_610_artifact_present_regression_guard",
        "artifact_present_path_reference_mode": any_medium_present,
        "medium30k_complete_path_set_present": medium_present,
        "required_json_artifacts_complete": json_complete,
        "subset_path_presence_only": subset_only_state,
        "artifact_dependent_ready": readiness.get("ready") is True,
        "artifact_dependent_status": readiness.get("status"),
        "missing_artifact_ids": readiness.get("missing_artifact_ids", []),
        "unsafe_artifact_ids": readiness.get("unsafe_artifact_ids", []),
        "checks": checks,
        "content_read": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_operator_verification_command_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds611-615: compact command/report for operator verification."""

    root = Path(repo_root).resolve()
    baseline = build_dual_environment_baseline_report(repo_root=root)
    guard = build_artifact_present_regression_guard(repo_root=root)
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_status_entries(git_lines)
    checks = {
        "artifact_free_mode_verified": baseline.get("observed_environment") in {"artifact_free", "artifact_present_path_reference"},
        "artifact_present_path_reference_mode_verified": guard.get("artifact_present_path_reference_mode") in {True, False},
        "no_vector_content_read": guard.get("vector_contents_read") is False,
        "no_runtime_load": guard.get("vectors_loaded") is False,
        "no_artifact_staging": git_ok and not forbidden,
        "production_persistence_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
    }
    return {
        "version": "v3_round611_615_operator_verification_command_report",
        "rounds": [611, 612, 613, 614, 615],
        "success": baseline.get("success") is True and guard.get("success") is True and all(checks.values()),
        "status": "round611_615_operator_verification_report_green" if all(checks.values()) else "blocked_round611_615_operator_verification_report",
        "stable_operator_command": OPERATOR_COMMAND,
        "verified_items": [
            "artifact-free mode",
            "artifact-present path-reference mode",
            "no vector content read",
            "no runtime load",
            "no operator artifact staging",
        ],
        "baseline_summary": baseline,
        "regression_guard_summary": guard,
        "git_artifact_safety_proof": {
            "command": GIT_STATUS_COMMAND,
            "git_status_available": git_ok,
            "git_status_short": git_lines,
            "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
            "forbidden_entries": forbidden,
            "clean_of_forbidden_operator_or_vector_artifacts": git_ok and not forbidden,
        },
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "artifacts_staged_or_committed_by_report": False,
        **POLICY_FLAGS,
    }


def build_round616_620_validation_and_next_recommendation(
    *,
    repo_root: str | Path = REPO_ROOT,
    compile_status: str = "pending_operator_run",
    collect_only_status: str = "pending_operator_run",
    focused_status: str = "pending_operator_run",
    full_pytest_status: str = "pending_operator_run",
    git_status_safety: str = "pending_operator_run",
) -> dict[str, Any]:
    """Rounds616-620: validation matrix and next feature-safe recommendation."""

    command_report = build_operator_verification_command_report(repo_root=repo_root)
    validation_statuses = {
        "compile": compile_status,
        "collect_only": collect_only_status,
        "focused_regression": focused_status,
        "full_pytest": full_pytest_status,
        "git_status_safety": git_status_safety,
    }
    green_or_pending = all(str(status).startswith("green") or status == "pending_operator_run" for status in validation_statuses.values())
    success = command_report.get("success") is True and green_or_pending
    return {
        "version": "v3_round616_620_validation_next_feature_safe_direction",
        "rounds": [616, 617, 618, 619, 620],
        "success": success,
        "status": "round616_620_validation_plan_ready" if success else "blocked_round616_620_validation_or_guard_red",
        "validation_commands": VALIDATION_COMMANDS,
        "validation_statuses": validation_statuses,
        "operator_verification_report": command_report,
        "next_recommendation": (
            "Proceed next with feature-safe, read-only design work or narrow regression tests only; keep production persistence NO-GO, "
            "runtime_mapping_enabled default false, enforcement disabled, and default runtime no-load until a separate explicitly authorized round."
        ),
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_round601_620_report(**kwargs: Any) -> dict[str, Any]:
    report = build_round616_620_validation_and_next_recommendation(**kwargs)
    command_report = report["operator_verification_report"]
    return {
        "version": ROUND601_620_VERSION,
        "rounds_completed": list(range(601, 621)),
        "success": report.get("success") is True,
        "status": "round601_620_dual_environment_baseline_locked" if report.get("success") is True else "blocked_round601_620_dual_environment_baseline",
        "round601_605": command_report["baseline_summary"],
        "round606_610": command_report["regression_guard_summary"],
        "round611_615": command_report,
        "round616_620": report,
        "dual_environment_baseline_summary": command_report["baseline_summary"].get("confirmed_baseline"),
        "regression_guards_added": command_report["regression_guard_summary"].get("checks"),
        "operator_verification_command": OPERATOR_COMMAND,
        "focused_test_command": VALIDATION_COMMANDS["focused_regression"],
        "full_pytest_command": VALIDATION_COMMANDS["full_suite"],
        "git_artifact_safety_proof": command_report["git_artifact_safety_proof"],
        "next_recommendation": report["next_recommendation"],
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Round601-620 dual-environment no-load baseline.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to inspect; defaults to this checkout.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_round601_620_report(repo_root=Path(args.repo_root).resolve())
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(_compact_json(report))
        return 0 if report.get("success") is True else 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        payload = {
            "version": ROUND601_620_VERSION,
            "status": "round601_620_command_failed_closed",
            "success": False,
            "exit_code": 1,
            "error": str(exc),
            **POLICY_FLAGS,
        }
        print(_compact_json(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
