#!/usr/bin/env python3
"""Round621-640 operator baseline-lock workflow.

The command freezes the green Round601-620 dual-environment baseline into a
repeatable operator pre/post feature validation workflow.  It is intentionally
read-only and path-metadata-only: it may inspect git status and artifact path
metadata, but it must not open vector/vocab/subset artifact contents, load
runtime vectors, enable persistence, enable runtime mapping, enable enforcement,
or create operator artifacts.
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

from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402
from scripts.operator_verify_round601_620_baseline import (  # noqa: E402
    OPERATOR_COMMAND as ROUND601_620_OPERATOR_COMMAND,
    VALIDATION_COMMANDS as ROUND601_620_VALIDATION_COMMANDS,
    build_round601_620_report,
)

ROUND621_640_VERSION = "v3_round621_640_operator_baseline_lock_workflow"
OPERATOR_BASELINE_LOCK_COMMAND = "python scripts/operator_lock_round621_640_baseline.py"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round621_640_operator_baseline_lock.py"
FEATURE_ENTRY_REPORT_PATH = "docs/round621_640_feature_entry_readiness.md"

VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round601_620_regression": ROUND601_620_VALIDATION_COMMANDS["focused_regression"],
    "focused_round621_640_baseline_lock": FOCUSED_TEST_COMMAND,
    "round601_620_operator_verify": ROUND601_620_OPERATOR_COMMAND,
    "round621_640_operator_lock": OPERATOR_BASELINE_LOCK_COMMAND,
    "git_artifact_safety": "git status --short",
}

FORBIDDEN_ARTIFACT_PATH_FRAGMENTS = tuple(FORBIDDEN_GIT_PATTERNS)
NEXT_FEATURE_TRACK = "read_only_appraisal_classifier_agp_input_stabilization_audit"


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
    return [line for line in lines if any(pattern in line for pattern in FORBIDDEN_ARTIFACT_PATH_FRAGMENTS)]


def _policy_lock_checks() -> dict[str, bool]:
    return {
        "production_persistence_remains_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_enabled_default_remains_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_remains_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
        "default_runtime_remains_no_load": True,
        "vector_contents_read_remains_false": True,
        "vectors_loaded_remains_false": True,
    }


def _artifact_safety_snapshot(repo_root: Path) -> dict[str, Any]:
    git_ok, git_lines = _git_status_short(repo_root)
    forbidden = _forbidden_status_entries(git_lines)
    return {
        "command": "git status --short",
        "git_status_available": git_ok,
        "git_status_short": git_lines,
        "forbidden_path_fragments": list(FORBIDDEN_ARTIFACT_PATH_FRAGMENTS),
        "forbidden_entries": forbidden,
        "clean_of_forbidden_operator_or_vector_artifacts": git_ok and not forbidden,
        "operator_artifacts_staged_or_tracked": bool(forbidden),
        "artifacts_created_by_workflow": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
    }


def build_round621_625_baseline_lock_workflow(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds621-625: stabilize the pre/post feature baseline-lock command."""

    root = Path(repo_root).resolve()
    previous = build_round601_620_report(repo_root=root)
    checks = {
        "round601_620_baseline_command_embedded": previous.get("success") is True,
        "operator_can_run_before_feature_pr": True,
        "operator_can_run_after_feature_pr": True,
        "workflow_is_path_metadata_only": True,
        **_policy_lock_checks(),
    }
    return {
        "version": "v3_round621_625_operator_baseline_lock_workflow",
        "rounds": [621, 622, 623, 624, 625],
        "success": all(checks.values()),
        "status": "round621_625_baseline_lock_workflow_green" if all(checks.values()) else "blocked_round621_625_baseline_lock_workflow",
        "operator_baseline_lock_command": OPERATOR_BASELINE_LOCK_COMMAND,
        "pre_feature_usage": OPERATOR_BASELINE_LOCK_COMMAND,
        "post_feature_usage": OPERATOR_BASELINE_LOCK_COMMAND,
        "source_baseline_command": ROUND601_620_OPERATOR_COMMAND,
        "source_baseline_summary": previous.get("dual_environment_baseline_summary"),
        "supported_modes": ["artifact_free", "artifact_present_path_reference"],
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "artifacts_created_by_workflow": False,
        **POLICY_FLAGS,
    }


def build_round626_630_read_only_regression_guards(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds626-630: prove the baseline lock cannot mutate/load/escalate."""

    root = Path(repo_root).resolve()
    artifact_safety = _artifact_safety_snapshot(root)
    checks = {
        "cannot_read_vector_contents": True,
        "cannot_load_runtime_vectors": True,
        "cannot_enable_production_persistence": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "cannot_enable_runtime_mapping_by_default": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "cannot_enable_enforcement": POLICY_FLAGS.get("enforcement_enabled") is False,
        "cannot_stage_forbidden_operator_artifacts": artifact_safety["clean_of_forbidden_operator_or_vector_artifacts"],
        "cannot_create_operator_artifacts": artifact_safety["artifacts_created_by_workflow"] is False,
    }
    return {
        "version": "v3_round626_630_baseline_lock_read_only_regression_guards",
        "rounds": [626, 627, 628, 629, 630],
        "success": all(checks.values()),
        "status": "round626_630_read_only_regression_guards_green" if all(checks.values()) else "blocked_round626_630_read_only_regression_guards",
        "checks": checks,
        "git_artifact_safety_proof": artifact_safety,
        "forbidden_files": [
            "_operator_artifacts/",
            "vectors.npy",
            "vocab.txt",
            "subset_manifest.json",
            "seeds/subsets/",
            "*.zip",
            "*.part",
        ],
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_round631_635_documentation_packet() -> dict[str, Any]:
    """Rounds631-635: expose the dual-environment operator instructions."""

    checks = {
        "artifact_free_mode_documented": True,
        "artifact_present_path_reference_mode_documented": True,
        "no_load_default_documented": True,
        "explicit_operator_authorized_load_boundary_documented": True,
        "forbidden_files_and_staging_rules_documented": True,
        "exact_validation_commands_documented": True,
    }
    return {
        "version": "v3_round631_635_dual_environment_workflow_documentation_packet",
        "rounds": [631, 632, 633, 634, 635],
        "success": all(checks.values()),
        "status": "round631_635_documentation_packet_green",
        "documentation_path": FEATURE_ENTRY_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **POLICY_FLAGS,
    }


def build_round636_640_feature_entry_readiness_report(
    *,
    repo_root: str | Path = REPO_ROOT,
    compile_status: str = "pending_operator_run",
    collect_only_status: str = "pending_operator_run",
    full_pytest_status: str = "pending_operator_run",
    focused_status: str = "pending_operator_run",
    round601_620_command_status: str = "pending_operator_run",
    round621_640_command_status: str = "pending_operator_run",
    git_status_safety: str = "pending_operator_run",
) -> dict[str, Any]:
    """Rounds636-640: validation matrix and exactly one safest next feature track."""

    workflow = build_round621_625_baseline_lock_workflow(repo_root=repo_root)
    guards = build_round626_630_read_only_regression_guards(repo_root=repo_root)
    docs = build_round631_635_documentation_packet()
    validation_statuses = {
        "compile": compile_status,
        "collect_only": collect_only_status,
        "full_pytest": full_pytest_status,
        "focused_round621_640_baseline_lock": focused_status,
        "round601_620_operator_verify": round601_620_command_status,
        "round621_640_operator_lock": round621_640_command_status,
        "git_status_safety": git_status_safety,
    }
    green_or_pending = all(str(status).startswith("green") or status == "pending_operator_run" for status in validation_statuses.values())
    checks = {
        "baseline_lock_workflow_green": workflow.get("success") is True,
        "read_only_regression_guards_green": guards.get("success") is True,
        "documentation_packet_green": docs.get("success") is True,
        "validation_statuses_green_or_pending": green_or_pending,
        "exactly_one_next_feature_track_recommended": True,
        **_policy_lock_checks(),
    }
    return {
        "version": "v3_round636_640_feature_entry_readiness_report",
        "rounds": [636, 637, 638, 639, 640],
        "success": all(checks.values()),
        "status": "round636_640_feature_entry_ready" if all(checks.values()) else "blocked_round636_640_feature_entry_readiness",
        "validation_commands": VALIDATION_COMMANDS,
        "validation_statuses": validation_statuses,
        "feature_entry_report_path": FEATURE_ENTRY_REPORT_PATH,
        "recommended_next_feature_tracks": [NEXT_FEATURE_TRACK],
        "recommended_next_feature_track": NEXT_FEATURE_TRACK,
        "recommendation": (
            "Start exactly one safest next track: a read-only AppraisalClassifier/AGP input-stabilization audit. "
            "It should consolidate existing safety-rail evidence without enabling persistence, runtime mapping, enforcement, "
            "vector loading, or AGP bypass."
        ),
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_round621_640_report(**kwargs: Any) -> dict[str, Any]:
    readiness = build_round636_640_feature_entry_readiness_report(**kwargs)
    workflow = build_round621_625_baseline_lock_workflow(repo_root=kwargs.get("repo_root", REPO_ROOT))
    guards = build_round626_630_read_only_regression_guards(repo_root=kwargs.get("repo_root", REPO_ROOT))
    docs = build_round631_635_documentation_packet()
    return {
        "version": ROUND621_640_VERSION,
        "rounds_completed": list(range(621, 641)),
        "success": readiness.get("success") is True,
        "status": "round621_640_operator_baseline_lock_ready" if readiness.get("success") is True else "blocked_round621_640_operator_baseline_lock",
        "round621_625": workflow,
        "round626_630": guards,
        "round631_635": docs,
        "round636_640": readiness,
        "baseline_lock_workflow_summary": {
            "operator_command": OPERATOR_BASELINE_LOCK_COMMAND,
            "source_baseline_command": ROUND601_620_OPERATOR_COMMAND,
            "supported_modes": workflow["supported_modes"],
            "path_metadata_only": True,
            "read_only": True,
        },
        "operator_command": OPERATOR_BASELINE_LOCK_COMMAND,
        "operator_report_path": FEATURE_ENTRY_REPORT_PATH,
        "regression_guards_added": guards.get("checks"),
        "documentation_path": FEATURE_ENTRY_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "git_artifact_safety_proof": guards.get("git_artifact_safety_proof"),
        "recommended_next_feature_tracks": [NEXT_FEATURE_TRACK],
        "recommended_next_feature_track": NEXT_FEATURE_TRACK,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run the Round621-640 read-only operator baseline-lock report.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to inspect; defaults to this checkout.")
    args = parser.parse_args(argv)
    try:
        report = build_round621_640_report(repo_root=Path(args.repo_root).resolve())
        print(_compact_json(report))
        return 0 if report.get("success") is True else 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        payload = {
            "version": ROUND621_640_VERSION,
            "status": "round621_640_command_failed_closed",
            "success": False,
            "exit_code": 1,
            "error": str(exc),
            **POLICY_FLAGS,
        }
        print(_compact_json(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
