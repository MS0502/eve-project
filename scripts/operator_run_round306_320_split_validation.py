#!/usr/bin/env python3
"""Rounds306-320 split-validation execution and guarded local artifact entrypoints.

This command continues the green Round291-305 split-validation packet without
turning operator-local artifacts into production state.  It can run the
artifact-free validation phase, report artifact-dependent readiness separately,
and fail closed before any artifact-dependent command is considered unless all
required local files are present, git-ignored, and untracked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_plan_round291_305_split_validation import (  # noqa: E402
    COLLECT_ONLY_COMMAND,
    COMPILE_COMMAND,
    DEFAULT_REQUIRED_LOCAL_ARTIFACTS,
    FOCUSED_TEST_COMMAND as ROUND291_305_FOCUSED_TEST_COMMAND,
    FULL_SUITE_COMMAND,
    GIT_STATUS_COMMAND,
    ROUND291_305_VERSION,
    build_guarded_artifact_staged_rehearsal_manifest,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import (  # noqa: E402
    AUTHORIZATION_TOKEN as ROUND261_275_AUTHORIZATION_TOKEN,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

ROUND306_320_VERSION = "v3_round306_320_split_validation_execution_guard"
ARTIFACT_FREE_COMMAND = "python scripts/operator_run_round306_320_split_validation.py --round291-305-json _operator_artifacts/round291_305_split_validation_manifest.json --artifact-free"
ARTIFACT_DEPENDENT_COMMAND = (
    "python scripts/operator_run_round306_320_split_validation.py "
    "--round291-305-json _operator_artifacts/round291_305_split_validation_manifest.json "
    f"--artifact-dependent --operator-authorized --authorization-token {ROUND261_275_AUTHORIZATION_TOKEN}"
)
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round306_320_split_validation_execution.py"

Runner = Callable[..., subprocess.CompletedProcess[str]]

ARTIFACT_FREE_PHASE_COMMANDS: tuple[str, ...] = (
    COMPILE_COMMAND,
    COLLECT_ONLY_COMMAND,
    ROUND291_305_FOCUSED_TEST_COMMAND,
)

ARTIFACT_DEPENDENT_PLANNED_COMMANDS: tuple[str, ...] = (
    "python scripts/operator_run_local_validation_suite.py",
    "python scripts/operator_rehearse_runtime_mapping_no_persistence.py --handoff-json _operator_artifacts/round236_260_runtime_mapping_acceptance_handoff.json --operator-authorized "
    f"--authorization-token {ROUND261_275_AUTHORIZATION_TOKEN} --output _operator_artifacts/round306_320_no_persistence_runtime_mapping_rehearsal.json",
    FULL_SUITE_COMMAND,
    GIT_STATUS_COMMAND,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_command(command: str) -> list[str]:
    return command.split()


def _subprocess_runner(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_command(command: str, *, runner: Runner = _subprocess_runner, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    completed = runner(_split_command(command), cwd=Path(repo_root))
    return {
        "command": command,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def consolidate_round306_split_validation_green_evidence(round291_305_report: Mapping[str, Any]) -> dict[str, Any]:
    """Round306: consolidate the green split-validation evidence from Round291-305."""

    manifest_behavior = round291_305_report.get("artifact_staged_manifest_behavior")
    if not isinstance(manifest_behavior, Mapping):
        manifest_behavior = {}
    checks = {
        "source_version_round291_305": round291_305_report.get("version") == ROUND291_305_VERSION,
        "source_status_green": round291_305_report.get("status") == "round291_305_split_validation_staged_manifest_ready",
        "source_success_true": round291_305_report.get("success") is True,
        "source_exit_code_zero": round291_305_report.get("exit_code") == 0,
        "round291_305_completed": round291_305_report.get("rounds_completed") == list(range(291, 306)),
        "rollback_audit_green_evidence": round291_305_report.get("rollback_audit_green_evidence") is True,
        "artifact_manifest_fails_closed": manifest_behavior.get("fails_closed_when_required_artifacts_are_missing") is True,
        "artifact_manifest_does_not_read_vectors": manifest_behavior.get("does_not_read_vector_contents") is True,
        "artifact_manifest_does_not_fabricate_vectors": manifest_behavior.get("does_not_create_or_fabricate_vectors") is True,
        "production_persistence_no_go": round291_305_report.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": round291_305_report.get("runtime_mapping_enabled_default") is False,
        "enforcement_disabled": round291_305_report.get("enforcement_enabled") is False,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": "v3_round306_split_validation_green_evidence_consolidation",
        "round": 306,
        "success": not blockers,
        "status": "round306_split_validation_green_evidence_consolidated" if not blockers else "blocked_round306_split_validation_evidence_not_green",
        "source_version": round291_305_report.get("version"),
        "source_status": round291_305_report.get("status"),
        "checks": checks,
        "blockers": blockers,
        **POLICY_FLAGS,
    }


def build_artifact_dependent_readiness(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Return local-only artifact-dependent readiness without reading artifact contents."""

    manifest = build_guarded_artifact_staged_rehearsal_manifest(repo_root=repo_root)
    ready = manifest.get("staging_ready") is True
    return {
        "version": "v3_round307_315_artifact_dependent_readiness",
        "rounds": list(range(307, 316)),
        "success": ready,
        "status": "artifact_dependent_validation_ready" if ready else "fail_closed_artifact_dependent_validation_not_ready",
        "ready": ready,
        "fail_closed": not ready,
        "required_artifacts": manifest.get("required_artifacts", []),
        "missing_artifact_ids": manifest.get("missing_artifact_ids", []),
        "unsafe_artifact_ids": manifest.get("unsafe_artifact_ids", []),
        "artifact_git_guard": manifest.get("artifact_git_guard", {}),
        "planned_commands": list(ARTIFACT_DEPENDENT_PLANNED_COMMANDS),
        "content_read": False,
        "vectors_created": False,
        "vectors_loaded": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "blockers": manifest.get("blockers", []),
        **POLICY_FLAGS,
    }


def run_artifact_free_validation_phase(
    round291_305_report: Mapping[str, Any],
    *,
    repo_root: str | Path = REPO_ROOT,
    runner: Runner = _subprocess_runner,
) -> dict[str, Any]:
    """Rounds307-310: run artifact-free validation and report artifact readiness separately."""

    evidence = consolidate_round306_split_validation_green_evidence(round291_305_report)
    steps: list[dict[str, Any]] = []
    if evidence.get("success") is True:
        for command in ARTIFACT_FREE_PHASE_COMMANDS:
            step = _run_command(command, runner=runner, repo_root=repo_root)
            steps.append(step)
            if not step["success"]:
                break
    readiness = build_artifact_dependent_readiness(repo_root=repo_root)
    artifact_free_success = evidence.get("success") is True and len(steps) == len(ARTIFACT_FREE_PHASE_COMMANDS) and all(step["success"] for step in steps)
    blockers = list(evidence.get("blockers", []))
    blockers.extend(f"artifact_free_failed:{step['command']}" for step in steps if not step["success"])
    return {
        "version": "v3_round307_310_artifact_free_validation_command",
        "rounds": [307, 308, 309, 310],
        "success": artifact_free_success,
        "exit_code": 0 if artifact_free_success else 1,
        "status": "round307_310_artifact_free_validation_green" if artifact_free_success else "blocked_round307_310_artifact_free_validation",
        "stable_operator_command": ARTIFACT_FREE_COMMAND,
        "round306_evidence": evidence,
        "artifact_free_commands": list(ARTIFACT_FREE_PHASE_COMMANDS),
        "steps": steps,
        "artifact_dependent_readiness_reported_separately": True,
        "artifact_dependent_readiness": readiness,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "blockers": blockers,
        **POLICY_FLAGS,
    }


def run_guarded_artifact_dependent_entrypoint(
    round291_305_report: Mapping[str, Any],
    *,
    operator_authorized: bool,
    authorization_token: str | None,
    execute: bool = False,
    repo_root: str | Path = REPO_ROOT,
    runner: Runner = _subprocess_runner,
) -> dict[str, Any]:
    """Rounds311-315: fail closed before artifact-dependent validation can execute."""

    evidence = consolidate_round306_split_validation_green_evidence(round291_305_report)
    readiness = build_artifact_dependent_readiness(repo_root=repo_root)
    auth_ok = operator_authorized is True and authorization_token == ROUND261_275_AUTHORIZATION_TOKEN
    blockers = list(evidence.get("blockers", []))
    if not auth_ok:
        blockers.append("operator_authorization_missing_or_invalid")
    blockers.extend(readiness.get("blockers", []))

    steps: list[dict[str, Any]] = []
    can_execute = evidence.get("success") is True and readiness.get("ready") is True and auth_ok
    if can_execute and execute:
        for command in ARTIFACT_DEPENDENT_PLANNED_COMMANDS:
            step = _run_command(command, runner=runner, repo_root=repo_root)
            steps.append(step)
            if not step["success"]:
                blockers.append(f"artifact_dependent_failed:{command}")
                break

    success = can_execute and (not execute or (len(steps) == len(ARTIFACT_DEPENDENT_PLANNED_COMMANDS) and all(step["success"] for step in steps)))
    status = "round311_315_artifact_dependent_validation_ready" if success and not execute else "round311_315_artifact_dependent_validation_green" if success else "fail_closed_round311_315_artifact_dependent_validation"
    return {
        "version": "v3_round311_315_guarded_artifact_dependent_entrypoint",
        "rounds": [311, 312, 313, 314, 315],
        "success": success,
        "exit_code": 0 if success else 1,
        "status": status,
        "stable_operator_command": ARTIFACT_DEPENDENT_COMMAND,
        "execute_requested": execute,
        "operator_authorized": operator_authorized,
        "authorization_token_valid": authorization_token == ROUND261_275_AUTHORIZATION_TOKEN,
        "round306_evidence": evidence,
        "artifact_dependent_readiness": readiness,
        "planned_commands": list(ARTIFACT_DEPENDENT_PLANNED_COMMANDS),
        "steps": steps,
        "fail_closed": not success,
        "content_read_before_safety_gate": False,
        "vectors_created": False,
        "vectors_loaded_before_safety_gate": False,
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "blockers": sorted(set(blockers)),
        **POLICY_FLAGS,
    }


def build_round316_320_validation_delta(
    artifact_free_report: Mapping[str, Any] | None = None,
    artifact_dependent_report: Mapping[str, Any] | None = None,
    *,
    broader_pytest_status: str = "pending_operator_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    """Rounds316-320: summarize validation delta and next recommendation."""

    artifact_free_green = artifact_free_report is not None and artifact_free_report.get("success") is True
    artifact_dependent_ready = artifact_dependent_report is not None and artifact_dependent_report.get("success") is True
    artifact_dependent_fail_closed = artifact_dependent_report is not None and artifact_dependent_report.get("fail_closed") is True
    remaining_taxonomy = [
        "production_persistence_remains_no_go",
        "runtime_mapping_enabled_default_false",
        "enforcement_disabled_by_default",
        "artifact_free_validation_green" if artifact_free_green else "artifact_free_validation_pending_or_red",
        "artifact_dependent_validation_ready_for_operator_local_run" if artifact_dependent_ready else "artifact_dependent_validation_fail_closed_or_pending",
        "artifact_dependent_fail_closed_confirmed" if artifact_dependent_fail_closed else "artifact_dependent_fail_closed_not_observed_in_this_report",
        "operator_artifacts_and_vectors_must_remain_uncommitted",
    ]
    return {
        "version": "v3_round316_320_validation_delta_next_recommendation",
        "rounds": [316, 317, 318, 319, 320],
        "success": artifact_free_green and (artifact_dependent_ready or artifact_dependent_fail_closed),
        "status": "round316_320_validation_delta_ready" if artifact_free_green else "blocked_round316_320_artifact_free_validation_not_green",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "artifact_free_validation_status": artifact_free_report.get("status") if artifact_free_report else "not_run",
        "artifact_dependent_validation_status": artifact_dependent_report.get("status") if artifact_dependent_report else "not_run",
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "remaining_taxonomy": remaining_taxonomy,
        "next_recommendation": (
            "Run the guarded artifact-dependent command only in an operator-local workspace with the required ignored/untracked artifacts present; "
            "if it is green, record full-suite taxonomy while keeping production persistence, runtime mapping defaults, and enforcement disabled."
        ),
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Round306-320 split-validation execution guards.")
    parser.add_argument("--round291-305-json", required=True, help="Path to the green Round291-305 split-validation JSON.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--artifact-free", action="store_true", help="Run artifact-free compile/collect/focused validation and report artifact readiness separately.")
    mode.add_argument("--artifact-dependent", action="store_true", help="Check guarded artifact-dependent readiness; execute only with --execute-artifact-dependent.")
    parser.add_argument("--operator-authorized", action="store_true", help="Required for artifact-dependent mode.")
    parser.add_argument("--authorization-token", help="Required deterministic authorization token for artifact-dependent mode.")
    parser.add_argument("--execute-artifact-dependent", action="store_true", help="Actually run artifact-dependent commands after all fail-closed guards pass.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = _read_json(args.round291_305_json)
        if args.artifact_free:
            report = run_artifact_free_validation_phase(source)
        else:
            report = run_guarded_artifact_dependent_entrypoint(
                source,
                operator_authorized=args.operator_authorized,
                authorization_token=args.authorization_token,
                execute=args.execute_artifact_dependent,
            )
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(_compact_json(report))
        return int(report.get("exit_code", 1))
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        payload = {
            "version": ROUND306_320_VERSION,
            "status": "round306_320_command_failed_closed",
            "success": False,
            "exit_code": 1,
            "error": str(exc),
            **POLICY_FLAGS,
        }
        print(_compact_json(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
