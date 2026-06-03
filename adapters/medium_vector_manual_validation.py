"""Round103 manual medium-vector install validation workflow.

Round102 added the Release restore path, but this execution environment cannot
fetch GitHub Release assets.  Round103 therefore provides the single command an
operator can run after manually installing the verified, ignored medium
``vectors.npy`` artifact.  The workflow is fail-closed: when the artifact is not
present and manifest-valid, it records a blocked status and does not pretend to
run downstream validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from adapters.external_seed_manifest import FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH
from adapters.medium_vector_restoration import scan_subset_artifact_paths

ROUND103_MANUAL_VALIDATION_VERSION = "v3_round103_manual_medium_vector_validation"
TARGET_VECTORS_RELATIVE_PATH = f"{FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH}/vectors.npy"

ROUND103_VALIDATION_COMMANDS: tuple[str, ...] = (
    "python -m compileall -q adapters tests main.py",
    "pytest -q tests/test_v3_round100_medium_vector_restoration.py",
    "pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
    "pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py "
    "tests/test_v3_round93_runtime_mapping_proposal_report.py "
    "tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py "
    "tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py "
    "tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py "
    "tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
)

CommandRunner = Callable[[str, Path], dict[str, Any]]


def _default_command_runner(command: str, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S602 - internal fixed validation commands only
        command,
        cwd=cwd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    return {
        "command": command,
        "exit_code": completed.returncode,
        "result": "passed" if completed.returncode == 0 else "failed",
        "output_tail": output[-4000:],
    }


def _git_tracked(path: str, repo_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def artifact_boundary_audit(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return whether the local medium vector is present but still out of git."""
    root = Path(repo_root)
    target = root / TARGET_VECTORS_RELATIVE_PATH
    tracked = _git_tracked(TARGET_VECTORS_RELATIVE_PATH, root)
    return {
        "version": ROUND103_MANUAL_VALIDATION_VERSION,
        "target_vectors_path": str(target),
        "target_vectors_present": target.is_file(),
        "target_vectors_tracked_by_git": tracked,
        "target_vectors_allowed_in_pr_diff": False,
        "boundary_ok": not tracked,
    }


def build_round103_manual_validation_plan(repo_root: str | Path = ".") -> dict[str, Any]:
    """Build a read-only validation plan for a manually installed artifact."""
    root = Path(repo_root)
    scan = scan_subset_artifact_paths(root)
    boundary = artifact_boundary_audit(root)
    medium_available = scan["medium_validation_status"] == "available"
    boundary_ok = bool(boundary["boundary_ok"])
    ready = bool(medium_available and boundary_ok)
    return {
        "version": ROUND103_MANUAL_VALIDATION_VERSION,
        "round": 103,
        "read_only": True,
        "status": "ready_to_run_validation" if ready else "blocked_manual_artifact_not_ready",
        "hard_stop_released": ready,
        "policy": {
            "operator_manual_install_path": True,
            "download_release_assets_in_codex": False,
            "commit_vectors_npy_to_pr": False,
            "create_dummy_vectors": False,
            "weaken_tests": False,
            "agp_bypass": False,
        },
        "scan": scan,
        "artifact_boundary": boundary,
        "validation_commands": list(ROUND103_VALIDATION_COMMANDS),
        "manual_restore_command": (
            "python -m adapters.medium_vector_release_restore "
            "--work-dir /tmp/eve_round102_medium_restore "
            "--asset-dir /path/to/downloaded/release-assets "
            "--no-download --install-to-repo "
            "--output eve_v3_autonomous_handoff/validation/ROUND102_MEDIUM_VECTOR_ARTIFACT_RESTORE_STATUS.json"
        ),
        "single_validation_command": (
            "python -m adapters.medium_vector_manual_validation "
            "--output eve_v3_autonomous_handoff/validation/ROUND103_MANUAL_MEDIUM_VECTOR_VALIDATION_STATUS.json"
        ),
        "blocked_reason": None
        if ready
        else "medium vectors.npy is absent/invalid or tracked by git; install the verified ignored artifact before validation",
    }


def run_round103_manual_validation(
    repo_root: str | Path = ".",
    *,
    execute: bool = True,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run the Round103 validation commands if the manual artifact is ready."""
    root = Path(repo_root)
    plan = build_round103_manual_validation_plan(root)
    results: list[dict[str, Any]] = []
    if not plan["hard_stop_released"]:
        return {
            **plan,
            "read_only": False,
            "status": "blocked_manual_artifact_not_ready",
            "commands_executed": False,
            "command_results": results,
        }
    if not execute:
        return {
            **plan,
            "read_only": True,
            "status": "ready_to_run_validation",
            "commands_executed": False,
            "command_results": results,
        }

    runner = command_runner or _default_command_runner
    for command in ROUND103_VALIDATION_COMMANDS:
        result = runner(command, root)
        results.append(result)
        if result.get("exit_code") != 0:
            break
    passed = bool(results) and all(result.get("exit_code") == 0 for result in results)
    return {
        **plan,
        "read_only": False,
        "status": "validation_passed" if passed else "validation_failed",
        "commands_executed": True,
        "command_results": results,
    }


def write_round103_manual_validation_status(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    payload = deepcopy(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "version": ROUND103_MANUAL_VALIDATION_VERSION,
        "path": str(out),
        "json_written": True,
        "vectors_written": False,
        "binary_artifact_written_to_pr": False,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round103 manual medium-vector validation workflow")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument("--plan-only", action="store_true", help="write readiness plan without running validation commands")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    data = run_round103_manual_validation(repo_root=args.repo_root, execute=not args.plan_only)
    if args.output:
        write_round103_manual_validation_status(args.output, data)
    print(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2))
    if data["status"] in {"validation_passed", "ready_to_run_validation"}:
        return 0
    return 2


__all__ = [
    "ROUND103_MANUAL_VALIDATION_VERSION",
    "ROUND103_VALIDATION_COMMANDS",
    "TARGET_VECTORS_RELATIVE_PATH",
    "artifact_boundary_audit",
    "build_round103_manual_validation_plan",
    "run_round103_manual_validation",
    "write_round103_manual_validation_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
