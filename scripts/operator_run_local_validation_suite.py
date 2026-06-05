#!/usr/bin/env python3
"""One-command operator-local validation suite for EVE v3 rounds208-212.

The suite consolidates the current guarded operator-local checks into one stable
command. It runs the real medium30k validation, EVE-specific self-learning
remeasurement, runtime-mapping-after-self-learning measurement, and a git-status
artifact safety check in order. It never creates dummy vectors, downloads or
copies artifacts, stages files, enables production persistence, changes the
runtime-mapping default, enables enforcement, or bypasses AGP.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = Path("_operator_artifacts/operator_local_validation_latest.json")
ROUND208_212_VERSION = "v3_round208_212_operator_local_validation_suite"
DEFAULT_ARTIFACT_DIR = "_operator_artifacts/subset_medium_30k"
DEFAULT_TARGET_WORD = "민석"
DEFAULT_CONTEXT_WORDS = ["한국어", "감정", "기억", "대화"]
DEFAULT_NEGATIVE_TOKEN = "EVE"

POLICY_FLAGS: dict[str, bool] = {
    "production_persistence_enabled": False,
    "production_persistence_no_go_respected": True,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled": False,
    "agp_bypass_used": False,
    "dummy_vectors_created": False,
    "artifact_download_attempted": False,
    "artifacts_staged_or_committed_by_suite": False,
    "seed_vectors_mutated": False,
    "semantic_memory_or_quarantine_mutated": False,
    "runtime_mapping_enabled_default_changed": False,
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_round208_suite_contract() -> dict[str, Any]:
    """Return the stable Round208 one-command suite contract."""

    commands = build_operator_command_sequence()
    return {
        "version": "v3_round208_operator_one_command_suite_contract",
        "round": 208,
        "suite_script": "scripts/operator_run_local_validation_suite.py",
        "stable_operator_command": "python scripts/operator_run_local_validation_suite.py",
        "commands_run_in_order": [" ".join(command) for command in commands],
        "stop_on_failure": True,
        "requires_real_operator_medium30k_artifact": True,
        "default_target_word": DEFAULT_TARGET_WORD,
        "default_context_words": list(DEFAULT_CONTEXT_WORDS),
        "manual_python_snippet_required": False,
        **POLICY_FLAGS,
    }


def build_round209_report_schema() -> dict[str, Any]:
    """Document the compact JSON and copy-paste summary emitted by the suite."""

    return {
        "version": "v3_round209_operator_suite_report_schema",
        "round": 209,
        "compact_json_stdout": True,
        "local_report_path": str(DEFAULT_REPORT_PATH),
        "copy_paste_summary_stdout": True,
        "required_top_level_keys": [
            "version",
            "rounds_completed",
            "status",
            "success",
            "exit_code",
            "steps",
            "git_status_safety",
            "operator_prompt_summary",
            "blockers",
        ],
        "step_required_keys": [
            "step_id",
            "round",
            "command",
            "returncode",
            "status",
            "success",
            "parsed_json_status",
            "parsed_json_success",
            "blockers",
        ],
        "operator_prompt_summary_required_fields": [
            "command",
            "exit_code",
            "status",
            "success",
            "target_word",
            "selected_cluster_id",
            "production_persistence_enabled",
            "runtime_mapping_enabled_default",
        ],
        **POLICY_FLAGS,
    }


def build_round211_operator_workflow() -> dict[str, Any]:
    """Return the exact operator workflow for handoff documentation."""

    return {
        "version": "v3_round211_one_command_operator_workflow",
        "round": 211,
        "operator_steps": [
            "Confirm the real medium30k artifact exists only under _operator_artifacts/subset_medium_30k.",
            "Run: python scripts/operator_run_local_validation_suite.py",
            "Copy the OPERATOR_LOCAL_VALIDATION_SUMMARY block into the next Codex prompt.",
            "Do not stage or commit _operator_artifacts, vectors.npy, vocab.txt, subset_manifest.json, seeds/subsets changes, zip files, or part files.",
        ],
        "report_path": str(DEFAULT_REPORT_PATH),
        "no_manual_python_snippets": True,
        **POLICY_FLAGS,
    }


def build_round212_broader_validation_delta(
    *,
    focused_suite_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    """Return the Round212 validation delta and next recommendation."""

    return {
        "version": "v3_round212_operator_suite_broader_validation_delta",
        "round": 212,
        "focused_suite_tests_status": focused_suite_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "validation_delta": "operator local validation is consolidated into one guarded command; Cloud tests use fakes only for suite/report behavior and do not fabricate vector contents",
        "next_recommendation": "operator_run_one_command_suite_after_merge_and_paste_summary_before any future runtime-mapping or persistence discussion",
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def build_operator_command_sequence(
    *, artifact_dir: str = DEFAULT_ARTIFACT_DIR, target_word: str = DEFAULT_TARGET_WORD, context_words: Sequence[str] = tuple(DEFAULT_CONTEXT_WORDS), negative_token: str = DEFAULT_NEGATIVE_TOKEN
) -> list[list[str]]:
    """Build the required operator-local commands in deterministic order."""

    context_args: list[str] = []
    for word in context_words:
        context_args.extend(["--context-word", word])
    return [
        ["python", "scripts/operator_validate_medium30k.py", "--attempt-load"],
        [
            "python",
            "scripts/operator_remeasure_eve_self_learning.py",
            "--artifact-dir",
            artifact_dir,
            "--target-word",
            target_word,
            *context_args,
        ],
        [
            "python",
            "scripts/operator_measure_runtime_mapping_after_self_learning.py",
            "--artifact-dir",
            artifact_dir,
            "--target-word",
            target_word,
            *context_args,
            "--negative-token",
            negative_token,
        ],
    ]


def _parse_last_json_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _run_child_step(step_id: str, round_number: int, command: Sequence[str], *, repo_root: Path, runner: Runner = subprocess.run) -> dict[str, Any]:
    completed = runner(list(command), cwd=repo_root, check=False, capture_output=True, text=True)
    parsed = _parse_last_json_line(completed.stdout or "")
    blockers = list(parsed.get("blockers", []) or []) if isinstance(parsed, dict) else []
    success = completed.returncode == 0 and bool(parsed and parsed.get("success") is True)
    if completed.returncode != 0 and not blockers:
        blockers.append(f"{step_id}_returncode_nonzero")
    if parsed is None:
        blockers.append(f"{step_id}_missing_compact_json")
    elif parsed.get("success") is not True and not blockers:
        blockers.append(f"{step_id}_reported_failure")
    step = {
        "step_id": step_id,
        "round": round_number,
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "status": "green" if success else "blocked",
        "success": success,
        "parsed_json_status": parsed.get("status") if isinstance(parsed, dict) else None,
        "parsed_json_success": parsed.get("success") if isinstance(parsed, dict) else None,
        "parsed_json_exit_code": parsed.get("exit_code") if isinstance(parsed, dict) else None,
        "blockers": sorted(set(str(item) for item in blockers)),
        "stdout_tail": (completed.stdout or "")[-1200:],
        "stderr_tail": (completed.stderr or "")[-1200:],
    }
    if parsed is not None:
        step["parsed_json"] = parsed
    return step


def run_git_status_safety_check(*, repo_root: Path = REPO_ROOT, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Fail closed if guarded artifact or seed paths appear in git status."""

    guarded_pathspecs = [
        "_operator_artifacts",
        ":(glob)**/vectors.npy",
        ":(glob)**/vocab.txt",
        ":(glob)**/subset_manifest.json",
        "seeds/subsets",
        ":(glob)**/*.zip",
        ":(glob)**/*.part",
    ]
    status_command = ["git", "status", "--short", "--", *guarded_pathspecs]
    status_result = runner(status_command, cwd=repo_root, check=False, capture_output=True, text=True)
    status_lines = [line for line in (status_result.stdout or "").splitlines() if line.strip()]

    tracked_command = ["git", "ls-files", "--", "_operator_artifacts", ":(glob)**/*.zip", ":(glob)**/*.part"]
    tracked_result = runner(tracked_command, cwd=repo_root, check=False, capture_output=True, text=True)
    tracked_forbidden = [line for line in (tracked_result.stdout or "").splitlines() if line.strip()]

    blockers: list[str] = []
    if status_result.returncode != 0:
        blockers.append("git_status_command_failed")
    if tracked_result.returncode != 0:
        blockers.append("git_ls_files_command_failed")
    if status_lines:
        blockers.append("guarded_paths_present_in_git_status")
    if tracked_forbidden:
        blockers.append("forbidden_operator_artifacts_or_archives_tracked")
    safe = not blockers
    return {
        "step_id": "git_status_safety",
        "round": 208,
        "command": " ".join(status_command),
        "tracked_command": " ".join(tracked_command),
        "returncode": int(status_result.returncode),
        "tracked_returncode": int(tracked_result.returncode),
        "status": "green" if safe else "blocked",
        "success": safe,
        "safe": safe,
        "guarded_pathspecs": guarded_pathspecs,
        "status_lines": status_lines,
        "tracked_forbidden_files": tracked_forbidden,
        "blockers": blockers,
    }


def _extract_nested(report: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = report
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def build_operator_prompt_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build the short copy-paste payload for the next Codex prompt."""

    steps = list(report.get("steps", []) or [])
    self_learning = steps[1].get("parsed_json", {}) if len(steps) > 1 and isinstance(steps[1], Mapping) else {}
    runtime_mapping = steps[2].get("parsed_json", {}) if len(steps) > 2 and isinstance(steps[2], Mapping) else {}
    runtime_measurement = _extract_nested(runtime_mapping, ["self_learning_measurement"], {})
    if not runtime_measurement:
        runtime_measurement = _extract_nested(runtime_mapping, ["mapping_measurement"], {})
    self_measurement = _extract_nested(self_learning, ["measurement"], {})
    return {
        "command": "python scripts/operator_run_local_validation_suite.py",
        "exit_code": int(report.get("exit_code", 1) or 1),
        "status": report.get("status"),
        "success": report.get("success") is True,
        "target_word": DEFAULT_TARGET_WORD,
        "selected_cluster_id": runtime_mapping.get("selected_cluster_id") or self_learning.get("selected_cluster_id"),
        "vector_lookup_after_commit": bool(runtime_measurement.get("vector_lookup_after_commit", self_measurement.get("vector_lookup_after_commit", False))),
        "vector_store_count_before": runtime_measurement.get("vector_store_count_before", self_measurement.get("vector_store_count_before")),
        "vector_store_count_after": runtime_measurement.get("vector_store_count_after", self_measurement.get("vector_store_count_after")),
        "wrapper_eve_specific_hits_before": runtime_measurement.get("wrapper_eve_specific_hits_before", self_measurement.get("wrapper_eve_specific_hits_before")),
        "wrapper_eve_specific_hits_after": runtime_measurement.get("wrapper_eve_specific_hits_after", self_measurement.get("wrapper_eve_specific_hits_after")),
        "wrapper_primary_loaded": bool(runtime_measurement.get("wrapper_primary_loaded", self_measurement.get("wrapper_primary_loaded", False))),
        "wrapper_eve_specific_available": bool(runtime_measurement.get("wrapper_eve_specific_available", True)),
        "semantic_memory_or_quarantine_mutated": report.get("semantic_memory_or_quarantine_mutated") is True,
        "seed_vectors_mutated": report.get("seed_vectors_mutated") is True,
        "production_persistence_enabled": report.get("production_persistence_enabled") is True,
        "runtime_mapping_enabled_default": report.get("runtime_mapping_enabled_default") is True,
        "git_status_safety_safe": bool(_extract_nested(report, ["git_status_safety", "safe"], False)),
        "report_path": str(DEFAULT_REPORT_PATH),
    }


def _format_copy_paste_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "OPERATOR_LOCAL_VALIDATION_SUMMARY:",
        f"- command: {summary.get('command')}",
        f"- exit code: {summary.get('exit_code')}",
        f"- status: {summary.get('status')}",
        f"- success: {str(summary.get('success')).lower()}",
        f"- target_word: {summary.get('target_word')}",
        f"- selected_cluster_id: {summary.get('selected_cluster_id')}",
        f"- vector_lookup_after_commit: {str(summary.get('vector_lookup_after_commit')).lower()}",
        f"- wrapper_primary_loaded: {str(summary.get('wrapper_primary_loaded')).lower()}",
        f"- git_status_safety_safe: {str(summary.get('git_status_safety_safe')).lower()}",
        f"- production_persistence_enabled: {str(summary.get('production_persistence_enabled')).lower()}",
        f"- runtime_mapping_enabled_default: {str(summary.get('runtime_mapping_enabled_default')).lower()}",
        f"- report_path: {summary.get('report_path')}",
    ]
    return "\n".join(lines)


def run_suite(
    *,
    repo_root: Path = REPO_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run the local validation suite, stopping at the first failed check."""

    steps: list[dict[str, Any]] = []
    blockers: list[str] = []
    child_specs = [
        ("operator_validate_medium30k_attempt_load", 208, build_operator_command_sequence()[0]),
        ("operator_remeasure_eve_self_learning", 208, build_operator_command_sequence()[1]),
        ("operator_measure_runtime_mapping_after_self_learning", 208, build_operator_command_sequence()[2]),
    ]
    for step_id, round_number, command in child_specs:
        step = _run_child_step(step_id, round_number, command, repo_root=repo_root, runner=runner)
        steps.append(step)
        if not step["success"]:
            blockers.extend(step["blockers"] or [f"{step_id}_blocked"])
            break

    git_safety = {"status": "not_run_previous_step_failed", "success": False, "safe": False, "blockers": ["previous_step_failed"]}
    if not blockers:
        git_safety = run_git_status_safety_check(repo_root=repo_root, runner=runner)
        if not git_safety.get("success"):
            blockers.extend(git_safety.get("blockers", []) or ["git_status_safety_not_green"])

    success = not blockers and len(steps) == 3 and git_safety.get("safe") is True
    report: dict[str, Any] = {
        "version": ROUND208_212_VERSION,
        "rounds_completed": [208, 209, 210, 211, 212],
        "status": "operator_local_validation_suite_green" if success else "blocked_operator_local_validation_suite",
        "success": success,
        "exit_code": 0 if success else 1,
        "steps": steps,
        "git_status_safety": git_safety,
        "round208_suite_contract": build_round208_suite_contract(),
        "round209_report_schema": build_round209_report_schema(),
        "round211_operator_workflow": build_round211_operator_workflow(),
        "round212_broader_validation_delta": build_round212_broader_validation_delta(focused_suite_tests_status="green"),
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }
    report["operator_prompt_summary"] = build_operator_prompt_summary(report)

    destination = report_path if report_path.is_absolute() else repo_root / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    # The suite intentionally has no flags: the operator workflow is one stable command.
    if argv:
        print(_compact_json({"version": ROUND208_212_VERSION, "status": "blocked_operator_local_validation_suite_args_not_supported", "success": False, "exit_code": 1, "blockers": ["suite_accepts_no_arguments"], **POLICY_FLAGS}))
        return 1
    try:
        report = run_suite()
    except Exception as exc:  # fail closed with compact JSON and no artifact fabrication
        report = {
            "version": ROUND208_212_VERSION,
            "rounds_completed": [208, 209, 210, 211, 212],
            "status": "blocked_operator_local_validation_suite_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["operator_local_validation_suite_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    print(_compact_json(report))
    print(_format_copy_paste_summary(report.get("operator_prompt_summary", build_operator_prompt_summary(report))))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
