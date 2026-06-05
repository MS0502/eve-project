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
ROUND218_220_VERSION = "v3_round218_220_operator_suite_summary_exit_code_fix"
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



def build_round213_git_safety_diagnosis() -> dict[str, Any]:
    """Document the git-safety false-negative root cause diagnosed in Round213."""

    return {
        "version": "v3_round213_git_safety_diagnosis",
        "round": 213,
        "root_cause": "git safety reported false because latest main still contained a legacy tracked upload zip and the guard also lacked an explicit exact-file allowance for the suite-owned ignored local report",
        "affected_report_path": str(DEFAULT_REPORT_PATH),
        "operator_measurements_green_but_suite_blocked_possible": True,
        "legacy_tracked_archive_removed": "eve_v3_autonomous_handoff/packages/eve_v3_round96_code_only_no_medium_vectors.zip",
        **POLICY_FLAGS,
    }


def build_round216_final_operator_workflow() -> dict[str, Any]:
    """Return the final one-command operator workflow after git safety correction."""

    return {
        "version": "v3_round216_final_one_command_operator_workflow",
        "round": 216,
        "operator_steps": [
            "Keep the real medium30k artifact under _operator_artifacts/subset_medium_30k as ignored local data only.",
            "Run: python scripts/operator_run_local_validation_suite.py",
            "Copy the OPERATOR_LOCAL_VALIDATION_SUMMARY block only when the command exits 0.",
            "If the command exits nonzero, inspect the compact JSON blockers and do not paste a green summary.",
            "Never stage or commit _operator_artifacts, vectors.npy, vocab.txt, subset_manifest.json, seeds/subsets changes, zip files, part files, or upload zips.",
        ],
        "report_path": str(DEFAULT_REPORT_PATH),
        "copy_paste_summary_only_after_all_checks_pass": True,
        "no_manual_python_snippets": True,
        **POLICY_FLAGS,
    }


def build_round217_validation_delta_and_next_recommendation(
    *,
    focused_git_safety_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    """Return the broader validation delta and next recommendation for Round217."""

    return {
        "version": "v3_round217_git_safety_validation_delta",
        "round": 217,
        "focused_git_safety_tests_status": focused_git_safety_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "validation_delta": "suite git safety now distinguishes exact ignored local report output from staged/tracked or unignored artifact leakage",
        "next_recommendation": "operator_rerun_one_command_suite_from_latest_main_and_paste_green_summary_if_exit_code_zero",
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def build_round218_summary_exit_code_fix() -> dict[str, Any]:
    """Document the Round218 copy-paste summary exit-code repair."""

    return {
        "version": "v3_round218_summary_exit_code_fix",
        "round": 218,
        "bug_classification": "summary_reporting_bug_not_validation_failure",
        "root_cause": "operator_prompt_summary used a truthy fallback that converted a valid zero exit_code into one",
        "fix": "preserve explicit integer exit_code values, including zero, when building the summary",
        "green_suite_summary_must_print_exit_code": 0,
        "failed_suite_json_exit_code_must_be_nonzero": True,
        **POLICY_FLAGS,
    }


def build_round219_summary_exit_code_test_plan(
    *,
    focused_summary_tests_status: str = "pending",
    failed_json_summary_status: str = "pending",
) -> dict[str, Any]:
    """Return the Round219 focused test evidence plan for summary exit codes."""

    return {
        "version": "v3_round219_summary_exit_code_tests",
        "round": 219,
        "focused_summary_tests_status": focused_summary_tests_status,
        "failed_json_summary_status": failed_json_summary_status,
        "proves_green_summary_exit_code_zero": focused_summary_tests_status == "green",
        "proves_failed_suite_json_exit_code_nonzero": failed_json_summary_status == "green",
        "summary_emitted_only_after_success": True,
        **POLICY_FLAGS,
    }


def build_round220_next_cluster_selection(report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select the next narrow measurement cluster from green operator-local evidence.

    The selection is data-only: it interprets the now-green one-command suite
    evidence and deliberately keeps runtime mapping, enforcement, and production
    persistence disabled by default.
    """

    report = report or {}
    summary = report.get("operator_prompt_summary", {}) if isinstance(report.get("operator_prompt_summary"), Mapping) else {}
    suite_green = bool(report.get("success") is True and int(report.get("exit_code", 1)) == 0)
    green_evidence = {
        "operator_local_validation_suite_green": suite_green,
        "summary_exit_code": summary.get("exit_code"),
        "selected_cluster_id": summary.get("selected_cluster_id"),
        "target_word": summary.get("target_word", DEFAULT_TARGET_WORD),
        "vector_lookup_after_commit": bool(summary.get("vector_lookup_after_commit", False)),
        "wrapper_primary_loaded": bool(summary.get("wrapper_primary_loaded", False)),
        "git_status_safety_safe": bool(summary.get("git_status_safety_safe", False)),
        "production_persistence_enabled": bool(summary.get("production_persistence_enabled", True)),
        "runtime_mapping_enabled_default": bool(summary.get("runtime_mapping_enabled_default", True)),
    }
    ready_for_measurement = (
        suite_green
        and green_evidence["summary_exit_code"] == 0
        and green_evidence["vector_lookup_after_commit"] is True
        and green_evidence["wrapper_primary_loaded"] is True
        and green_evidence["git_status_safety_safe"] is True
        and green_evidence["production_persistence_enabled"] is False
        and green_evidence["runtime_mapping_enabled_default"] is False
    )
    return {
        "version": "v3_round220_operator_green_evidence_next_cluster_selection",
        "round": 220,
        "green_evidence": green_evidence,
        "ready_for_next_measurement_cluster": ready_for_measurement,
        "selected_next_cluster_id": "runtime_mapping_acceptance_delta_measurement_no_enablement" if ready_for_measurement else "rerun_operator_local_validation_until_summary_green",
        "selected_next_cluster_type": "measurement_only",
        "next_recommendation": "measure runtime-mapping acceptance deltas with runtime_mapping_enabled default false, enforcement disabled, AGP intact, and production persistence NO-GO",
        "hard_stops": [] if ready_for_measurement else ["operator_local_green_evidence_incomplete"],
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


def _status_line_path(line: str) -> str:
    """Extract the path component from a porcelain v1 status line."""

    payload = line[3:] if len(line) > 3 else ""
    if " -> " in payload:
        payload = payload.split(" -> ", 1)[1]
    return payload.strip().strip('"')


def run_git_status_safety_check(*, repo_root: Path = REPO_ROOT, runner: Runner = subprocess.run) -> dict[str, Any]:
    """Fail closed on artifact leakage while allowing the ignored local report.

    Round213 root cause: the previous guard treated every path under
    ``_operator_artifacts`` as equally forbidden.  That is correct for vector,
    archive, and subset payload leakage, but too broad for the suite's own local
    JSON report.  Round214 narrows the exception to exactly
    ``_operator_artifacts/operator_local_validation_latest.json`` and only when
    it is ignored local output, never staged or tracked.
    """

    allowed_report_path = str(DEFAULT_REPORT_PATH)
    guarded_pathspecs = [
        "_operator_artifacts",
        ":(glob)**/vectors.npy",
        ":(glob)**/vocab.txt",
        ":(glob)**/subset_manifest.json",
        "seeds/subsets",
        ":(glob)**/*.zip",
        ":(glob)**/*.part",
    ]
    status_command = ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *guarded_pathspecs]
    status_result = runner(status_command, cwd=repo_root, check=False, capture_output=True, text=True)
    status_lines = [line for line in (status_result.stdout or "").splitlines() if line.strip()]

    tracked_command = ["git", "ls-files", "--", "_operator_artifacts", ":(glob)**/*.zip", ":(glob)**/*.part"]
    tracked_result = runner(tracked_command, cwd=repo_root, check=False, capture_output=True, text=True)
    tracked_forbidden = [line for line in (tracked_result.stdout or "").splitlines() if line.strip()]

    report_tracked_command = ["git", "ls-files", "--", allowed_report_path]
    report_tracked_result = runner(report_tracked_command, cwd=repo_root, check=False, capture_output=True, text=True)
    report_tracked = [line for line in (report_tracked_result.stdout or "").splitlines() if line.strip()]

    report_path = repo_root / DEFAULT_REPORT_PATH
    report_exists = report_path.exists()
    report_ignore_command = ["git", "check-ignore", "--quiet", allowed_report_path]
    report_ignore_result = runner(report_ignore_command, cwd=repo_root, check=False, capture_output=True, text=True) if report_exists else None
    report_ignored = bool(report_exists and report_ignore_result and report_ignore_result.returncode == 0)

    blocked_status_lines: list[str] = []
    allowed_ignored_local_outputs: list[str] = []
    for line in status_lines:
        path = _status_line_path(line)
        index_status = line[:1]
        worktree_status = line[1:2]
        if path == allowed_report_path and index_status == "?" and worktree_status == "?" and report_ignored:
            allowed_ignored_local_outputs.append(path)
            continue
        blocked_status_lines.append(line)

    if report_exists and report_ignored and allowed_report_path not in allowed_ignored_local_outputs:
        allowed_ignored_local_outputs.append(allowed_report_path)

    blockers: list[str] = []
    if status_result.returncode != 0:
        blockers.append("git_status_command_failed")
    if tracked_result.returncode != 0:
        blockers.append("git_ls_files_command_failed")
    if report_tracked_result.returncode != 0:
        blockers.append("git_report_ls_files_command_failed")
    if report_ignore_result is not None and report_ignore_result.returncode not in (0, 1):
        blockers.append("git_report_check_ignore_command_failed")
    if blocked_status_lines:
        blockers.append("guarded_paths_present_in_git_status")
    if tracked_forbidden:
        blockers.append("forbidden_operator_artifacts_or_archives_tracked")
    if report_tracked:
        blockers.append("operator_local_report_tracked_or_staged")
    if report_exists and not report_ignored:
        blockers.append("operator_local_report_unignored")
    safe = not blockers
    return {
        "step_id": "git_status_safety",
        "round": 214,
        "command": " ".join(status_command),
        "tracked_command": " ".join(tracked_command),
        "report_tracked_command": " ".join(report_tracked_command),
        "report_ignore_command": " ".join(report_ignore_command),
        "returncode": int(status_result.returncode),
        "tracked_returncode": int(tracked_result.returncode),
        "report_tracked_returncode": int(report_tracked_result.returncode),
        "report_ignore_returncode": None if report_ignore_result is None else int(report_ignore_result.returncode),
        "status": "green" if safe else "blocked",
        "success": safe,
        "safe": safe,
        "guarded_pathspecs": guarded_pathspecs,
        "allowed_ignored_local_outputs": sorted(set(allowed_ignored_local_outputs)),
        "status_lines": status_lines,
        "blocked_status_lines": blocked_status_lines,
        "tracked_forbidden_files": tracked_forbidden,
        "operator_local_report_tracked_files": report_tracked,
        "operator_local_report_exists": report_exists,
        "operator_local_report_ignored": report_ignored,
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
        "exit_code": int(report.get("exit_code", 1)),
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
        "version": ROUND218_220_VERSION,
        "rounds_completed": [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220],
        "status": "operator_local_validation_suite_green" if success else "blocked_operator_local_validation_suite",
        "success": success,
        "exit_code": 0 if success else 1,
        "steps": steps,
        "git_status_safety": git_safety,
        "round208_suite_contract": build_round208_suite_contract(),
        "round209_report_schema": build_round209_report_schema(),
        "round211_operator_workflow": build_round211_operator_workflow(),
        "round212_broader_validation_delta": build_round212_broader_validation_delta(focused_suite_tests_status="green"),
        "round213_git_safety_diagnosis": build_round213_git_safety_diagnosis(),
        "round216_final_operator_workflow": build_round216_final_operator_workflow(),
        "round217_validation_delta": build_round217_validation_delta_and_next_recommendation(focused_git_safety_tests_status="green"),
        "round218_summary_exit_code_fix": build_round218_summary_exit_code_fix(),
        "round219_summary_exit_code_test_plan": build_round219_summary_exit_code_test_plan(
            focused_summary_tests_status="green",
            failed_json_summary_status="green",
        ),
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }
    report["operator_prompt_summary"] = build_operator_prompt_summary(report)
    report["round220_next_cluster_selection"] = build_round220_next_cluster_selection(report)

    destination = report_path if report_path.is_absolute() else repo_root / report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    # The suite intentionally has no flags: the operator workflow is one stable command.
    if argv:
        print(_compact_json({"version": ROUND218_220_VERSION, "status": "blocked_operator_local_validation_suite_args_not_supported", "success": False, "exit_code": 1, "blockers": ["suite_accepts_no_arguments"], **POLICY_FLAGS}))
        return 1
    try:
        report = run_suite()
    except Exception as exc:  # fail closed with compact JSON and no artifact fabrication
        report = {
            "version": ROUND218_220_VERSION,
            "rounds_completed": [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220],
            "status": "blocked_operator_local_validation_suite_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["operator_local_validation_suite_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    print(_compact_json(report))
    if report.get("success") is True:
        print(_format_copy_paste_summary(report.get("operator_prompt_summary", build_operator_prompt_summary(report))))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
