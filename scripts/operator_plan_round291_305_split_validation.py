#!/usr/bin/env python3
"""Rounds291-305 rollback evidence, split validation, and staged artifact planning.

This command consumes the green Round276-290 rollback/no-persistence audit packet
(or builds one from a supplied rehearsal JSON) and creates a deterministic,
read-only continuation packet for Rounds291-305.  It plans split full-suite
validation and a guarded artifact-staged rehearsal manifest, but it never loads
vectors, creates vectors, enables production persistence, changes the runtime
mapping default, enables enforcement, bypasses AGP, or stages operator artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_plan_round276_290_rehearsal_audit import (  # noqa: E402
    FOCUSED_TEST_COMMAND as ROUND276_290_FOCUSED_TEST_COMMAND,
    ROUND276_290_VERSION,
    build_round276_290_report,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import (  # noqa: E402
    AUTHORIZATION_TOKEN as ROUND261_275_AUTHORIZATION_TOKEN,
    FORBIDDEN_GIT_PATTERNS,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

ROUND291_305_VERSION = "v3_round291_305_split_validation_staged_artifact_manifest"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round291_305_split_validation.py"

COMPILE_COMMAND = "python -m compileall -q adapters tests main.py scripts"
COLLECT_ONLY_COMMAND = "pytest --collect-only -q"
FULL_SUITE_COMMAND = "python -m pytest -q"
GIT_STATUS_COMMAND = "git status --short"

DEFAULT_REQUIRED_LOCAL_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "artifact_id": "round261_275_no_persistence_rehearsal_json",
        "path": "_operator_artifacts/round261_275_no_persistence_runtime_mapping_rehearsal.json",
        "kind": "json",
        "required_for": "rollback_audit_replay",
    },
    {
        "artifact_id": "round236_260_acceptance_handoff_json",
        "path": "_operator_artifacts/round236_260_runtime_mapping_acceptance_handoff.json",
        "kind": "json",
        "required_for": "artifact_staged_rehearsal_replay",
    },
    {
        "artifact_id": "medium30k_vocab",
        "path": "_operator_artifacts/subset_medium_30k/vocab.txt",
        "kind": "operator_local_seed_subset_file",
        "required_for": "artifact_dependent_full_suite_segment",
    },
    {
        "artifact_id": "medium30k_vectors",
        "path": "_operator_artifacts/subset_medium_30k/vectors.npy",
        "kind": "operator_local_seed_subset_file",
        "required_for": "artifact_dependent_full_suite_segment",
    },
    {
        "artifact_id": "medium30k_subset_manifest",
        "path": "_operator_artifacts/subset_medium_30k/subset_manifest.json",
        "kind": "operator_local_seed_subset_file",
        "required_for": "artifact_dependent_full_suite_segment",
    },
)

VALIDATION_COMMANDS: dict[str, str] = {
    "compile": COMPILE_COMMAND,
    "collect_only": COLLECT_ONLY_COMMAND,
    "focused": FOCUSED_TEST_COMMAND,
    "round276_290_focused_regression": ROUND276_290_FOCUSED_TEST_COMMAND,
    "git_status_safety": GIT_STATUS_COMMAND,
    "full_suite": FULL_SUITE_COMMAND,
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _repo_relative(path: str | Path, *, repo_root: Path = REPO_ROOT) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def _git_lines(command: Sequence[str], *, repo_root: Path = REPO_ROOT) -> tuple[int, list[str]]:
    proc = subprocess.run(
        list(command),
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return proc.returncode, lines


def _path_is_git_ignored(relative_path: str, *, repo_root: Path = REPO_ROOT) -> bool:
    returncode, _ = _git_lines(["git", "check-ignore", "--quiet", relative_path], repo_root=repo_root)
    return returncode == 0


def _path_is_git_tracked(relative_path: str, *, repo_root: Path = REPO_ROOT) -> bool:
    returncode, _ = _git_lines(["git", "ls-files", "--error-unmatch", "--", relative_path], repo_root=repo_root)
    return returncode == 0


def _git_forbidden_artifact_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    returncode, lines = _git_lines(["git", "status", "--short"], repo_root=repo_root)
    forbidden = [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]
    return {
        "git_status_available": returncode == 0,
        "git_status_short": lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "clean_of_forbidden_operator_or_vector_artifacts": returncode == 0 and not forbidden,
    }


def consolidate_round291_rollback_audit_green_evidence(round276_290_report: Mapping[str, Any]) -> dict[str, Any]:
    """Round291: normalize the operator-local green rollback audit evidence."""

    source = deepcopy(dict(round276_290_report))
    audit = _as_mapping(source.get("round281_285_rehearsal_rollback_audit_proof"))
    audit_checks = _as_mapping(audit.get("audit_checks"))
    recommendation = _as_mapping(source.get("round286_290_validation_delta_next_recommendation"))
    checks = {
        "source_version_round276_290": source.get("version") == ROUND276_290_VERSION,
        "source_status_green": source.get("status") == "round276_290_rehearsal_evidence_rollback_audit_green",
        "source_success_true": source.get("success") is True,
        "source_exit_code_zero": source.get("exit_code") == 0,
        "round276_290_completed": source.get("rounds_completed") == list(range(276, 291)),
        "rollback_audit_success_true": audit.get("success") is True,
        "rollback_audit_score_full": _as_mapping(audit.get("audit_score")).get("rate") == 1.0,
        "runtime_mapping_not_persisted": audit_checks.get("runtime_mapping_not_persisted") is True,
        "embedding_lookup_not_called": audit_checks.get("embedding_lookup_not_called") is True,
        "eve_specific_vector_commit_not_called": audit_checks.get("eve_specific_vector_commit_not_called") is True,
        "blocked_control_not_mapped": audit_checks.get("blocked_control_not_mapped") is True,
        "production_persistence_no_go": source.get("production_persistence_enabled") is False,
        "runtime_mapping_default_false": source.get("runtime_mapping_enabled_default") is False,
        "runtime_mapping_default_unchanged": source.get("runtime_mapping_enabled_default_changed") is False,
        "enforcement_disabled": source.get("enforcement_enabled") is False,
        "seed_vectors_not_mutated": source.get("seed_vectors_mutated") is False,
        "semantic_memory_or_quarantine_not_mutated": source.get("semantic_memory_or_quarantine_mutated") is False,
        "next_recommendation_keeps_go_flags_false": recommendation.get("production_persistence_go") is False
        and recommendation.get("runtime_mapping_default_go") is False
        and recommendation.get("enforcement_go") is False,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": "v3_round291_rollback_audit_green_evidence_consolidation",
        "round": 291,
        "success": not blockers,
        "status": "round291_rollback_audit_green_evidence_consolidated" if not blockers else "blocked_round291_rollback_audit_evidence_not_green",
        "source_version": source.get("version"),
        "source_status": source.get("status"),
        "rollback_audit_status": audit.get("status"),
        "rollback_audit_score": audit.get("audit_score"),
        "checks": checks,
        "blockers": blockers,
        **POLICY_FLAGS,
    }


def build_split_full_suite_validation_planner(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Rounds292-295: split local validation into artifact-free and artifact-dependent phases."""

    success = evidence.get("success") is True
    phases = [
        {
            "round": 292,
            "phase": "artifact_free_compile_and_collection",
            "commands": [COMPILE_COMMAND, COLLECT_ONLY_COMMAND],
            "requires_operator_artifacts": False,
            "stop_on_failure": True,
        },
        {
            "round": 293,
            "phase": "focused_regression_segment",
            "commands": [FOCUSED_TEST_COMMAND, ROUND276_290_FOCUSED_TEST_COMMAND],
            "requires_operator_artifacts": False,
            "stop_on_failure": True,
        },
        {
            "round": 294,
            "phase": "artifact_dependent_local_segment",
            "commands": [
                "python scripts/operator_run_local_validation_suite.py",
                f"python scripts/operator_rehearse_runtime_mapping_no_persistence.py --handoff-json _operator_artifacts/round236_260_runtime_mapping_acceptance_handoff.json --operator-authorized --authorization-token {ROUND261_275_AUTHORIZATION_TOKEN} --output _operator_artifacts/round291_305_no_persistence_runtime_mapping_rehearsal.json",
                "python scripts/operator_plan_round291_305_split_validation.py --round276-290-json _operator_artifacts/round276_290_rehearsal_rollback_audit.json --output _operator_artifacts/round291_305_split_validation_manifest.json",
            ],
            "requires_operator_artifacts": True,
            "required_local_artifact_ids": [row["artifact_id"] for row in DEFAULT_REQUIRED_LOCAL_ARTIFACTS],
            "fail_closed_if_artifacts_missing": True,
            "stop_on_failure": True,
        },
        {
            "round": 295,
            "phase": "full_suite_and_git_safety_delta",
            "commands": [FULL_SUITE_COMMAND, GIT_STATUS_COMMAND],
            "requires_operator_artifacts": "only_for_artifact_dependent_tests",
            "git_status_must_prove_no_operator_or_vector_artifacts_staged": True,
            "stop_on_failure": False,
        },
    ]
    return {
        "version": "v3_round292_295_split_full_suite_validation_planner",
        "rounds": [292, 293, 294, 295],
        "success": success,
        "status": "round292_295_split_validation_planner_ready" if success else "blocked_round292_295_source_evidence_not_green",
        "source_evidence_version": evidence.get("version"),
        "validation_commands": VALIDATION_COMMANDS,
        "phases": phases,
        "execution_policy": {
            "split_artifact_free_from_artifact_dependent_tests": True,
            "do_not_skip_or_xfail_tests": True,
            "record_exact_failure_taxonomy_if_full_suite_red": True,
            "production_persistence_go": False,
            "runtime_mapping_default_go": False,
            "enforcement_go": False,
        },
        "blockers": [] if success else ["round291_evidence_not_green"],
        **POLICY_FLAGS,
    }


def build_guarded_artifact_staged_rehearsal_manifest(
    required_artifacts: Sequence[Mapping[str, Any]] | None = None,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Rounds296-300: record local artifact requirements and fail closed if absent.

    The manifest records paths only.  It does not read vector contents, compute
    vector checksums, copy files, create placeholder artifacts, or stage files.
    """

    root = Path(repo_root)
    rows = []
    for raw in required_artifacts or DEFAULT_REQUIRED_LOCAL_ARTIFACTS:
        artifact = dict(raw)
        relative_path = _repo_relative(str(artifact.get("path", "")), repo_root=root)
        path = root / relative_path
        local_only = not Path(relative_path).is_absolute() and relative_path.startswith("_operator_artifacts/")
        git_ignored = _path_is_git_ignored(relative_path, repo_root=root)
        git_tracked = _path_is_git_tracked(relative_path, repo_root=root)
        exists = path.exists()
        safe_to_reference = local_only and git_ignored and not git_tracked
        rows.append(
            {
                **artifact,
                "path": relative_path,
                "local_only": local_only,
                "git_ignored": git_ignored,
                "git_tracked": git_tracked,
                "exists": exists,
                "artifact_present": exists,
                "safe_to_reference": safe_to_reference,
                "artifact_safe_to_reference": safe_to_reference,
                "safe_to_execute": safe_to_reference and exists,
                "artifact_safe_to_load": False,
                "content_read": False,
                "vector_contents_read": False,
                "runtime_loaded": False,
                "checksum_computed": False,
                "placeholder_created": False,
            }
        )
    missing = sorted(row["artifact_id"] for row in rows if not row["exists"])
    unsafe = sorted(row["artifact_id"] for row in rows if not row["safe_to_reference"])
    git_guard = _git_forbidden_artifact_snapshot(root)
    staging_ready = not missing and not unsafe and git_guard.get("clean_of_forbidden_operator_or_vector_artifacts") is True
    return {
        "version": "v3_round296_300_guarded_artifact_staged_rehearsal_manifest",
        "rounds": [296, 297, 298, 299, 300],
        "success": not unsafe and git_guard.get("git_status_available") is True,
        "status": "round296_300_artifact_staged_manifest_ready" if staging_ready else "fail_closed_round296_300_artifact_staged_manifest_not_ready",
        "staging_ready": staging_ready,
        "fail_closed": not staging_ready,
        "required_artifacts": rows,
        "missing_artifact_ids": missing,
        "unsafe_artifact_ids": unsafe,
        "artifact_git_guard": git_guard,
        "manifest_behavior": {
            "records_required_local_artifacts_without_committing_them": True,
            "does_not_read_vector_contents": True,
            "does_not_create_or_fabricate_vectors": True,
            "does_not_stage_or_commit_artifacts": True,
            "requires_explicit_operator_local_paths": True,
            "fails_closed_when_required_artifacts_are_missing": True,
        },
        "blockers": [f"missing:{artifact_id}" for artifact_id in missing] + [f"unsafe:{artifact_id}" for artifact_id in unsafe],
        **POLICY_FLAGS,
    }


def build_round301_305_validation_delta_and_next_recommendation(
    evidence: Mapping[str, Any],
    planner: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Rounds301-305: focused-test plan, broader validation delta, and next step."""

    guard_success = evidence.get("success") is True and planner.get("success") is True and manifest.get("success") is True
    remaining_taxonomy = [
        "production_persistence_remains_no_go",
        "runtime_mapping_enabled_default_false",
        "enforcement_disabled_by_default",
        "artifact_staged_execution_blocked_until_all_required_local_artifacts_exist" if manifest.get("staging_ready") is not True else "artifact_staged_execution_ready_for_operator_local_run",
        "operator_artifacts_and_vectors_must_remain_uncommitted",
        "full_suite_result_must_be_recorded_with_exact_failure_taxonomy_if_red",
        "EVE_blocked_control_requires_concept_SA_AGP_evidence",
    ]
    return {
        "version": "v3_round301_305_validation_delta_next_recommendation",
        "rounds": [301, 302, 303, 304, 305],
        "success": guard_success,
        "status": "round301_305_validation_delta_ready" if guard_success else "blocked_round301_305_validation_delta",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "focused_test_scope": [
            "Round291 rollback audit green evidence consolidation",
            "Round292-295 split full-suite validation planner",
            "Round296-300 artifact-staged manifest fail-closed behavior",
        ],
        "broader_validation_delta": {
            "round291": "consolidates operator-local rollback/no-persistence audit green evidence from Round276-290",
            "round292_295": "adds a split validation planner that separates artifact-free checks from operator-local artifact-dependent checks",
            "round296_300": "adds a guarded artifact-staged manifest that records required local artifacts and fails closed when missing",
            "round301_305": "defines focused tests, broader validation requirements, remaining taxonomy, and next recommendation",
        },
        "remaining_taxonomy": remaining_taxonomy,
        "next_recommendation": (
            "run the split validation planner locally with real ignored _operator_artifacts present; if the manifest is staging_ready and focused/collect/compile/full-suite results are green, "
            "continue with another no-persistence rehearsal review, not production persistence enablement"
        ),
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "blockers": [] if guard_success else sorted(set(_as_list(evidence.get("blockers")) + _as_list(planner.get("blockers")) + _as_list(manifest.get("unsafe_artifact_ids")))),
        **POLICY_FLAGS,
    }


def build_round291_305_report(round276_290_report: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    evidence = consolidate_round291_rollback_audit_green_evidence(round276_290_report)
    planner = build_split_full_suite_validation_planner(evidence)
    manifest = build_guarded_artifact_staged_rehearsal_manifest(repo_root=repo_root)
    recommendation = build_round301_305_validation_delta_and_next_recommendation(evidence, planner, manifest)
    success = evidence.get("success") is True and planner.get("success") is True and manifest.get("success") is True and recommendation.get("success") is True
    return {
        "version": ROUND291_305_VERSION,
        "rounds_completed": list(range(291, 306)),
        "status": "round291_305_split_validation_staged_manifest_ready" if success else "blocked_round291_305_split_validation_staged_manifest",
        "success": success,
        "exit_code": 0 if success else 1,
        "round291_rollback_audit_green_evidence": evidence,
        "round292_295_split_full_suite_validation_planner": planner,
        "round296_300_guarded_artifact_staged_rehearsal_manifest": manifest,
        "round301_305_validation_delta_next_recommendation": recommendation,
        "rollback_audit_green_evidence": evidence.get("success") is True,
        "split_validation_planner": planner,
        "artifact_staged_manifest_behavior": manifest.get("manifest_behavior"),
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "validation_commands": VALIDATION_COMMANDS,
        "broader_validation_delta": recommendation.get("broader_validation_delta"),
        "remaining_taxonomy": recommendation.get("remaining_taxonomy"),
        "next_recommendation": recommendation.get("next_recommendation"),
        "blockers": [] if success else sorted(set(_as_list(evidence.get("blockers")) + _as_list(planner.get("blockers")) + _as_list(manifest.get("blockers")) + _as_list(recommendation.get("blockers")))),
        **POLICY_FLAGS,
    }


def write_round291_305_report(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("version", ROUND291_305_VERSION),
        "path": str(out),
        "json_only": True,
        "runtime_mapping_persisted": False,
        "production_persistence_enabled": False,
        "vectors_written": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Round291-305 split validation and staged artifact manifest packet.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--round276-290-json", help="Path to the operator-local Round276-290 audit JSON.")
    source.add_argument("--rehearsal-json", help="Path to the operator-local Round261-275 rehearsal JSON; a Round276-290 report is built in memory first.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.round276_290_json:
            round276_290_report = _read_json(args.round276_290_json)
        else:
            round276_290_report = build_round276_290_report(_read_json(args.rehearsal_json))
        report = build_round291_305_report(round276_290_report)
    except Exception as exc:  # fail closed with machine-readable output
        report = {
            "version": ROUND291_305_VERSION,
            "rounds_completed": list(range(291, 306)),
            "status": "blocked_round291_305_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["round291_305_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    if args.output:
        write_round291_305_report(args.output, report)
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
