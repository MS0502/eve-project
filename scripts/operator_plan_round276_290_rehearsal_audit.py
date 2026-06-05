#!/usr/bin/env python3
"""Rounds276-290 rehearsal evidence consolidation and rollback/audit planning.

This command consumes the operator-local Round261-275 no-persistence runtime
mapping rehearsal JSON and builds a deterministic, read-only continuation packet
for Rounds276-290.  It does not re-enter the rehearsal scope, build the
production engine, load vectors, enable production persistence, enable runtime
mapping by default, enable enforcement, bypass AGP, or stage artifacts.
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

from scripts.operator_rehearse_runtime_mapping_no_persistence import (  # noqa: E402
    AUTHORIZATION_TOKEN as ROUND261_275_AUTHORIZATION_TOKEN,
    FOCUSED_TEST_COMMAND as ROUND261_275_FOCUSED_TEST_COMMAND,
    FORBIDDEN_GIT_PATTERNS,
    ROUND261_275_VERSION,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

ROUND276_290_VERSION = "v3_round276_290_rehearsal_evidence_rollback_audit"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round276_290_rehearsal_audit.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "focused": FOCUSED_TEST_COMMAND,
    "round261_275_focused_regression": ROUND261_275_FOCUSED_TEST_COMMAND,
    "full_suite": "python -m pytest -q",
}
EXPECTED_ACCEPTED_TOKENS = ["민석"]
EXPECTED_BLOCKED_CONTROL_TOKENS = ["EVE"]
STAGED_REHEARSAL_OUTPUT = "_operator_artifacts/round276_290_no_persistence_runtime_mapping_rehearsal.json"
STAGED_AUDIT_OUTPUT = "_operator_artifacts/round276_290_rehearsal_rollback_audit.json"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _all_bool_values_true(mapping: Mapping[str, Any]) -> bool:
    return bool(mapping) and all(value is True for value in mapping.values() if isinstance(value, bool))


def _git_forbidden_artifact_snapshot(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    forbidden = [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]
    return {
        "git_status_available": proc.returncode == 0,
        "git_status_short": lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "clean_of_forbidden_operator_or_vector_artifacts": proc.returncode == 0 and not forbidden,
    }


def consolidate_no_persistence_rehearsal_green_evidence(rehearsal_report: Mapping[str, Any]) -> dict[str, Any]:
    """Round276: normalize the operator-local green rehearsal evidence."""
    source = deepcopy(dict(rehearsal_report))
    source_handoff = _as_mapping(source.get("source_handoff"))
    smoke = _as_mapping(source.get("round268_271_rehearsal_smoke"))
    no_persistence = _as_mapping(source.get("no_persistence_proof"))
    no_vectors = _as_mapping(source.get("no_vector_mutation_proof"))
    rollback = _as_mapping(source.get("rollback_proof"))
    artifact_guard = _as_mapping(source.get("artifact_git_guard"))
    fixture_preservation = _as_mapping(source.get("korean_fixture_preservation"))

    accepted_tokens = [str(token) for token in _as_list(source_handoff.get("accepted_tokens"))]
    blocked_tokens = [str(token) for token in _as_list(source_handoff.get("blocked_control_tokens"))]
    checks = {
        "source_version_round261_275": source.get("version") == ROUND261_275_VERSION,
        "source_status_green": source.get("status") == "no_persistence_runtime_mapping_rehearsal_green",
        "source_success_true": source.get("success") is True,
        "source_exit_code_zero": source.get("exit_code") == 0,
        "rehearsal_executed_true": source.get("rehearsal_executed") is True,
        "accepted_tokens_match": accepted_tokens == EXPECTED_ACCEPTED_TOKENS,
        "blocked_control_tokens_match": blocked_tokens == EXPECTED_BLOCKED_CONTROL_TOKENS,
        "runtime_mapping_enabled_default_false": source.get("runtime_mapping_enabled_default") is False,
        "production_persistence_disabled": source.get("production_persistence_enabled") is False,
        "enforcement_disabled": source.get("enforcement_enabled") is False,
        "runtime_mapping_enabled_after_rehearsal_false": smoke.get("runtime_mapping_enabled_after_rollback") is False,
        "enforcement_enabled_after_rehearsal_false": smoke.get("enforcement_enabled_after_rollback") is False,
        "no_persistence_proof_passed": no_persistence.get("passed") is True,
        "no_vector_mutation_proof_passed": no_vectors.get("passed") is True,
        "rollback_proof_verified": rollback.get("rollback_verified") is True,
        "korean_fixtures_preserved": fixture_preservation.get("preserved") is True,
        "minsok_preserved_exact": fixture_preservation.get("contains_minsok_exact") is True,
        "blocked_control_not_mapped": "EVE" not in [str(token) for token in _as_list(smoke.get("mapped_tokens"))],
        "artifact_git_guard_clean": artifact_guard.get("clean_of_forbidden_operator_or_vector_artifacts") is True,
        "agp_bypass_not_used": source.get("agp_bypass_used") is False,
        "semantic_memory_or_quarantine_not_mutated": source.get("semantic_memory_or_quarantine_mutated") is False,
        "seed_vectors_not_mutated": source.get("seed_vectors_mutated") is False,
    }
    blockers = sorted(key for key, passed in checks.items() if not passed)
    return {
        "version": "v3_round276_no_persistence_rehearsal_green_evidence",
        "round": 276,
        "success": not blockers,
        "status": "round276_no_persistence_rehearsal_green_evidence_consolidated" if not blockers else "blocked_round276_rehearsal_evidence_not_green",
        "source_version": source.get("version"),
        "source_status": source.get("status"),
        "accepted_tokens": accepted_tokens,
        "blocked_control_tokens": blocked_tokens,
        "operator_local_rehearsal_status": source.get("status"),
        "operator_local_rehearsal_success": source.get("success") is True,
        "checks": checks,
        "blockers": blockers,
        **POLICY_FLAGS,
    }


def design_next_isolated_rehearsal_steps(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Rounds277-280: split validation, artifact-staged rehearsal, rollback audit."""
    success = evidence.get("success") is True
    phases = [
        {
            "round": 277,
            "phase": "split_full_suite_validation",
            "purpose": "prove compile, collection, focused regression, and full pytest delta independently before any persistence decision",
            "commands": [
                VALIDATION_COMMANDS["compile"],
                VALIDATION_COMMANDS["collect_only"],
                VALIDATION_COMMANDS["focused"],
                VALIDATION_COMMANDS["round261_275_focused_regression"],
                VALIDATION_COMMANDS["full_suite"],
            ],
            "mutates_runtime_mapping": False,
            "mutates_persistence": False,
        },
        {
            "round": 278,
            "phase": "artifact_staged_rehearsal",
            "purpose": "allow operator to write a JSON-only rehearsal artifact under ignored _operator_artifacts for review",
            "operator_command": (
                "python scripts/operator_rehearse_runtime_mapping_no_persistence.py "
                "--handoff-json ./ROUND_V3_R236_R260_VALIDATION.json "
                f"--operator-authorized --authorization-token {ROUND261_275_AUTHORIZATION_TOKEN} "
                f"--output {STAGED_REHEARSAL_OUTPUT}"
            ),
            "allowed_output_path": STAGED_REHEARSAL_OUTPUT,
            "json_only": True,
            "must_not_be_committed": True,
        },
        {
            "round": 279,
            "phase": "rollback_audit",
            "purpose": "derive an explicit rollback/no-persistence audit snapshot from the staged rehearsal output",
            "operator_command": (
                "python scripts/operator_plan_round276_290_rehearsal_audit.py "
                f"--rehearsal-json {STAGED_REHEARSAL_OUTPUT} --output {STAGED_AUDIT_OUTPUT}"
            ),
            "allowed_output_path": STAGED_AUDIT_OUTPUT,
            "json_only": True,
            "must_not_be_committed": True,
        },
        {
            "round": 280,
            "phase": "go_no_go_review",
            "purpose": "recommend next isolated rehearsal only if evidence, validation, git artifact guard, and rollback audit are green",
            "production_persistence_go": False,
            "runtime_mapping_default_go": False,
            "enforcement_go": False,
        },
    ]
    return {
        "version": "v3_round277_280_next_isolated_rehearsal_step_design",
        "rounds": [277, 278, 279, 280],
        "success": success,
        "status": "round277_280_next_rehearsal_steps_designed" if success else "blocked_round277_280_source_evidence_not_green",
        "source_evidence_version": evidence.get("version"),
        "phases": phases,
        "hard_boundaries": {
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled": False,
            "agp_bypass_used": False,
            "operator_artifacts_must_not_be_committed": True,
        },
        "blockers": [] if success else ["round276_evidence_not_green"],
        **POLICY_FLAGS,
    }


def build_rehearsal_rollback_audit(rehearsal_report: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds281-285: concrete guarded observability improvement.

    The audit is read-only and deterministic: it turns the nested rehearsal
    rollback/no-persistence proofs into an operator-facing checklist with a
    measurable score and explicit failed-check taxonomy.
    """
    source = deepcopy(dict(rehearsal_report))
    smoke = _as_mapping(source.get("round268_271_rehearsal_smoke"))
    safety = _as_mapping(source.get("round272_274_safety_proofs"))
    rollback_checks = dict(_as_mapping(_as_mapping(source.get("rollback_proof")).get("checks")))
    rollback_checks.update(dict(_as_mapping(safety.get("rollback_checks"))))
    no_persistence_checks = dict(_as_mapping(_as_mapping(source.get("no_persistence_proof")).get("checks")))
    no_persistence_checks.update(dict(_as_mapping(safety.get("no_persistence_checks"))))
    no_vector_checks = dict(_as_mapping(_as_mapping(source.get("no_vector_mutation_proof")).get("checks")))
    no_vector_checks.update(dict(_as_mapping(safety.get("no_vector_mutation_checks"))))
    mutation_checks = dict(_as_mapping(smoke.get("mutation_checks")))
    artifact_guard = _git_forbidden_artifact_snapshot(Path(repo_root))

    audit_checks = {
        "source_rehearsal_green": source.get("success") is True,
        "rollback_checks_all_true": _all_bool_values_true(rollback_checks),
        "no_persistence_checks_all_true": _all_bool_values_true(no_persistence_checks),
        "no_vector_checks_all_true": _all_bool_values_true(no_vector_checks),
        "runtime_mapping_not_enabled_after_rollback": smoke.get("runtime_mapping_enabled_after_rollback") is False,
        "enforcement_not_enabled_after_rollback": smoke.get("enforcement_enabled_after_rollback") is False,
        "runtime_mapping_not_persisted": mutation_checks.get("runtime_mapping_persisted") is False,
        "embedding_lookup_not_called": mutation_checks.get("embedding_lookup_called_during_enable_smoke") is False,
        "eve_specific_vector_commit_not_called": mutation_checks.get("eve_specific_vector_commit_called") is False,
        "blocked_control_not_mapped": "EVE" not in [str(token) for token in _as_list(smoke.get("mapped_tokens"))],
        "git_status_clean_of_forbidden_artifacts": artifact_guard.get("clean_of_forbidden_operator_or_vector_artifacts") is True,
        "policy_flags_remain_safe": all(
            source.get(key) == expected for key, expected in POLICY_FLAGS.items()
        ),
    }
    failed = sorted(key for key, passed in audit_checks.items() if not passed)
    passed_count = sum(1 for passed in audit_checks.values() if passed)
    total_count = len(audit_checks)
    return {
        "version": "v3_round281_285_rehearsal_rollback_audit_proof",
        "rounds": [281, 282, 283, 284, 285],
        "success": not failed,
        "status": "round281_285_rehearsal_rollback_audit_green" if not failed else "blocked_round281_285_rehearsal_rollback_audit",
        "source_version": source.get("version"),
        "source_status": source.get("status"),
        "audit_score": {"passed": passed_count, "total": total_count, "rate": passed_count / total_count if total_count else 0.0},
        "audit_checks": audit_checks,
        "failed_checks": failed,
        "rollback_checks": rollback_checks,
        "no_persistence_checks": no_persistence_checks,
        "no_vector_mutation_checks": no_vector_checks,
        "mutation_checks": mutation_checks,
        "artifact_git_guard": artifact_guard,
        "measurable_behavior_improvement": "adds deterministic rollback/no-persistence audit scoring over operator-local rehearsal evidence without enabling production persistence",
        "mutations_performed": [],
        **POLICY_FLAGS,
    }


def build_validation_delta_and_recommendation(evidence: Mapping[str, Any], design: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Rounds286-290: validation delta, taxonomy, and next recommendation."""
    guards_green = evidence.get("success") is True and design.get("success") is True and audit.get("success") is True
    remaining_taxonomy = [
        "production_persistence_no_go",
        "runtime_mapping_enabled_default_false",
        "enforcement_disabled_by_default",
        "EVE_blocked_control_requires_concept_SA_AGP_evidence",
        "operator_artifacts_must_remain_uncommitted",
        "full_suite_validation_required_before_any_persistence_decision",
    ]
    return {
        "version": "v3_round286_290_validation_delta_next_recommendation",
        "rounds": [286, 287, 288, 289, 290],
        "success": guards_green,
        "status": "round286_290_validation_delta_ready" if guards_green else "blocked_round286_290_validation_delta",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "validation_commands": VALIDATION_COMMANDS,
        "broader_validation_delta": {
            "round276": "consolidated operator-local no-persistence rehearsal green evidence",
            "round277_280": "split full-suite / artifact-staged rehearsal / rollback audit design without production persistence",
            "round281_285": "implemented deterministic rollback/no-persistence audit scoring",
            "round286_290": "defined focused validation delta and no-go taxonomy for the next operator step",
        },
        "remaining_taxonomy": remaining_taxonomy,
        "next_recommendation": (
            "run split validation and the JSON-only staged rehearsal/audit locally; keep production persistence NO-GO, "
            "runtime_mapping_enabled default false, and enforcement disabled until a separate explicit persistence decision round"
        ),
        "production_persistence_go": False,
        "runtime_mapping_default_go": False,
        "enforcement_go": False,
        "blockers": [] if guards_green else ["round276_285_guard_not_green"],
        **POLICY_FLAGS,
    }


def build_round276_290_report(rehearsal_report: Mapping[str, Any], *, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    evidence = consolidate_no_persistence_rehearsal_green_evidence(rehearsal_report)
    design = design_next_isolated_rehearsal_steps(evidence)
    audit = build_rehearsal_rollback_audit(rehearsal_report, repo_root=repo_root)
    recommendation = build_validation_delta_and_recommendation(evidence, design, audit)
    success = evidence.get("success") is True and design.get("success") is True and audit.get("success") is True and recommendation.get("success") is True
    return {
        "version": ROUND276_290_VERSION,
        "rounds_completed": list(range(276, 291)),
        "status": "round276_290_rehearsal_evidence_rollback_audit_green" if success else "blocked_round276_290_rehearsal_evidence_rollback_audit",
        "success": success,
        "exit_code": 0 if success else 1,
        "round276_no_persistence_rehearsal_green_evidence": evidence,
        "round277_280_next_isolated_rehearsal_step_design": design,
        "round281_285_rehearsal_rollback_audit_proof": audit,
        "round286_290_validation_delta_next_recommendation": recommendation,
        "no_persistence_rehearsal_green_evidence": evidence.get("success") is True,
        "rollback_no_persistence_proof": audit,
        "concrete_implementation": "scripts/operator_plan_round276_290_rehearsal_audit.py deterministic audit/proposal surface",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "validation_commands": VALIDATION_COMMANDS,
        "remaining_taxonomy": recommendation.get("remaining_taxonomy", []),
        "next_recommendation": recommendation.get("next_recommendation"),
        "blockers": [] if success else sorted(set(_as_list(evidence.get("blockers")) + _as_list(audit.get("failed_checks")) + _as_list(recommendation.get("blockers")))),
        **POLICY_FLAGS,
    }


def write_round276_290_report(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("version", ROUND276_290_VERSION),
        "path": str(out),
        "json_only": True,
        "runtime_mapping_persisted": False,
        "production_persistence_enabled": False,
        "vectors_written": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Round276-290 rehearsal evidence and rollback audit packet.")
    parser.add_argument("--rehearsal-json", required=True, help="Path to the operator-local Round261-275 rehearsal JSON.")
    parser.add_argument("--output", help="Optional JSON output path, normally under ignored _operator_artifacts/.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rehearsal_report = _read_json(args.rehearsal_json)
        report = build_round276_290_report(rehearsal_report)
    except Exception as exc:  # fail closed with machine-readable output
        report = {
            "version": ROUND276_290_VERSION,
            "rounds_completed": list(range(276, 291)),
            "status": "blocked_round276_290_rehearsal_audit_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["round276_290_rehearsal_audit_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    if args.output:
        write_round276_290_report(args.output, report)
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":  # pragma: no cover - operator command entrypoint
    raise SystemExit(main())
