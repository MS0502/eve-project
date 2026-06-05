#!/usr/bin/env python3
"""Round231-235 runtime-mapping acceptance repair selection report.

This helper continues from the green Round226-230 acceptance-taxonomy report and
selects one narrow, non-persistence reporting improvement: an accepted/blocked
stage matrix for the existing runtime-mapping acceptance rows.  It reads the
ignored operator-local suite report, derives existing acceptance/taxonomy data,
and emits deterministic JSON only.  It never builds an engine, loads medium30k,
enables runtime mapping by default, enables enforcement, calls AGP, fabricates
vectors, or mutates vectors, semantic memory, quarantine, or persistence state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_report_runtime_mapping_acceptance_taxonomy import (  # noqa: E402
    ROUND226_230_VERSION,
    SELECTED_REPORTING_GAP,
    build_round226_230_report,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import DEFAULT_REPORT_PATH, POLICY_FLAGS  # noqa: E402

ROUND231_235_VERSION = "v3_round231_235_runtime_mapping_acceptance_repair_selection"
SELECTED_NEXT_REPAIR_TARGET = "runtime_mapping_accepted_blocked_stage_matrix_reporting_no_enablement"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round231_235_runtime_mapping_acceptance_repair_selection.py"
_STAGE_SEQUENCE = [
    "would_map_at_gate",
    "blocked_at_gate",
    "ready_at_precheck",
    "mapped_in_controlled_smoke",
    "rollback_complete",
    "runtime_mapping_enabled_after_false",
    "enforcement_enabled_after_false",
    "runtime_mapping_not_persisted",
]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _policy_guard_snapshot(*reports: Mapping[str, Any]) -> dict[str, bool]:
    merged: dict[str, Any] = {}
    for report in reports:
        merged.update({key: report.get(key) for key in POLICY_FLAGS})
    return {
        "production_persistence_disabled": merged.get("production_persistence_enabled") is False,
        "production_persistence_no_go_respected": merged.get("production_persistence_no_go_respected") is True,
        "runtime_mapping_enabled_default_false": merged.get("runtime_mapping_enabled_default") is False,
        "runtime_mapping_enabled_default_unchanged": merged.get("runtime_mapping_enabled_default_changed") is False,
        "enforcement_disabled": merged.get("enforcement_enabled") is False,
        "agp_bypass_not_used": merged.get("agp_bypass_used") is False,
        "dummy_vectors_not_created": merged.get("dummy_vectors_created") is False,
        "seed_vectors_not_mutated": merged.get("seed_vectors_mutated") is False,
        "semantic_memory_or_quarantine_not_mutated": merged.get("semantic_memory_or_quarantine_mutated") is False,
    }


def _delta_rows_from_taxonomy(taxonomy_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    acceptance_delta = _as_mapping(taxonomy_report.get("round221_225_acceptance_delta"))
    round223 = _as_mapping(acceptance_delta.get("round223_guarded_acceptance_delta_path"))
    return [_as_mapping(row) for row in _as_list(round223.get("delta_rows"))]


def _classify_row(row: Mapping[str, Any]) -> str:
    if row.get("accepted_for_future_operator_review") is True:
        return "accepted_for_future_operator_review"
    if row.get("blocked_control") is True:
        return "blocked_control"
    return "unresolved_failure"


def _stage_matrix_row(row: Mapping[str, Any]) -> dict[str, Any]:
    stages = _as_mapping(row.get("stages"))
    classification = _classify_row(row)
    first_failed_stage = row.get("first_failed_stage")
    if classification == "blocked_control":
        stop_stage = "blocked_at_gate" if stages.get("blocked_at_gate") is True else None
    elif classification == "accepted_for_future_operator_review":
        stop_stage = None
    else:
        stop_stage = str(first_failed_stage) if first_failed_stage else None
    return {
        "token": str(row.get("token")),
        "classification": classification,
        "first_failed_stage": None if first_failed_stage is None else str(first_failed_stage),
        "operator_stop_stage": stop_stage,
        "stage_outcomes": [
            {"stage": stage, "passed": stages.get(stage) is True}
            for stage in _STAGE_SEQUENCE
            if stage in stages
        ],
    }


def _stage_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_stage_matrix_row(row) for row in rows]


def build_round231_acceptance_taxonomy_green_evidence(taxonomy_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Consolidate Round226-230 taxonomy-green evidence before selecting a target."""

    taxonomy_report = _as_mapping(taxonomy_report)
    evidence = _as_mapping(taxonomy_report.get("acceptance_delta_green_evidence"))
    guards = _policy_guard_snapshot(taxonomy_report)
    success = bool(
        taxonomy_report.get("success") is True
        and taxonomy_report.get("status") == "runtime_mapping_acceptance_taxonomy_ready"
        and evidence.get("source_success") is True
        and taxonomy_report.get("selected_reporting_gap") == SELECTED_REPORTING_GAP
        and all(guards.values())
    )
    return {
        "version": "v3_round231_acceptance_taxonomy_green_evidence_consolidation",
        "round": 231,
        "source_version": taxonomy_report.get("version", ROUND226_230_VERSION),
        "taxonomy_status": taxonomy_report.get("status"),
        "taxonomy_success": taxonomy_report.get("success") is True,
        "taxonomy_green_evidence": evidence,
        "selected_reporting_gap": taxonomy_report.get("selected_reporting_gap"),
        "policy_guard_snapshot": guards,
        "hard_stops": [] if success else ["acceptance_taxonomy_green_evidence_incomplete_or_policy_guard_failed"],
        "success": success,
        **POLICY_FLAGS,
    }


def build_round232_next_repair_target_selection(
    taxonomy_report: Mapping[str, Any] | None = None,
    round231: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the next concrete non-persistence target from the taxonomy."""

    taxonomy_report = _as_mapping(taxonomy_report)
    round231 = _as_mapping(round231)
    ready = round231.get("success") is True
    return {
        "version": "v3_round232_next_non_persistence_repair_target_selection",
        "round": 232,
        "source_reporting_gap": taxonomy_report.get("selected_reporting_gap"),
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET if ready else "blocked_until_round231_taxonomy_green",
        "target_definition": (
            "Expose an operator-ready accepted/blocked stage matrix from existing acceptance-delta rows so the accepted token path, "
            "blocked-control stop point, and any unresolved first-failed stages are visible without enablement."
        ),
        "implementation_scope": "reporting-only matrix over existing runtime-mapping acceptance rows",
        "explicit_non_goals": [
            "production_persistence_enablement",
            "runtime_mapping_enabled_default_true",
            "enforcement_enablement",
            "AGP_bypass",
            "new_vector_creation_or_fabrication",
            "semantic_memory_or_quarantine_mutation",
        ],
        "hard_stops": [] if ready else ["round231_not_ready"],
        "success": ready,
        **POLICY_FLAGS,
    }


def build_round233_stage_matrix_reporting_improvement(
    taxonomy_report: Mapping[str, Any] | None = None,
    round232: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Implement the selected reporting improvement as deterministic data."""

    taxonomy_report = _as_mapping(taxonomy_report)
    round232 = _as_mapping(round232)
    rows = _delta_rows_from_taxonomy(taxonomy_report)
    matrix = _stage_matrix(rows)
    accepted = [row["token"] for row in matrix if row["classification"] == "accepted_for_future_operator_review"]
    blocked = [row["token"] for row in matrix if row["classification"] == "blocked_control"]
    unresolved = [row for row in matrix if row["classification"] == "unresolved_failure"]
    guards = _policy_guard_snapshot(taxonomy_report, round232)
    success = bool(round232.get("success") is True and matrix and accepted and blocked and all(guards.values()))
    return {
        "version": "v3_round233_accepted_blocked_stage_matrix_reporting_improvement",
        "round": 233,
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET,
        "measurement_only": True,
        "stage_sequence": list(_STAGE_SEQUENCE),
        "accepted_blocked_stage_matrix": matrix,
        "operator_summary": {
            "status": "accepted_blocked_stage_matrix_ready" if success else "blocked_accepted_blocked_stage_matrix",
            "accepted_tokens_for_future_review": accepted,
            "blocked_control_tokens": blocked,
            "blocked_control_stop_stages": {
                row["token"]: row["operator_stop_stage"] for row in matrix if row["classification"] == "blocked_control"
            },
            "unresolved_failure_tokens": [row["token"] for row in unresolved],
            "unresolved_first_failed_stages": {
                row["token"]: row["first_failed_stage"] for row in unresolved
            },
            "focused_test_command": FOCUSED_TEST_COMMAND,
            "next_recommendation": "validate the matrix, then use unresolved_first_failed_stages only for the next narrow non-persistence measurement target",
        },
        "policy_guard_snapshot": guards,
        "hard_stops": [] if success else ["stage_matrix_inputs_incomplete_or_policy_guard_failed"],
        "success": success,
        **POLICY_FLAGS,
    }


def build_round234_focused_test_plan(*, focused_tests_status: str = "pending") -> dict[str, Any]:
    return {
        "version": "v3_round234_stage_matrix_focused_tests",
        "round": 234,
        "focused_tests_status": focused_tests_status,
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "test_scope": [
            "Round231 consolidates green acceptance-taxonomy evidence and preserves policy guards",
            "Round232 selects one concrete non-persistence stage-matrix reporting target",
            "Round233 emits accepted/blocked stage rows for the Korean acceptance fixture without enablement",
            "CLI fails closed when the operator-local suite report is missing",
        ],
        **POLICY_FLAGS,
    }


def build_round235_broader_validation_delta(
    *,
    focused_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    return {
        "version": "v3_round235_stage_matrix_validation_delta_and_next_recommendation",
        "round": 235,
        "focused_tests_status": focused_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "broader_validation_delta": "added read-only accepted/blocked stage-matrix reporting over green runtime-mapping acceptance taxonomy evidence",
        "remaining_failures": [] if broader_pytest_status in {"green", "not_run"} else ["see_broader_pytest_taxonomy"],
        "next_recommendation": "use the stage matrix to choose the next narrow non-persistence measurement repair; production persistence remains NO-GO and runtime_mapping_enabled default remains false",
        **POLICY_FLAGS,
    }


def build_round231_235_report(suite_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    suite_report = _as_mapping(suite_report)
    taxonomy_report = build_round226_230_report(suite_report)
    round231 = build_round231_acceptance_taxonomy_green_evidence(taxonomy_report)
    round232 = build_round232_next_repair_target_selection(taxonomy_report, round231)
    round233 = build_round233_stage_matrix_reporting_improvement(taxonomy_report, round232)
    success = bool(taxonomy_report.get("success") is True and round231["success"] and round232["success"] and round233["success"])
    blockers = (
        _as_list(taxonomy_report.get("blockers"))
        + _as_list(round231.get("hard_stops"))
        + _as_list(round232.get("hard_stops"))
        + _as_list(round233.get("hard_stops"))
    )
    return {
        "version": ROUND231_235_VERSION,
        "rounds_completed": [231, 232, 233, 234, 235],
        "status": "runtime_mapping_acceptance_repair_selection_ready" if success else "blocked_runtime_mapping_acceptance_repair_selection",
        "success": success,
        "exit_code": 0 if success else 1,
        "round226_230_acceptance_taxonomy": taxonomy_report,
        "round231_acceptance_taxonomy_green_evidence": round231,
        "round232_next_repair_target_selection": round232,
        "round233_stage_matrix_reporting_improvement": round233,
        "round234_focused_test_plan": build_round234_focused_test_plan(focused_tests_status="defined"),
        "round235_broader_validation_delta": build_round235_broader_validation_delta(focused_tests_status="defined"),
        "taxonomy_green_evidence": {
            "source_status": taxonomy_report.get("status"),
            "source_success": taxonomy_report.get("success") is True,
            "selected_reporting_gap": taxonomy_report.get("selected_reporting_gap"),
            "accepted_tokens": _as_mapping(taxonomy_report.get("acceptance_delta_green_evidence")).get("accepted_tokens", []),
            "blocked_control_tokens": _as_mapping(taxonomy_report.get("acceptance_delta_green_evidence")).get("blocked_control_tokens", []),
        },
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET,
        "selected_reporting_gap": SELECTED_REPORTING_GAP,
        "implementation_summary": "read-only accepted/blocked stage-matrix reporting improvement; no enablement or persistence changes",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }


def run_acceptance_repair_selection_report(*, suite_report_path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    path = Path(suite_report_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return {
            "version": ROUND231_235_VERSION,
            "rounds_completed": [231, 232, 233, 234, 235],
            "status": "blocked_runtime_mapping_acceptance_repair_selection_missing_suite_report",
            "success": False,
            "exit_code": 1,
            "suite_report_path": str(suite_report_path),
            "blockers": ["operator_local_validation_suite_report_missing"],
            **POLICY_FLAGS,
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("operator suite report must be a JSON object")
    result = build_round231_235_report(report)
    result["suite_report_path"] = str(suite_report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the next runtime-mapping acceptance repair as a read-only stage matrix report.")
    parser.add_argument("--suite-report", default=str(DEFAULT_REPORT_PATH), help="Ignored operator-local suite JSON report path.")
    parser.add_argument("--output", help="Optional JSON output path for the repair-selection report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_acceptance_repair_selection_report(suite_report_path=args.suite_report)
    except Exception as exc:  # fail closed with compact JSON
        report = {
            "version": ROUND231_235_VERSION,
            "rounds_completed": [231, 232, 233, 234, 235],
            "status": "blocked_runtime_mapping_acceptance_repair_selection_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["runtime_mapping_acceptance_repair_selection_exception"],
            "exception": f"{exc.__class__.__name__}:{exc}",
            **POLICY_FLAGS,
        }
    if args.output:
        destination = Path(args.output)
        if not destination.is_absolute():
            destination = REPO_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_compact_json(report))
    return int(report.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
