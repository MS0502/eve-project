#!/usr/bin/env python3
"""Round226-230 runtime-mapping acceptance taxonomy report.

This reporting helper starts from the green Round221-225 acceptance-delta
measurement and adds a deterministic operator-facing taxonomy for the next
non-persistence repair target.  It reads only the ignored one-command operator
suite report, never builds an engine, never loads medium30k, never enables
runtime mapping by default, never enables enforcement, never calls AGP, and
never mutates vectors, semantic memory, quarantine, or persistence state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_measure_runtime_mapping_acceptance_delta import (  # noqa: E402
    ROUND221_225_VERSION,
    SELECTED_FAILURE_CLUSTER,
    build_round221_225_report,
)
from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import DEFAULT_REPORT_PATH, POLICY_FLAGS  # noqa: E402

ROUND226_230_VERSION = "v3_round226_230_runtime_mapping_acceptance_taxonomy"
SELECTED_NEXT_REPAIR_TARGET = "runtime_mapping_acceptance_taxonomy_reporting_no_enablement"
SELECTED_REPORTING_GAP = "accepted_blocked_stage_taxonomy_missing_from_acceptance_delta"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round226_230_runtime_mapping_acceptance_taxonomy.py"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _delta_rows(acceptance_delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    round223 = _as_mapping(acceptance_delta.get("round223_guarded_acceptance_delta_path"))
    return [_as_mapping(row) for row in _as_list(round223.get("delta_rows"))]


def _accepted_tokens(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("token")) for row in rows if row.get("accepted_for_future_operator_review") is True]


def _blocked_control_tokens(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("token")) for row in rows if row.get("blocked_control") is True]


def _failed_stage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if row.get("accepted_for_future_operator_review") is True or row.get("blocked_control") is True:
            continue
        stage = row.get("first_failed_stage")
        counter[str(stage or "unclassified_failure")] += 1
    return dict(sorted(counter.items()))


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


def build_round226_acceptance_delta_green_consolidation(
    acceptance_delta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consolidate Round221-225 green evidence and identify the next target."""

    acceptance_delta = _as_mapping(acceptance_delta)
    rows = _delta_rows(acceptance_delta)
    guards = _policy_guard_snapshot(acceptance_delta)
    success = bool(acceptance_delta.get("success") is True and rows and all(guards.values()))
    return {
        "version": "v3_round226_acceptance_delta_green_evidence_consolidation",
        "round": 226,
        "source_version": acceptance_delta.get("version", ROUND221_225_VERSION),
        "acceptance_delta_success": acceptance_delta.get("success") is True,
        "acceptance_delta_status": acceptance_delta.get("status"),
        "accepted_tokens": _accepted_tokens(rows),
        "blocked_control_tokens": _blocked_control_tokens(rows),
        "policy_guard_snapshot": guards,
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET if success else "rerun_round221_225_acceptance_delta_until_green",
        "next_concrete_non_persistence_repair_target": SELECTED_NEXT_REPAIR_TARGET if success else "blocked_until_acceptance_delta_green",
        "hard_stops": [] if success else ["acceptance_delta_green_evidence_incomplete_or_policy_guard_failed"],
        "success": success,
        **POLICY_FLAGS,
    }


def build_round227_reporting_gap_selection(round226: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select one narrow acceptance/reporting gap without enablement."""

    round226 = _as_mapping(round226)
    ready = round226.get("success") is True
    return {
        "version": "v3_round227_runtime_mapping_reporting_gap_selection",
        "round": 227,
        "selected_reporting_gap": SELECTED_REPORTING_GAP if ready else "blocked_until_round226_green_consolidation",
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET if ready else "blocked_until_round226_green_consolidation",
        "selected_failure_cluster_id": SELECTED_FAILURE_CLUSTER if ready else "round226_green_consolidation_missing",
        "gap_definition": (
            "Round221-225 exposes per-token rows, but does not provide an operator-ready taxonomy that summarizes accepted tokens, "
            "blocked controls, unresolved first-failed-stage counts, policy guard status, and the exact next validation command."
        ),
        "implementation_scope": "reporting-only taxonomy over existing acceptance-delta rows",
        "explicit_non_goals": [
            "production_persistence_enablement",
            "runtime_mapping_enabled_default_true",
            "enforcement_enablement",
            "AGP_bypass",
            "new_vector_creation_or_fabrication",
            "semantic_memory_or_quarantine_mutation",
        ],
        "hard_stops": [] if ready else ["round226_not_ready"],
        "success": ready,
        **POLICY_FLAGS,
    }


def build_round228_acceptance_taxonomy_improvement(
    acceptance_delta: Mapping[str, Any] | None = None,
    round227: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Implement the narrow reporting improvement as read-only data."""

    acceptance_delta = _as_mapping(acceptance_delta)
    round227 = _as_mapping(round227)
    rows = _delta_rows(acceptance_delta)
    accepted = _accepted_tokens(rows)
    blocked = _blocked_control_tokens(rows)
    failed_counts = _failed_stage_counts(rows)
    policy_guards = _policy_guard_snapshot(acceptance_delta, round227)
    success = bool(round227.get("success") is True and rows and accepted and blocked and all(policy_guards.values()))
    return {
        "version": "v3_round228_acceptance_taxonomy_reporting_improvement",
        "round": 228,
        "selected_reporting_gap": SELECTED_REPORTING_GAP,
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET,
        "measurement_only": True,
        "taxonomy": {
            "total_rows": len(rows),
            "accepted_count": len(accepted),
            "blocked_control_count": len(blocked),
            "unresolved_failure_count": sum(failed_counts.values()),
            "accepted_tokens": accepted,
            "blocked_control_tokens": blocked,
            "failed_stage_counts": failed_counts,
        },
        "operator_summary": {
            "status": "runtime_mapping_acceptance_taxonomy_ready" if success else "blocked_runtime_mapping_acceptance_taxonomy",
            "accepted_tokens_for_future_review": accepted,
            "blocked_control_tokens": blocked,
            "first_failed_stage_counts": failed_counts,
            "focused_test_command": FOCUSED_TEST_COMMAND,
            "next_recommendation": "keep runtime mapping disabled by default and use this taxonomy to choose the next narrow reporting or acceptance repair",
        },
        "policy_guard_snapshot": policy_guards,
        "hard_stops": [] if success else ["acceptance_taxonomy_inputs_incomplete_or_policy_guard_failed"],
        "success": success,
        **POLICY_FLAGS,
    }


def build_round229_focused_test_plan(*, focused_tests_status: str = "pending") -> dict[str, Any]:
    return {
        "version": "v3_round229_acceptance_taxonomy_focused_tests",
        "round": 229,
        "focused_tests_status": focused_tests_status,
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "test_scope": [
            "Round226 consolidates green acceptance-delta evidence and preserves no-go policy flags",
            "Round227 selects one reporting-only gap without runtime enablement",
            "Round228 emits accepted/blocked/failure taxonomy and focused command",
            "CLI fails closed when the operator-local suite report is missing",
        ],
        **POLICY_FLAGS,
    }


def build_round230_broader_validation_delta(
    *,
    focused_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    return {
        "version": "v3_round230_acceptance_taxonomy_validation_and_next_recommendation",
        "round": 230,
        "focused_tests_status": focused_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "broader_validation_delta": "added read-only taxonomy reporting over the green runtime-mapping acceptance-delta measurement",
        "remaining_failures": [] if broader_pytest_status in {"green", "not_run"} else ["see_broader_pytest_taxonomy"],
        "next_recommendation": "operator may use the taxonomy to inspect unresolved first_failed_stage counts; production persistence remains NO-GO and runtime_mapping_enabled default remains false",
        **POLICY_FLAGS,
    }


def build_round226_230_report(suite_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    suite_report = _as_mapping(suite_report)
    acceptance_delta = build_round221_225_report(suite_report)
    round226 = build_round226_acceptance_delta_green_consolidation(acceptance_delta)
    round227 = build_round227_reporting_gap_selection(round226)
    round228 = build_round228_acceptance_taxonomy_improvement(acceptance_delta, round227)
    success = bool(acceptance_delta.get("success") is True and round226["success"] and round227["success"] and round228["success"])
    blockers = (
        _as_list(acceptance_delta.get("blockers"))
        + _as_list(round226.get("hard_stops"))
        + _as_list(round227.get("hard_stops"))
        + _as_list(round228.get("hard_stops"))
    )
    return {
        "version": ROUND226_230_VERSION,
        "rounds_completed": [226, 227, 228, 229, 230],
        "status": "runtime_mapping_acceptance_taxonomy_ready" if success else "blocked_runtime_mapping_acceptance_taxonomy",
        "success": success,
        "exit_code": 0 if success else 1,
        "round221_225_acceptance_delta": acceptance_delta,
        "round226_acceptance_delta_green_consolidation": round226,
        "round227_reporting_gap_selection": round227,
        "round228_acceptance_taxonomy_improvement": round228,
        "round229_focused_test_plan": build_round229_focused_test_plan(focused_tests_status="defined"),
        "round230_broader_validation_delta": build_round230_broader_validation_delta(focused_tests_status="defined"),
        "acceptance_delta_green_evidence": {
            "source_status": acceptance_delta.get("status"),
            "source_success": acceptance_delta.get("success") is True,
            "selected_failure_cluster_id": acceptance_delta.get("selected_failure_cluster_id"),
            "accepted_tokens": round226.get("accepted_tokens", []),
            "blocked_control_tokens": round226.get("blocked_control_tokens", []),
        },
        "selected_next_repair_target": SELECTED_NEXT_REPAIR_TARGET,
        "selected_reporting_gap": SELECTED_REPORTING_GAP,
        "implementation_summary": "read-only runtime-mapping acceptance taxonomy/reporting improvement; no enablement or persistence changes",
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }


def run_acceptance_taxonomy_report(*, suite_report_path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    path = Path(suite_report_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return {
            "version": ROUND226_230_VERSION,
            "rounds_completed": [226, 227, 228, 229, 230],
            "status": "blocked_runtime_mapping_acceptance_taxonomy_missing_suite_report",
            "success": False,
            "exit_code": 1,
            "suite_report_path": str(suite_report_path),
            "blockers": ["operator_local_validation_suite_report_missing"],
            **POLICY_FLAGS,
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("operator suite report must be a JSON object")
    result = build_round226_230_report(report)
    result["suite_report_path"] = str(suite_report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize runtime-mapping acceptance deltas as a read-only taxonomy report.")
    parser.add_argument("--suite-report", default=str(DEFAULT_REPORT_PATH), help="Ignored operator-local suite JSON report path.")
    parser.add_argument("--output", help="Optional JSON output path for the acceptance-taxonomy report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_acceptance_taxonomy_report(suite_report_path=args.suite_report)
    except Exception as exc:  # fail closed with compact JSON
        report = {
            "version": ROUND226_230_VERSION,
            "rounds_completed": [226, 227, 228, 229, 230],
            "status": "blocked_runtime_mapping_acceptance_taxonomy_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["runtime_mapping_acceptance_taxonomy_exception"],
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
