#!/usr/bin/env python3
"""Round236-260 runtime-mapping acceptance handoff planner.

This helper continues from the Round231-235 accepted/blocked stage-matrix
report.  It converts the green, operator-local runtime-mapping acceptance
measurement into a deterministic handoff dossier with five five-round clusters:

* Round236-240: matrix quality gates and stop-stage taxonomy.
* Round241-245: operator handoff packet and replay checklist.
* Round246-250: validation manifest and command coverage.
* Round251-255: remaining taxonomy and non-go decision record.
* Round256-260: final consolidation and next recommendation.

The helper is reporting-only.  It never builds the engine, loads medium30k,
enables production persistence, changes the runtime_mapping_enabled default,
enables enforcement, calls AGP, fabricates vectors, mutates vectors, changes
semantic memory/quarantine, or writes production persistence state.
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

from scripts.operator_remeasure_eve_self_learning import _compact_json  # noqa: E402
from scripts.operator_run_local_validation_suite import DEFAULT_REPORT_PATH, POLICY_FLAGS  # noqa: E402
from scripts.operator_select_runtime_mapping_acceptance_repair import (  # noqa: E402
    FOCUSED_TEST_COMMAND as ROUND231_235_FOCUSED_TEST_COMMAND,
    ROUND231_235_VERSION,
    SELECTED_NEXT_REPAIR_TARGET,
    build_round231_235_report,
)

ROUND236_260_VERSION = "v3_round236_260_runtime_mapping_acceptance_handoff"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round236_260_runtime_mapping_acceptance_handoff.py"
ADJACENT_TEST_COMMANDS = [
    "python -m pytest -q tests/test_v3_round221_225_runtime_mapping_acceptance_delta.py tests/test_v3_round226_230_runtime_mapping_acceptance_taxonomy.py tests/test_v3_round231_235_runtime_mapping_acceptance_repair_selection.py tests/test_v3_round236_260_runtime_mapping_acceptance_handoff.py",
]
BROADER_VALIDATION_COMMANDS = [
    "python -m pytest --collect-only -q",
    "python -m compileall -q .",
]
CLUSTERS = [
    {
        "cluster_id": "round236_240_stage_matrix_quality_gates",
        "rounds": [236, 237, 238, 239, 240],
        "cluster": "stage_matrix_quality_gates",
        "purpose": "turn the accepted/blocked stage matrix into explicit quality gates, stop-stage counts, and a handoff-safe focused-test plan",
    },
    {
        "cluster_id": "round241_245_operator_handoff_packet",
        "rounds": [241, 242, 243, 244, 245],
        "cluster": "operator_handoff_packet",
        "purpose": "package accepted and blocked rows into an operator replay checklist without enabling runtime mapping or persistence",
    },
    {
        "cluster_id": "round246_250_validation_manifest",
        "rounds": [246, 247, 248, 249, 250],
        "cluster": "validation_manifest",
        "purpose": "record focused, adjacent, collect-only, and compile validation commands and the broader validation delta",
    },
    {
        "cluster_id": "round251_255_remaining_taxonomy",
        "rounds": [251, 252, 253, 254, 255],
        "cluster": "remaining_taxonomy",
        "purpose": "classify remaining taxonomy into accepted tokens, blocked controls, unresolved failures, no-go items, and explicit non-goals",
    },
    {
        "cluster_id": "round256_260_final_consolidation",
        "rounds": [256, 257, 258, 259, 260],
        "cluster": "final_consolidation",
        "purpose": "consolidate the five clusters into a final PR-ready recommendation while preserving all safety defaults",
    },
]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _policy_guard_snapshot(*reports: Mapping[str, Any]) -> dict[str, bool]:
    merged: dict[str, Any] = {}
    for report in reports:
        for key in POLICY_FLAGS:
            if key in report:
                merged[key] = report.get(key)
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


def _round233(previous_report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(previous_report.get("round233_stage_matrix_reporting_improvement"))


def _matrix(previous_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_as_mapping(row) for row in _as_list(_round233(previous_report).get("accepted_blocked_stage_matrix"))]


def _tokens_by_class(matrix: Sequence[Mapping[str, Any]], classification: str) -> list[str]:
    return [str(row.get("token")) for row in matrix if row.get("classification") == classification]


def _stop_stage_counts(matrix: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in matrix:
        stage = row.get("operator_stop_stage")
        key = "none" if stage is None else str(stage)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stage_coverage(matrix: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for row in matrix:
        for stage in _as_list(row.get("stage_outcomes")):
            stage_map = _as_mapping(stage)
            name = str(stage_map.get("stage"))
            bucket = coverage.setdefault(name, {"passed": 0, "failed": 0})
            if stage_map.get("passed") is True:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
    return {key: coverage[key] for key in sorted(coverage)}


def _matrix_quality_gate(previous_report: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _matrix(previous_report)
    accepted = _tokens_by_class(matrix, "accepted_for_future_operator_review")
    blocked = _tokens_by_class(matrix, "blocked_control")
    unresolved = _tokens_by_class(matrix, "unresolved_failure")
    stop_stage_counts = _stop_stage_counts(matrix)
    guards = _policy_guard_snapshot(previous_report, _round233(previous_report))
    row_count_matches_taxonomy = len(matrix) == len(accepted) + len(blocked) + len(unresolved)
    blocked_controls_stop_at_gate = all(
        row.get("operator_stop_stage") == "blocked_at_gate"
        for row in matrix
        if row.get("classification") == "blocked_control"
    )
    accepted_rows_do_not_stop = all(
        row.get("operator_stop_stage") is None
        for row in matrix
        if row.get("classification") == "accepted_for_future_operator_review"
    )
    success = bool(
        previous_report.get("success") is True
        and matrix
        and row_count_matches_taxonomy
        and blocked_controls_stop_at_gate
        and accepted_rows_do_not_stop
        and not unresolved
        and all(guards.values())
    )
    return {
        "version": "v3_round236_240_stage_matrix_quality_gate",
        "rounds": [236, 237, 238, 239, 240],
        "cluster": "stage_matrix_quality_gates",
        "success": success,
        "measurement_only": True,
        "source_version": previous_report.get("version", ROUND231_235_VERSION),
        "selected_input_target": previous_report.get("selected_next_repair_target", SELECTED_NEXT_REPAIR_TARGET),
        "matrix_row_count": len(matrix),
        "accepted_tokens": accepted,
        "blocked_control_tokens": blocked,
        "unresolved_failure_tokens": unresolved,
        "stop_stage_counts": stop_stage_counts,
        "stage_coverage": _stage_coverage(matrix),
        "quality_gates": {
            "row_count_matches_taxonomy": row_count_matches_taxonomy,
            "blocked_controls_stop_at_gate": blocked_controls_stop_at_gate,
            "accepted_rows_do_not_stop": accepted_rows_do_not_stop,
            "unresolved_failure_tokens_empty": not unresolved,
            **guards,
        },
        "focused_test_command": FOCUSED_TEST_COMMAND,
        **POLICY_FLAGS,
    }


def _operator_handoff_packet(previous_report: Mapping[str, Any], quality_gate: Mapping[str, Any]) -> dict[str, Any]:
    accepted = _as_list(quality_gate.get("accepted_tokens"))
    blocked = _as_list(quality_gate.get("blocked_control_tokens"))
    unresolved = _as_list(quality_gate.get("unresolved_failure_tokens"))
    replay_rows = [
        {
            "token": token,
            "operator_action": "review_as_future_runtime_mapping_candidate",
            "required_boundary": "must remain explicit operator authorization; do not change runtime_mapping_enabled default",
        }
        for token in accepted
    ] + [
        {
            "token": token,
            "operator_action": "keep_as_blocked_control_until concept/SA/AGP evidence exists",
            "required_boundary": "blocked control is evidence that broad lexical auto-mapping remains unsafe",
        }
        for token in blocked
    ]
    guards = _policy_guard_snapshot(previous_report, quality_gate)
    success = bool(quality_gate.get("success") is True and replay_rows and not unresolved and all(guards.values()))
    return {
        "version": "v3_round241_245_operator_handoff_packet",
        "rounds": [241, 242, 243, 244, 245],
        "cluster": "operator_handoff_packet",
        "success": success,
        "measurement_only": True,
        "accepted_for_future_operator_review": accepted,
        "blocked_controls": blocked,
        "unresolved_failure_tokens": unresolved,
        "operator_replay_rows": replay_rows,
        "operator_summary": {
            "status": "handoff_packet_ready" if success else "blocked_handoff_packet",
            "accepted_count": len(accepted),
            "blocked_control_count": len(blocked),
            "unresolved_failure_count": len(unresolved),
            "runtime_mapping_enablement": "not_enabled",
            "production_persistence": "not_enabled",
            "enforcement": "not_enabled",
        },
        "handoff_documents": [
            "ROUND_V3_R236_R240_HANDOFF.md",
            "ROUND_V3_R241_R245_HANDOFF.md",
        ],
        "quality_gates": guards,
        **POLICY_FLAGS,
    }


def _validation_manifest(previous_report: Mapping[str, Any], quality_gate: Mapping[str, Any], handoff: Mapping[str, Any]) -> dict[str, Any]:
    commands = {
        "focused": [FOCUSED_TEST_COMMAND],
        "previous_focused": [ROUND231_235_FOCUSED_TEST_COMMAND],
        "adjacent": ADJACENT_TEST_COMMANDS,
        "broader": BROADER_VALIDATION_COMMANDS,
    }
    guards = _policy_guard_snapshot(previous_report, quality_gate, handoff)
    success = bool(quality_gate.get("success") is True and handoff.get("success") is True and all(guards.values()))
    return {
        "version": "v3_round246_250_validation_manifest",
        "rounds": [246, 247, 248, 249, 250],
        "cluster": "validation_manifest",
        "success": success,
        "measurement_only": True,
        "validation_commands": commands,
        "focused_test_status": "defined_for_round236_260",
        "adjacent_test_status": "defined_for_runtime_mapping_acceptance_lineage",
        "broader_validation_delta": "adds handoff-quality gates and taxonomy summaries; no runtime enablement or persistence delta",
        "validation_json_files": [
            "ROUND_V3_R236_R240_VALIDATION.json",
            "ROUND_V3_R241_R245_VALIDATION.json",
            "ROUND_V3_R246_R250_VALIDATION.json",
        ],
        "quality_gates": guards,
        **POLICY_FLAGS,
    }


def _remaining_taxonomy(quality_gate: Mapping[str, Any], handoff: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    unresolved = _as_list(quality_gate.get("unresolved_failure_tokens"))
    no_go_items = [
        "production_persistence_enablement",
        "runtime_mapping_enabled_default_true",
        "enforcement_enablement",
        "agp_bypass",
        "fabricated_or_dummy_vectors",
        "seed_vector_mutation",
        "semantic_memory_or_quarantine_mutation_from_mapping_report",
        "broad_lexical_auto_mapping",
    ]
    remaining = {
        "accepted_for_future_operator_review": _as_list(quality_gate.get("accepted_tokens")),
        "blocked_controls": _as_list(quality_gate.get("blocked_control_tokens")),
        "unresolved_failures": unresolved,
        "explicit_no_go_items": no_go_items,
    }
    success = bool(quality_gate.get("success") is True and handoff.get("success") is True and validation.get("success") is True and not unresolved)
    return {
        "version": "v3_round251_255_remaining_taxonomy",
        "rounds": [251, 252, 253, 254, 255],
        "cluster": "remaining_taxonomy",
        "success": success,
        "measurement_only": True,
        "remaining_taxonomy": remaining,
        "decision_record": {
            "runtime_mapping_production_enablement": "no_go",
            "reason": "acceptance evidence is operator-review measurement only; default no-load/no-enable boundary must remain intact",
            "next_safe_action": "operator may review the accepted token row and blocked-control row before any separate authorization round",
        },
        "handoff_documents": ["ROUND_V3_R251_R255_HANDOFF.md"],
        **POLICY_FLAGS,
    }


def _final_consolidation(
    previous_report: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    handoff: Mapping[str, Any],
    validation: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_statuses = {
        "round236_240_stage_matrix_quality_gates": quality_gate.get("success") is True,
        "round241_245_operator_handoff_packet": handoff.get("success") is True,
        "round246_250_validation_manifest": validation.get("success") is True,
        "round251_255_remaining_taxonomy": taxonomy.get("success") is True,
    }
    guards = _policy_guard_snapshot(previous_report, quality_gate, handoff, validation, taxonomy)
    success = bool(all(cluster_statuses.values()) and all(guards.values()))
    return {
        "version": "v3_round256_260_final_consolidation",
        "rounds": [256, 257, 258, 259, 260],
        "cluster": "final_consolidation",
        "success": success,
        "measurement_only": True,
        "cluster_statuses": cluster_statuses,
        "clusters_selected": [cluster["cluster"] for cluster in CLUSTERS],
        "rounds_completed": list(range(236, 261)),
        "fixes_measurement_improvements": [
            "stage-matrix quality gates over accepted/blocked runtime-mapping acceptance rows",
            "operator handoff replay rows for accepted and blocked-control tokens",
            "validation manifest linking focused, adjacent, collect-only, and compile checks",
            "remaining taxonomy with explicit no-go items and next safe action",
        ],
        "broader_validation_delta": validation.get("broader_validation_delta"),
        "remaining_taxonomy": _as_mapping(taxonomy.get("remaining_taxonomy")),
        "next_recommendation": "stop short of production enablement; if the operator explicitly authorizes a future round, use the handoff packet to design one isolated no-persistence runtime-mapping rehearsal with split full-suite validation",
        "quality_gates": guards,
        **POLICY_FLAGS,
    }


def build_round236_260_report(suite_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    suite_report = _as_mapping(suite_report)
    previous_report = build_round231_235_report(suite_report)
    quality_gate = _matrix_quality_gate(previous_report)
    handoff = _operator_handoff_packet(previous_report, quality_gate)
    validation = _validation_manifest(previous_report, quality_gate, handoff)
    taxonomy = _remaining_taxonomy(quality_gate, handoff, validation)
    final = _final_consolidation(previous_report, quality_gate, handoff, validation, taxonomy)
    blockers: list[str] = []
    if previous_report.get("success") is not True:
        blockers.append("round231_235_source_not_green")
    for key, value in final.get("quality_gates", {}).items():
        if value is not True:
            blockers.append(f"policy_guard_failed:{key}")
    success = bool(previous_report.get("success") is True and final.get("success") is True and not blockers)
    return {
        "version": ROUND236_260_VERSION,
        "rounds_completed": list(range(236, 261)),
        "status": "runtime_mapping_acceptance_handoff_ready" if success else "blocked_runtime_mapping_acceptance_handoff",
        "success": success,
        "exit_code": 0 if success else 1,
        "source_round231_235": previous_report,
        "clusters_selected": CLUSTERS,
        "round236_240_stage_matrix_quality_gates": quality_gate,
        "round241_245_operator_handoff_packet": handoff,
        "round246_250_validation_manifest": validation,
        "round251_255_remaining_taxonomy": taxonomy,
        "round256_260_final_consolidation": final,
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "adjacent_test_commands": ADJACENT_TEST_COMMANDS,
        "broader_validation_commands": BROADER_VALIDATION_COMMANDS,
        "blockers": sorted(set(blockers)),
        **POLICY_FLAGS,
    }


def run_runtime_mapping_acceptance_handoff_report(*, suite_report_path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    path = Path(suite_report_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return {
            "version": ROUND236_260_VERSION,
            "rounds_completed": [],
            "status": "blocked_runtime_mapping_acceptance_handoff_missing_suite_report",
            "success": False,
            "exit_code": 1,
            "suite_report_path": str(suite_report_path),
            "blockers": ["operator_local_validation_suite_report_missing"],
            **POLICY_FLAGS,
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("operator suite report must be a JSON object")
    result = build_round236_260_report(report)
    result["suite_report_path"] = str(suite_report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Round236-260 runtime-mapping acceptance handoff report.")
    parser.add_argument("--suite-report", default=str(DEFAULT_REPORT_PATH), help="Ignored operator-local suite JSON report path.")
    parser.add_argument("--output", help="Optional JSON output path for the handoff report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_runtime_mapping_acceptance_handoff_report(suite_report_path=args.suite_report)
    except Exception as exc:  # fail closed with compact JSON
        report = {
            "version": ROUND236_260_VERSION,
            "rounds_completed": [],
            "status": "blocked_runtime_mapping_acceptance_handoff_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["runtime_mapping_acceptance_handoff_exception"],
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
