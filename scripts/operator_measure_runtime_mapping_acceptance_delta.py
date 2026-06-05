#!/usr/bin/env python3
"""Round221-225 runtime-mapping acceptance delta measurement.

This script consumes the already-green one-command operator-local suite report
and normalizes the runtime-mapping-after-self-learning evidence into a compact,
stage-by-stage acceptance delta.  It does not build an engine, load medium30k,
enable runtime mapping by default, enable enforcement, enable production
persistence, call AGP, or create/vectorize any tokens.  Missing suite evidence
fails closed so the operator can rerun the existing one-command suite instead of
using manual Python snippets.
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

from scripts.operator_run_local_validation_suite import DEFAULT_REPORT_PATH, POLICY_FLAGS
from scripts.operator_remeasure_eve_self_learning import DEFAULT_TARGET_WORD, _compact_json
from scripts.operator_measure_runtime_mapping_after_self_learning import DEFAULT_NEGATIVE_TOKEN

ROUND221_225_VERSION = "v3_round221_225_runtime_mapping_acceptance_delta"
SELECTED_REPAIR_TARGET = "runtime_mapping_acceptance_delta_measurement_no_enablement"
SELECTED_FAILURE_CLUSTER = "post_green_runtime_mapping_stage_delta_visibility_gap"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _contains_token(value: Any, token: str) -> bool:
    return token in [str(item) for item in _as_list(value)]


def _operator_summary_from_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(report.get("operator_prompt_summary"))


def _mapping_measurement_from_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    for step in _as_list(report.get("steps")):
        parsed = _as_mapping(_as_mapping(step).get("parsed_json"))
        mapping = _as_mapping(parsed.get("mapping_measurement"))
        if mapping:
            return mapping
    return {}


def build_round221_green_evidence_interpretation(report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Interpret the green operator-local suite evidence as a repair target gate."""

    report = _as_mapping(report)
    summary = _operator_summary_from_report(report)
    exit_code = report.get("exit_code", 1)
    suite_green = report.get("success") is True and isinstance(exit_code, int) and exit_code == 0
    evidence = {
        "operator_local_validation_suite_green": suite_green,
        "summary_exit_code": summary.get("exit_code"),
        "status": summary.get("status") or report.get("status"),
        "target_word": summary.get("target_word", DEFAULT_TARGET_WORD),
        "selected_cluster_id": summary.get("selected_cluster_id"),
        "vector_lookup_after_commit": summary.get("vector_lookup_after_commit") is True,
        "wrapper_primary_loaded": summary.get("wrapper_primary_loaded") is True,
        "git_status_safety_safe": summary.get("git_status_safety_safe") is True,
        "production_persistence_enabled": summary.get("production_persistence_enabled") is True,
        "runtime_mapping_enabled_default": summary.get("runtime_mapping_enabled_default") is True,
    }
    ready = bool(
        evidence["operator_local_validation_suite_green"]
        and evidence["summary_exit_code"] == 0
        and evidence["vector_lookup_after_commit"] is True
        and evidence["wrapper_primary_loaded"] is True
        and evidence["git_status_safety_safe"] is True
        and evidence["production_persistence_enabled"] is False
        and evidence["runtime_mapping_enabled_default"] is False
    )
    return {
        "version": "v3_round221_operator_green_evidence_repair_target_gate",
        "round": 221,
        "green_evidence": evidence,
        "ready_for_next_narrow_repair_target": ready,
        "next_narrow_repair_target": SELECTED_REPAIR_TARGET if ready else "rerun_operator_local_validation_suite_until_green",
        "hard_stops": [] if ready else ["operator_local_green_evidence_incomplete"],
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def build_round222_failure_cluster_selection(round221: Mapping[str, Any]) -> dict[str, Any]:
    """Select one concrete cluster whose behavior can improve without enablement."""

    ready = round221.get("ready_for_next_narrow_repair_target") is True
    return {
        "version": "v3_round222_runtime_mapping_failure_cluster_selection",
        "round": 222,
        "selected_repair_target": SELECTED_REPAIR_TARGET if ready else "blocked_until_operator_suite_green",
        "selected_failure_cluster_id": SELECTED_FAILURE_CLUSTER if ready else "operator_green_evidence_missing",
        "cluster_definition": (
            "The operator suite proves the 민석 vector, guarded medium30k load, and controlled runtime-mapping smoke are green, "
            "but the next Codex round still lacks a normalized per-token stage delta showing exactly where future accepted or blocked tokens pass/fail."
        ),
        "behavior_to_improve": "operator-facing measurement determinism and failure localization, not runtime enablement",
        "why_no_persistence_needed": "all source evidence comes from the ignored local one-command suite JSON and in-memory controlled-smoke results",
        "blocked_actions": [
            "production_persistence_enablement",
            "runtime_mapping_enabled_default_true",
            "enforcement_enablement",
            "AGP_bypass",
            "vector_fabrication",
            "seed_or_subset_mutation",
        ],
        "hard_stops": [] if ready else ["round221_not_ready"],
        **POLICY_FLAGS,
    }


def classify_runtime_mapping_token(mapping_measurement: Mapping[str, Any], token: str) -> dict[str, Any]:
    """Classify one token across the existing mapping stages without mutation."""

    token = str(token)
    gate = _as_mapping(mapping_measurement.get("gate"))
    precheck = _as_mapping(mapping_measurement.get("precheck"))
    smoke = _as_mapping(mapping_measurement.get("controlled_smoke"))
    stages = {
        "would_map_at_gate": _contains_token(gate.get("would_map_tokens"), token) or mapping_measurement.get("target_would_map") is True and token == mapping_measurement.get("target_word"),
        "blocked_at_gate": _contains_token(gate.get("blocked_tokens"), token),
        "ready_at_precheck": _contains_token(precheck.get("ready_tokens"), token) or mapping_measurement.get("target_precheck_ready") is True and token == mapping_measurement.get("target_word"),
        "mapped_in_controlled_smoke": _contains_token(smoke.get("mapped_tokens"), token) or mapping_measurement.get("target_mapped_in_controlled_smoke") is True and token == mapping_measurement.get("target_word"),
        "rollback_complete": mapping_measurement.get("rollback_complete") is True or _as_mapping(smoke.get("rollback")).get("rollback_complete") is True,
        "runtime_mapping_enabled_after_false": mapping_measurement.get("runtime_mapping_enabled_after") is False,
        "enforcement_enabled_after_false": mapping_measurement.get("enforcement_enabled_after") is False,
        "runtime_mapping_not_persisted": mapping_measurement.get("runtime_mapping_persisted") is False,
    }
    accepted = bool(
        stages["would_map_at_gate"]
        and stages["ready_at_precheck"]
        and stages["mapped_in_controlled_smoke"]
        and stages["rollback_complete"]
        and stages["runtime_mapping_enabled_after_false"]
        and stages["enforcement_enabled_after_false"]
        and stages["runtime_mapping_not_persisted"]
    )
    blocked = bool(stages["blocked_at_gate"] and not accepted)
    failed_stage = None
    for name in ["would_map_at_gate", "ready_at_precheck", "mapped_in_controlled_smoke", "rollback_complete", "runtime_mapping_enabled_after_false", "enforcement_enabled_after_false", "runtime_mapping_not_persisted"]:
        if not stages[name]:
            failed_stage = name
            break
    return {
        "token": token,
        "classification_version": ROUND221_225_VERSION,
        "accepted_for_future_operator_review": accepted,
        "blocked_control": blocked,
        "first_failed_stage": None if accepted or blocked else failed_stage,
        "stages": stages,
    }


def build_round223_guarded_acceptance_delta_path(
    report: Mapping[str, Any] | None = None,
    *,
    mapping_measurement: Mapping[str, Any] | None = None,
    target_word: str = DEFAULT_TARGET_WORD,
    negative_token: str = DEFAULT_NEGATIVE_TOKEN,
) -> dict[str, Any]:
    """Build the guarded measurement/repair path for the selected cluster."""

    report = _as_mapping(report)
    mapping = _as_mapping(mapping_measurement) or _mapping_measurement_from_report(report)
    target = str(mapping.get("target_word") or target_word)
    negative = str(mapping.get("negative_token") or negative_token)
    tokens = [target]
    if negative not in tokens:
        tokens.append(negative)
    rows = [classify_runtime_mapping_token(mapping, token) for token in tokens] if mapping else []
    accepted_tokens = [row["token"] for row in rows if row["accepted_for_future_operator_review"]]
    blocked_controls = [row["token"] for row in rows if row["blocked_control"]]
    hard_stops = [] if mapping else ["mapping_measurement_missing_from_operator_suite_report"]
    status = "round223_guarded_acceptance_delta_ready" if not hard_stops else "blocked_round223_guarded_acceptance_delta"
    return {
        "version": "v3_round223_guarded_runtime_mapping_acceptance_delta_path",
        "round": 223,
        "selected_repair_target": SELECTED_REPAIR_TARGET,
        "selected_failure_cluster_id": SELECTED_FAILURE_CLUSTER,
        "status": status,
        "success": not hard_stops,
        "target_word": target,
        "negative_token": negative,
        "delta_rows": rows,
        "accepted_tokens": accepted_tokens,
        "blocked_control_tokens": blocked_controls,
        "repair_behavior": "normalize existing operator-local mapping evidence into deterministic per-token stage deltas for future focused repairs",
        "measurement_only": True,
        "hard_stops": hard_stops,
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def build_round224_focused_test_plan(*, focused_tests_status: str = "pending") -> dict[str, Any]:
    return {
        "version": "v3_round224_focused_acceptance_delta_tests",
        "round": 224,
        "focused_tests_status": focused_tests_status,
        "focused_test_command": "python -m pytest -q tests/test_v3_round221_225_runtime_mapping_acceptance_delta.py",
        "test_scope": [
            "green operator suite evidence selects the repair target",
            "one concrete failure cluster is selected without persistence or enablement",
            "민석 acceptance and EVE blocked-control rows are normalized from mapping evidence",
            "missing operator suite mapping evidence fails closed",
        ],
        **POLICY_FLAGS,
    }


def build_round225_broader_validation_delta(
    *,
    focused_tests_status: str = "pending",
    compileall_status: str = "pending",
    collect_only_status: str = "pending",
    broader_pytest_status: str = "not_run",
    broader_failure_count: int | None = None,
) -> dict[str, Any]:
    return {
        "version": "v3_round225_acceptance_delta_validation_and_next_recommendation",
        "round": 225,
        "focused_tests_status": focused_tests_status,
        "compileall_status": compileall_status,
        "collect_only_status": collect_only_status,
        "broader_pytest_status": broader_pytest_status,
        "broader_failure_count": broader_failure_count,
        "broader_validation_delta": "added read-only acceptance-delta normalization over green operator-local runtime-mapping evidence",
        "operator_local_suite_evidence_interpretation": "green one-command evidence is sufficient to choose measurement visibility as the next target; it is not evidence for production persistence enablement",
        "next_recommendation": "run the focused acceptance-delta command after the operator-local suite, then use the first_failed_stage taxonomy to choose the next narrow non-persistence repair",
        "remaining_failures": [] if broader_pytest_status in {"green", "not_run"} else ["see_broader_pytest_taxonomy"],
        "production_persistence_remains_no_go": True,
        **POLICY_FLAGS,
    }


def build_round221_225_report(report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report = _as_mapping(report)
    round221 = build_round221_green_evidence_interpretation(report)
    round222 = build_round222_failure_cluster_selection(round221)
    round223 = build_round223_guarded_acceptance_delta_path(report)
    success = bool(round221["ready_for_next_narrow_repair_target"] and not round222["hard_stops"] and round223["success"])
    blockers = list(round221.get("hard_stops", [])) + list(round222.get("hard_stops", [])) + list(round223.get("hard_stops", []))
    return {
        "version": ROUND221_225_VERSION,
        "rounds_completed": [221, 222, 223, 224, 225],
        "status": "runtime_mapping_acceptance_delta_ready" if success else "blocked_runtime_mapping_acceptance_delta",
        "success": success,
        "exit_code": 0 if success else 1,
        "round221_green_evidence_interpretation": round221,
        "round222_failure_cluster_selection": round222,
        "round223_guarded_acceptance_delta_path": round223,
        "round224_focused_test_plan": build_round224_focused_test_plan(focused_tests_status="defined"),
        "round225_broader_validation_delta": build_round225_broader_validation_delta(focused_tests_status="defined"),
        "selected_next_repair_target": SELECTED_REPAIR_TARGET,
        "selected_failure_cluster_id": SELECTED_FAILURE_CLUSTER,
        "blockers": sorted(set(str(item) for item in blockers)),
        **POLICY_FLAGS,
    }


def run_acceptance_delta_measurement(*, suite_report_path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    path = Path(suite_report_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return {
            "version": ROUND221_225_VERSION,
            "rounds_completed": [221, 222, 223, 224, 225],
            "status": "blocked_runtime_mapping_acceptance_delta_missing_suite_report",
            "success": False,
            "exit_code": 1,
            "suite_report_path": str(suite_report_path),
            "blockers": ["operator_local_validation_suite_report_missing"],
            **POLICY_FLAGS,
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("operator suite report must be a JSON object")
    result = build_round221_225_report(report)
    result["suite_report_path"] = str(suite_report_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize green operator-local runtime-mapping evidence into acceptance deltas.")
    parser.add_argument("--suite-report", default=str(DEFAULT_REPORT_PATH), help="Ignored operator-local suite JSON report path.")
    parser.add_argument("--output", help="Optional JSON output path for the acceptance-delta report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_acceptance_delta_measurement(suite_report_path=args.suite_report)
    except Exception as exc:  # fail closed with compact JSON
        report = {
            "version": ROUND221_225_VERSION,
            "rounds_completed": [221, 222, 223, 224, 225],
            "status": "blocked_runtime_mapping_acceptance_delta_exception",
            "success": False,
            "exit_code": 1,
            "blockers": ["runtime_mapping_acceptance_delta_exception"],
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
