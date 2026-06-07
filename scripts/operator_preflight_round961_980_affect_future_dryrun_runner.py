#!/usr/bin/env python3
"""Emit compact Round961-980 affect future dry-run runner preflight JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_event_to_axis_proposal_map import event_to_axis_proposal_map
from adapters.affect_future_dryrun_simulation_request_packet import build_affect_future_dryrun_simulation_request_packet
from adapters.affect_future_dryrun_simulation_runner_preflight import (
    affect_future_dryrun_simulation_runner_preflight_summary,
    build_affect_future_dryrun_simulation_runner_preflight,
    build_affect_future_dryrun_simulation_runner_preflight_from_request_packet,
    validate_affect_future_dryrun_simulation_runner_preflight,
)


def _compact(preflight: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_affect_future_dryrun_simulation_runner_preflight(preflight)
    safety = preflight.get("event_safety_result", {}) if isinstance(preflight.get("event_safety_result"), Mapping) else {}
    return {
        "runner_preflight_passed": preflight.get("runner_preflight_passed"),
        "runner_preflight_status": preflight.get("runner_preflight_status"),
        "validation_passed": validation.get("validation_passed"),
        "event_category": preflight.get("event_category"),
        "source_operator_decision": preflight.get("source_operator_decision"),
        "source_request_packet_passed": preflight.get("source_request_packet_passed"),
        "source_future_request_allowed": preflight.get("source_future_request_allowed"),
        "source_requested_now": preflight.get("source_requested_now"),
        "source_dryrun_apply_allowed": preflight.get("source_dryrun_apply_allowed"),
        "source_live_apply_allowed": preflight.get("source_live_apply_allowed"),
        "runner_preflight_built": preflight.get("runner_preflight_built"),
        "request_packet_accepted": preflight.get("request_packet_accepted"),
        "future_simulation_runner_eligible": preflight.get("future_simulation_runner_eligible"),
        "future_simulation_execution_requested_now": preflight.get("future_simulation_execution_requested_now"),
        "future_simulation_executed": preflight.get("future_simulation_executed"),
        "dryrun_apply_executed": preflight.get("dryrun_apply_executed"),
        "dryrun_apply_allowed": preflight.get("dryrun_apply_allowed"),
        "live_apply_allowed": preflight.get("live_apply_allowed"),
        "apply_permission_granted": preflight.get("apply_permission_granted"),
        "checkpoint_created": preflight.get("checkpoint_created"),
        "rollback_created": preflight.get("rollback_created"),
        "audit_written": preflight.get("audit_written"),
        "checkpoint_write_allowed": preflight.get("checkpoint_write_allowed"),
        "rollback_write_allowed": preflight.get("rollback_write_allowed"),
        "audit_write_allowed": preflight.get("audit_write_allowed"),
        "runtime_mutation_performed": preflight.get("runtime_mutation_performed"),
        "persistence_write_performed": preflight.get("persistence_write_performed"),
        "memory_write_performed": preflight.get("memory_write_performed"),
        "vector_content_read_load": preflight.get("vector_contents_read") is True or preflight.get("vectors_loaded") is True or preflight.get("vector_read_performed") is True or preflight.get("vector_load_performed") is True,
        "artifact_created_or_staged": preflight.get("artifact_created_or_staged"),
        "global_synchrony_blocked": preflight.get("global_synchrony_blocked"),
        "agp_fallback_non_bypass": safety.get("agp_fallback_non_bypass"),
        "blocked_reasons": preflight.get("blocked_reasons"),
        "validation_blocked_reasons": validation.get("blocked_reasons"),
        "event_safety_result": safety,
    }


def _proof(preflight: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_execution_now": preflight.get("execution_requested_now") is False and preflight.get("execution_allowed_now") is False and preflight.get("future_simulation_execution_requested_now") is False,
        "no_future_simulation_executed": preflight.get("future_simulation_executed") is False,
        "no_dryrun_apply_executed": preflight.get("dryrun_apply_executed") is False,
        "no_checkpoint_created": preflight.get("checkpoint_created") is False,
        "no_rollback_created": preflight.get("rollback_created") is False,
        "no_audit_written": preflight.get("audit_written") is False,
        "no_checkpoint_write": preflight.get("checkpoint_write_allowed") is False,
        "no_rollback_write": preflight.get("rollback_write_allowed") is False,
        "no_audit_write": preflight.get("audit_write_allowed") is False,
        "no_dryrun_apply_permission": preflight.get("dryrun_apply_allowed") is False and preflight.get("apply_permission_granted") is False,
        "no_live_apply": preflight.get("live_apply_allowed") is False,
        "no_runtime_mutation": preflight.get("runtime_mutation_performed") is False and preflight.get("state_mutation_performed") is False,
        "no_persistence": preflight.get("persistence_write_performed") is False and preflight.get("production_persistence_enabled") is False,
        "no_memory_write": preflight.get("memory_write_performed") is False,
        "no_vector_content_read_load": preflight.get("vector_read_performed") is False and preflight.get("vector_load_performed") is False and preflight.get("vector_contents_read") is False and preflight.get("vectors_loaded") is False,
        "no_artifact_creation_staging": preflight.get("artifact_created_or_staged") is False,
        "hardware_non_panic": preflight.get("event_safety_result", {}).get("hardware_low_power_and_below_non_panic_operational_only") is True,
        "anti_global_synchrony": preflight.get("global_synchrony_blocked") is True and preflight.get("event_safety_result", {}).get("global_synchrony_blocked") is True,
        "agp_fallback_non_bypass": preflight.get("agp_bypass_allowed") is False and preflight.get("fallback_bypass_allowed") is False and preflight.get("event_safety_result", {}).get("agp_fallback_non_bypass") is True,
    }


def _mutated_source_request(**overrides: Any) -> dict[str, Any]:
    packet = build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation")
    packet.update(overrides)
    return packet


def main() -> None:
    common_imagination = {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True}
    preflights = {
        "approve_requested_runner_preflight_result": build_affect_future_dryrun_simulation_runner_preflight("praise", {"competence_drive": 0.02}),
        "non_approval_request_blocked_result": build_affect_future_dryrun_simulation_runner_preflight("praise", {"competence_drive": 0.02}, "reject"),
        "source_request_failure_blocks_runner_proof": build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_mutated_source_request(request_packet_passed=False)),
        "requested_now_source_blocks_runner_proof": build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_mutated_source_request(requested_now=True)),
        "dryrun_apply_allowed_source_blocks_runner_proof": build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_mutated_source_request(dryrun_apply_allowed=True)),
        "live_apply_allowed_source_blocks_runner_proof": build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(_mutated_source_request(live_apply_allowed=True)),
        "unknown_event_fail_closed_result": build_affect_future_dryrun_simulation_runner_preflight("unknown_event", {}),
        "malicious_comment_runner_preflight_safety_result": build_affect_future_dryrun_simulation_runner_preflight("malicious_comment", {"boundary_defense": 0.02}),
        "identity_attack_runner_preflight_safety_result": build_affect_future_dryrun_simulation_runner_preflight("identity_attack", {"threat_pressure": 0.02}),
        "hardware_normal_zero_delta_runner_result": build_affect_future_dryrun_simulation_runner_preflight("hardware_normal", {}),
        "hardware_low_power_non_panic_runner_result": build_affect_future_dryrun_simulation_runner_preflight("hardware_low_power", {"energy_budget": -0.02}),
        "hardware_prediction_error_diagnostic_only_runner_result": build_affect_future_dryrun_simulation_runner_preflight("hardware_prediction_error", {"stability_need": 0.02}),
        "speech_pressure_agp_fallback_safe_runner_result": build_affect_future_dryrun_simulation_runner_preflight("speech_output_pressure", {"expression_pressure": 0.02}),
        "imagination_negative_spiral_boundary_preserving_runner_result": build_affect_future_dryrun_simulation_runner_preflight("imagination_negative_spiral", {"recovery_need": 0.02}, metadata=common_imagination),
        "memory_self_update_quarantine_preserving_runner_result": build_affect_future_dryrun_simulation_runner_preflight("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}),
    }
    all_axis = {axis: 0.01 for row in event_to_axis_proposal_map().values() for axis in row.get("allowed_axis_deltas", {})}
    preflights["anti_global_synchrony_all_axis_blocked_result"] = build_affect_future_dryrun_simulation_runner_preflight("praise", all_axis)

    summary = affect_future_dryrun_simulation_runner_preflight_summary()
    approve = preflights["approve_requested_runner_preflight_result"]
    report = {
        "round": "961_980",
        "feature_track": summary["feature_track"],
        "runner_preflight_summary": summary,
        "runner_preflight_schema_summary": summary["runner_preflight_schema_summary"],
        **{name: _compact(preflight) for name, preflight in preflights.items()},
        "future_request_packet_compatibility_proof": bool(summary["future_request_packet_compatibility_summary"]),
        "operator_decision_packet_compatibility_proof": bool(summary["operator_decision_packet_compatibility_summary"]),
        "execution_dryrun_trace_compatibility_proof": bool(summary["execution_dryrun_trace_compatibility_summary"]),
        "checkpoint_rollback_plan_compatibility_proof": bool(summary["checkpoint_rollback_plan_compatibility_summary"]),
        "dry_run_bridge_compatibility_proof": bool(summary["dryrun_bridge_compatibility_summary"]),
        "handoff_compatibility_proof": bool(summary["handoff_compatibility_summary"]),
        "builder_compatibility_proof": bool(summary["builder_compatibility_summary"]),
        "proposal_validator_compatibility_proof": bool(summary["proposal_validator_compatibility_summary"]),
        "emotion_transition_validator_compatibility_proof": bool(summary["emotion_transition_validator_compatibility_summary"]),
        "emotion_transition_gate_compatibility_proof": bool(summary["emotion_transition_gate_compatibility_summary"]),
        "dry_run_apply_plan_compatibility_proof": bool(summary["dryrun_apply_plan_compatibility_summary"]),
        "no_execution_now_proof": _proof(approve)["no_execution_now"],
        "no_future_simulation_executed_proof": _proof(approve)["no_future_simulation_executed"],
        "no_dryrun_apply_executed_proof": _proof(approve)["no_dryrun_apply_executed"],
        "no_checkpoint_created_proof": _proof(approve)["no_checkpoint_created"],
        "no_rollback_created_proof": _proof(approve)["no_rollback_created"],
        "no_audit_written_proof": _proof(approve)["no_audit_written"],
        "no_checkpoint_write_proof": _proof(approve)["no_checkpoint_write"],
        "no_rollback_write_proof": _proof(approve)["no_rollback_write"],
        "no_audit_write_proof": _proof(approve)["no_audit_write"],
        "no_dryrun_apply_permission_proof": _proof(approve)["no_dryrun_apply_permission"],
        "no_live_apply_proof": _proof(approve)["no_live_apply"],
        "no_runtime_mutation_proof": _proof(approve)["no_runtime_mutation"],
        "no_persistence_proof": _proof(approve)["no_persistence"],
        "no_memory_write_proof": _proof(approve)["no_memory_write"],
        "no_vector_content_read_load_proof": _proof(approve)["no_vector_content_read_load"],
        "no_artifact_creation_staging_proof": _proof(approve)["no_artifact_creation_staging"],
        "hardware_non_panic_proof": _proof(preflights["hardware_low_power_non_panic_runner_result"])["hardware_non_panic"],
        "anti_global_synchrony_proof": _proof(approve)["anti_global_synchrony"],
        "agp_fallback_non_bypass_proof": _proof(approve)["agp_fallback_non_bypass"],
        "exactly_one_next_implementation_recommendation": summary["exactly_one_next_implementation_recommendation"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
