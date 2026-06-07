#!/usr/bin/env python3
"""Emit compact Round941-960 affect future dry-run simulation request JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_dryrun_trace_operator_decision_packet import build_affect_dryrun_trace_operator_decision_packet
from adapters.affect_future_dryrun_simulation_request_packet import (
    affect_future_dryrun_simulation_request_packet_summary,
    build_affect_future_dryrun_simulation_request_from_decision_packet,
    build_affect_future_dryrun_simulation_request_packet,
    validate_affect_future_dryrun_simulation_request_packet,
)


def _compact(packet: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_affect_future_dryrun_simulation_request_packet(packet)
    safety = packet.get("event_safety_result", {}) if isinstance(packet.get("event_safety_result"), Mapping) else {}
    return {
        "request_packet_passed": packet.get("request_packet_passed"),
        "request_packet_status": packet.get("request_packet_status"),
        "validation_passed": validation.get("validation_passed"),
        "event_category": packet.get("event_category"),
        "source_operator_decision": packet.get("source_operator_decision"),
        "source_decision_packet_passed": packet.get("source_decision_packet_passed"),
        "source_trace_passed": packet.get("source_trace_passed"),
        "future_dryrun_simulation_request_recorded": packet.get("future_dryrun_simulation_request_recorded"),
        "future_dryrun_simulation_request_allowed": packet.get("future_dryrun_simulation_request_allowed"),
        "requested_now": packet.get("requested_now"),
        "execution_allowed_now": packet.get("execution_allowed_now"),
        "dryrun_apply_executed": packet.get("dryrun_apply_executed"),
        "dryrun_apply_allowed": packet.get("dryrun_apply_allowed"),
        "live_apply_allowed": packet.get("live_apply_allowed"),
        "apply_permission_granted": packet.get("apply_permission_granted"),
        "checkpoint_created": packet.get("checkpoint_created"),
        "rollback_created": packet.get("rollback_created"),
        "audit_written": packet.get("audit_written"),
        "checkpoint_write_allowed": packet.get("checkpoint_write_allowed"),
        "rollback_write_allowed": packet.get("rollback_write_allowed"),
        "audit_write_allowed": packet.get("audit_write_allowed"),
        "runtime_mutation_performed": packet.get("runtime_mutation_performed"),
        "persistence_write_performed": packet.get("persistence_write_performed"),
        "memory_write_performed": packet.get("memory_write_performed"),
        "vector_content_read_load": packet.get("vector_contents_read") is True or packet.get("vectors_loaded") is True or packet.get("vector_read_performed") is True or packet.get("vector_load_performed") is True,
        "artifact_created_or_staged": packet.get("artifact_created_or_staged"),
        "global_synchrony_blocked": packet.get("global_synchrony_blocked"),
        "agp_fallback_non_bypass": safety.get("agp_fallback_non_bypass"),
        "blocked_reasons": packet.get("blocked_reasons"),
        "validation_blocked_reasons": validation.get("blocked_reasons"),
        "event_safety_result": safety,
    }


def _proof(packet: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_execution_now": packet.get("requested_now") is False and packet.get("execution_allowed_now") is False and packet.get("future_dryrun_simulation_requested_now") is False,
        "no_checkpoint_created": packet.get("checkpoint_created") is False,
        "no_rollback_created": packet.get("rollback_created") is False,
        "no_audit_written": packet.get("audit_written") is False,
        "no_checkpoint_write": packet.get("checkpoint_write_allowed") is False,
        "no_rollback_write": packet.get("rollback_write_allowed") is False,
        "no_audit_write": packet.get("audit_write_allowed") is False,
        "no_dryrun_apply_permission": packet.get("dryrun_apply_allowed") is False and packet.get("dryrun_apply_executed") is False and packet.get("apply_permission_granted") is False,
        "no_live_apply": packet.get("live_apply_allowed") is False,
        "no_runtime_mutation": packet.get("runtime_mutation_performed") is False and packet.get("state_mutation_performed") is False,
        "no_persistence": packet.get("persistence_write_performed") is False and packet.get("production_persistence_enabled") is False,
        "no_memory_write": packet.get("memory_write_performed") is False,
        "no_vector_content_read_load": packet.get("vector_read_performed") is False and packet.get("vector_load_performed") is False and packet.get("vector_contents_read") is False and packet.get("vectors_loaded") is False,
        "no_artifact_creation_staging": packet.get("artifact_created_or_staged") is False,
        "hardware_non_panic": packet.get("event_safety_result", {}).get("hardware_low_power_and_below_non_panic_operational_only") is True,
        "anti_global_synchrony": packet.get("global_synchrony_blocked") is True and packet.get("event_safety_result", {}).get("global_synchrony_blocked") is True,
        "agp_fallback_non_bypass": packet.get("agp_bypass_allowed") is False and packet.get("fallback_bypass_allowed") is False and packet.get("event_safety_result", {}).get("agp_fallback_non_bypass") is True,
    }


def main() -> None:
    common_imagination = {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True}
    packets = {
        "approve_for_future_dryrun_simulation_request_packet_result": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation"),
        "reject_no_request_result": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "reject"),
        "hold_for_review_no_request_result": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "hold_for_review"),
        "request_revision_no_request_result": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "request_revision"),
        "invalid_decision_fail_closed_result": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_live_apply"),
        "unknown_event_fail_closed_result": build_affect_future_dryrun_simulation_request_packet("unknown_event", {}, "approve_for_future_dryrun_simulation"),
        "source_trace_failure_blocks_request_proof": build_affect_future_dryrun_simulation_request_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation", {"dryrun_bridge_failure_simulated": True}),
        "malicious_comment_request_packet_safety_result": build_affect_future_dryrun_simulation_request_packet("malicious_comment", {"boundary_defense": 0.02}, "approve_for_future_dryrun_simulation"),
        "identity_attack_request_packet_safety_result": build_affect_future_dryrun_simulation_request_packet("identity_attack", {"threat_pressure": 0.02}, "approve_for_future_dryrun_simulation"),
        "hardware_normal_zero_delta_request_result": build_affect_future_dryrun_simulation_request_packet("hardware_normal", {}, "approve_for_future_dryrun_simulation"),
        "hardware_low_power_non_panic_request_result": build_affect_future_dryrun_simulation_request_packet("hardware_low_power", {"energy_budget": -0.02}, "approve_for_future_dryrun_simulation"),
        "hardware_prediction_error_diagnostic_only_request_result": build_affect_future_dryrun_simulation_request_packet("hardware_prediction_error", {"stability_need": 0.02}, "approve_for_future_dryrun_simulation"),
        "speech_pressure_agp_fallback_safe_request_result": build_affect_future_dryrun_simulation_request_packet("speech_output_pressure", {"expression_pressure": 0.02}, "approve_for_future_dryrun_simulation"),
        "imagination_negative_spiral_boundary_preserving_request_result": build_affect_future_dryrun_simulation_request_packet("imagination_negative_spiral", {"recovery_need": 0.02}, "approve_for_future_dryrun_simulation", common_imagination),
        "memory_self_update_quarantine_preserving_request_result": build_affect_future_dryrun_simulation_request_packet("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}, "approve_for_future_dryrun_simulation"),
    }
    failed_source = build_affect_dryrun_trace_operator_decision_packet("hardware_low_power", {"social_trust": 0.01}, "approve_for_future_dryrun_simulation")
    packets["source_decision_failure_blocks_request_proof"] = build_affect_future_dryrun_simulation_request_from_decision_packet(failed_source)

    summary = affect_future_dryrun_simulation_request_packet_summary()
    approve = packets["approve_for_future_dryrun_simulation_request_packet_result"]
    report = {
        "round": "941_960",
        "feature_track": summary["feature_track"],
        "request_packet_summary": summary,
        "request_packet_schema_summary": summary["request_packet_schema_summary"],
        **{name: _compact(packet) for name, packet in packets.items()},
        "operator_decision_packet_compatibility_proof": bool(summary["decision_packet_compatibility_summary"]),
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
        "hardware_non_panic_proof": _proof(packets["hardware_low_power_non_panic_request_result"])["hardware_non_panic"],
        "anti_global_synchrony_proof": _proof(approve)["anti_global_synchrony"],
        "agp_fallback_non_bypass_proof": _proof(approve)["agp_fallback_non_bypass"],
        "exactly_one_next_implementation_recommendation": summary["exactly_one_next_implementation_recommendation"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
