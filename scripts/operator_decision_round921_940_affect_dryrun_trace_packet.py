#!/usr/bin/env python3
"""Emit compact Round921-940 affect dry-run trace operator decision packet JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_dryrun_trace_operator_decision_packet import (
    affect_dryrun_trace_operator_decision_packet_summary,
    build_affect_dryrun_trace_operator_decision_packet,
    validate_affect_dryrun_trace_operator_decision_packet,
)


def _compact(packet: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_affect_dryrun_trace_operator_decision_packet(packet)
    safety = packet.get("event_safety_result", {}) if isinstance(packet.get("event_safety_result"), Mapping) else {}
    return {
        "packet_passed": packet.get("packet_passed"),
        "packet_status": packet.get("packet_status"),
        "validation_passed": validation.get("validation_passed"),
        "event_category": packet.get("event_category"),
        "operator_decision": packet.get("operator_decision"),
        "decision_recorded": packet.get("decision_recorded"),
        "decision_packet_built": packet.get("decision_packet_built"),
        "trace_passed": packet.get("trace_passed"),
        "trace_accepted": packet.get("trace_accepted"),
        "approve_for_future_dryrun_simulation": packet.get("approve_for_future_dryrun_simulation"),
        "reject_requested": packet.get("reject_requested"),
        "hold_for_review_requested": packet.get("hold_for_review_requested"),
        "revision_requested": packet.get("revision_requested"),
        "blocked_reasons": packet.get("blocked_reasons"),
        "validation_blocked_reasons": validation.get("blocked_reasons"),
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
        "event_safety_result": safety,
    }


def _proof(packet: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_checkpoint_created": packet.get("checkpoint_created") is False,
        "no_rollback_created": packet.get("rollback_created") is False,
        "no_audit_written": packet.get("audit_written") is False,
        "no_checkpoint_write": packet.get("checkpoint_write_allowed") is False,
        "no_rollback_write": packet.get("rollback_write_allowed") is False,
        "no_audit_write": packet.get("audit_write_allowed") is False,
        "no_dryrun_apply_permission": packet.get("dryrun_apply_allowed") is False and packet.get("apply_permission_granted") is False,
        "no_live_apply": packet.get("live_apply_allowed") is False,
        "no_runtime_mutation": packet.get("runtime_mutation_performed") is False,
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
        "approve_for_future_dryrun_simulation_packet_result": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation"),
        "reject_packet_result": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "reject"),
        "hold_for_review_packet_result": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "hold_for_review"),
        "request_revision_packet_result": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "request_revision"),
        "invalid_decision_fail_closed_result": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "approve_live_apply"),
        "unknown_event_fail_closed_result": build_affect_dryrun_trace_operator_decision_packet("unknown_event", {}, "reject"),
        "trace_failure_blocks_approval_proof": build_affect_dryrun_trace_operator_decision_packet("praise", {"competence_drive": 0.02}, "approve_for_future_dryrun_simulation", {"dryrun_bridge_failure_simulated": True}),
        "malicious_comment_decision_packet_safety_result": build_affect_dryrun_trace_operator_decision_packet("malicious_comment", {"boundary_defense": 0.02}, "hold_for_review"),
        "identity_attack_decision_packet_safety_result": build_affect_dryrun_trace_operator_decision_packet("identity_attack", {"threat_pressure": 0.02}, "hold_for_review"),
        "hardware_normal_zero_delta_decision_result": build_affect_dryrun_trace_operator_decision_packet("hardware_normal", {}, "approve_for_future_dryrun_simulation"),
        "hardware_low_power_non_panic_decision_result": build_affect_dryrun_trace_operator_decision_packet("hardware_low_power", {"energy_budget": -0.02}, "hold_for_review"),
        "hardware_prediction_error_diagnostic_only_decision_result": build_affect_dryrun_trace_operator_decision_packet("hardware_prediction_error", {"stability_need": 0.02}, "hold_for_review"),
        "speech_pressure_agp_fallback_safe_decision_result": build_affect_dryrun_trace_operator_decision_packet("speech_output_pressure", {"expression_pressure": 0.02}, "hold_for_review"),
        "imagination_negative_spiral_boundary_preserving_decision_result": build_affect_dryrun_trace_operator_decision_packet("imagination_negative_spiral", {"recovery_need": 0.02}, "hold_for_review", common_imagination),
        "memory_self_update_quarantine_preserving_decision_result": build_affect_dryrun_trace_operator_decision_packet("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}, "hold_for_review"),
    }
    summary = affect_dryrun_trace_operator_decision_packet_summary()
    approve = packets["approve_for_future_dryrun_simulation_packet_result"]
    compatibility = summary["execution_trace_compatibility_summary"]
    report = {
        "round": "921_940",
        "feature_track": summary["feature_track"],
        "decision_packet_summary": summary,
        "decision_packet_schema_summary": summary["decision_packet_schema_summary"],
        **{name: _compact(packet) for name, packet in packets.items()},
        "execution_dryrun_trace_compatibility_proof": bool(compatibility),
        "checkpoint_rollback_plan_compatibility_proof": bool(compatibility.get("checkpoint_rollback_plan_compatibility_summary")),
        "dry_run_bridge_compatibility_proof": compatibility.get("dryrun_bridge_compatibility_summary", {}).get("dryrun_preflight_only") is True,
        "handoff_compatibility_proof": bool(compatibility.get("handoff_compatibility_summary")),
        "builder_compatibility_proof": bool(compatibility.get("builder_compatibility_summary")),
        "proposal_validator_compatibility_proof": bool(compatibility.get("proposal_validator_compatibility_summary")),
        "emotion_transition_validator_compatibility_proof": bool(compatibility.get("emotion_transition_validator_compatibility_summary")),
        "emotion_transition_gate_compatibility_proof": bool(compatibility.get("emotion_transition_gate_compatibility_summary")),
        "dry_run_apply_plan_compatibility_proof": bool(compatibility.get("dryrun_apply_plan_compatibility_summary")),
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
        "hardware_non_panic_proof": _proof(packets["hardware_low_power_non_panic_decision_result"])["hardware_non_panic"],
        "anti_global_synchrony_proof": _proof(approve)["anti_global_synchrony"],
        "agp_fallback_non_bypass_proof": _proof(approve)["agp_fallback_non_bypass"],
        "exactly_one_next_implementation_recommendation": "Round941 may add an operator-reviewed dry-run simulation request intake schema; keep dry-run apply, live apply, checkpoint creation, rollback creation, audit writes, memory, persistence, runtime mapping, enforcement, AGP/fallback bypass, and vector reads disabled until a separate explicit authorization round.",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
