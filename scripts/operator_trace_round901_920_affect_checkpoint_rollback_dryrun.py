#!/usr/bin/env python3
"""Emit compact Round901-920 affect checkpoint/rollback dry-run trace JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_transition_checkpoint_rollback_dryrun_trace import (
    affect_checkpoint_rollback_execution_dryrun_trace_summary,
    build_affect_checkpoint_rollback_execution_dryrun_trace,
    validate_affect_checkpoint_rollback_execution_dryrun_trace,
)


def _compact(trace: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_affect_checkpoint_rollback_execution_dryrun_trace(trace)
    return {
        "trace_passed": trace.get("trace_passed"),
        "trace_status": trace.get("trace_status"),
        "validation_passed": validation.get("validation_passed"),
        "validation_status": validation.get("validation_status"),
        "event_category": trace.get("event_category"),
        "checkpoint_plan_passed": trace.get("checkpoint_plan_passed"),
        "rollback_plan_passed": trace.get("rollback_plan_passed"),
        "execution_trace_built": trace.get("execution_trace_built"),
        "checkpoint_capture_simulated": trace.get("checkpoint_capture_simulated"),
        "rollback_restore_simulated": trace.get("rollback_restore_simulated"),
        "checkpoint_created": trace.get("checkpoint_created"),
        "rollback_created": trace.get("rollback_created"),
        "audit_written": trace.get("audit_written"),
        "checkpoint_write_allowed": trace.get("checkpoint_write_allowed"),
        "rollback_write_allowed": trace.get("rollback_write_allowed"),
        "audit_write_allowed": trace.get("audit_write_allowed"),
        "dryrun_apply_allowed": trace.get("dryrun_apply_allowed"),
        "live_apply_allowed": trace.get("live_apply_allowed"),
        "apply_permission_granted": trace.get("apply_permission_granted"),
        "state_mutation_performed": trace.get("state_mutation_performed"),
        "runtime_mutation_performed": trace.get("runtime_mutation_performed"),
        "memory_write_performed": trace.get("memory_write_performed"),
        "persistence_write_performed": trace.get("persistence_write_performed"),
        "vector_read_performed": trace.get("vector_read_performed"),
        "vector_load_performed": trace.get("vector_load_performed"),
        "artifact_created_or_staged": trace.get("artifact_created_or_staged"),
        "agp_bypass_allowed": trace.get("agp_bypass_allowed"),
        "fallback_bypass_allowed": trace.get("fallback_bypass_allowed"),
        "global_synchrony_blocked": trace.get("global_synchrony_blocked"),
        "runtime_mapping_enabled_default": trace.get("runtime_mapping_enabled_default"),
        "enforcement_enabled_default": trace.get("enforcement_enabled_default"),
        "production_persistence_enabled": trace.get("production_persistence_enabled"),
        "vector_contents_read": trace.get("vector_contents_read"),
        "vectors_loaded": trace.get("vectors_loaded"),
        "blocked_reasons": trace.get("blocked_reasons"),
        "validation_blocked_reasons": validation.get("blocked_reasons"),
        "guard_preservation_result": trace.get("guard_preservation_result"),
        "preflight_validation_sequence_count": len(trace.get("preflight_validation_sequence", ())),
        "checkpoint_capture_sequence_count": len(trace.get("checkpoint_capture_sequence", ())),
        "bounded_apply_candidate_sequence_count": len(trace.get("bounded_apply_candidate_sequence", ())),
        "rollback_restore_sequence_count": len(trace.get("rollback_restore_sequence", ())),
        "audit_event_sequence_count": len(trace.get("audit_event_sequence", ())),
        "failure_handling_sequence_count": len(trace.get("failure_handling_sequence", ())),
        "guard_preservation_sequence_count": len(trace.get("guard_preservation_sequence", ())),
    }


def _proof(trace: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "no_checkpoint_created": trace.get("checkpoint_created") is False,
        "no_rollback_created": trace.get("rollback_created") is False,
        "no_audit_written": trace.get("audit_written") is False,
        "no_checkpoint_write": trace.get("checkpoint_write_allowed") is False,
        "no_rollback_write": trace.get("rollback_write_allowed") is False,
        "no_audit_write": trace.get("audit_write_allowed") is False,
        "no_apply_permission": trace.get("apply_permission_granted") is False and trace.get("dryrun_apply_allowed") is False,
        "no_live_apply": trace.get("live_apply_allowed") is False,
        "no_runtime_mutation": trace.get("state_mutation_performed") is False and trace.get("runtime_mutation_performed") is False,
        "no_persistence": trace.get("persistence_write_performed") is False and trace.get("production_persistence_enabled") is False,
        "no_memory_write": trace.get("memory_write_performed") is False,
        "no_vector_content_read_load": trace.get("vector_read_performed") is False and trace.get("vector_load_performed") is False and trace.get("vector_contents_read") is False and trace.get("vectors_loaded") is False,
        "no_artifact_creation_staging": trace.get("artifact_created_or_staged") is False,
        "hardware_non_panic": trace.get("guard_preservation_result", {}).get("hardware_non_panic_operational_only") is True,
        "anti_global_synchrony": trace.get("global_synchrony_blocked") is True,
        "agp_fallback_non_bypass": trace.get("agp_bypass_allowed") is False and trace.get("fallback_bypass_allowed") is False,
    }


def main() -> None:
    common_imagination = {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True}
    traces = {
        "valid_praise_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("praise", {"competence_drive": 0.02}),
        "valid_useful_criticism_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("useful_criticism", {"learning_pressure": 0.02}),
        "malicious_comment_trace_safety_result": build_affect_checkpoint_rollback_execution_dryrun_trace("malicious_comment", {"boundary_defense": 0.02}),
        "identity_attack_trace_safety_result": build_affect_checkpoint_rollback_execution_dryrun_trace("identity_attack", {"threat_pressure": 0.02}),
        "hardware_normal_zero_delta_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_normal", {}),
        "hardware_low_power_non_panic_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_low_power", {"energy_budget": -0.02}),
        "hardware_prediction_error_diagnostic_only_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("hardware_prediction_error", {"stability_need": 0.02}),
        "unknown_event_fail_closed_result": build_affect_checkpoint_rollback_execution_dryrun_trace("unknown_event", {}),
        "speech_pressure_agp_fallback_safe_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("speech_output_pressure", {"expression_pressure": 0.02}),
        "imagination_negative_spiral_boundary_preserving_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("imagination_negative_spiral", {"recovery_need": 0.02}, common_imagination),
        "memory_self_update_quarantine_preserving_trace_result": build_affect_checkpoint_rollback_execution_dryrun_trace("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02}),
    }
    summary = affect_checkpoint_rollback_execution_dryrun_trace_summary()
    praise = traces["valid_praise_trace_result"]
    report = {
        "round": "901_920",
        "feature_track": summary["feature_track"],
        "execution_dryrun_trace_summary": summary,
        "trace_schema_summary": summary["trace_schema_summary"],
        "preflight_validation_sequence_summary": summary["preflight_validation_sequence_summary"],
        "checkpoint_capture_sequence_summary": summary["checkpoint_capture_sequence_summary"],
        "bounded_apply_candidate_sequence_summary": summary["bounded_apply_candidate_sequence_summary"],
        "rollback_restore_sequence_summary": summary["rollback_restore_sequence_summary"],
        "audit_event_sequence_summary": summary["audit_event_sequence_summary"],
        "failure_handling_sequence_summary": summary["failure_handling_sequence_summary"],
        "guard_preservation_sequence_summary": summary["guard_preservation_sequence_summary"],
        **{name: _compact(trace) for name, trace in traces.items()},
        "checkpoint_rollback_plan_compatibility_proof": bool(summary["checkpoint_rollback_plan_compatibility_summary"]),
        "dry_run_bridge_compatibility_proof": summary["dryrun_bridge_compatibility_summary"].get("dryrun_preflight_only") is True,
        "handoff_compatibility_proof": bool(summary["handoff_compatibility_summary"]),
        "builder_compatibility_proof": bool(summary["builder_compatibility_summary"]),
        "proposal_validator_compatibility_proof": bool(summary["proposal_validator_compatibility_summary"]),
        "emotion_transition_validator_compatibility_proof": bool(summary["emotion_transition_validator_compatibility_summary"]),
        "emotion_transition_gate_compatibility_proof": bool(summary["emotion_transition_gate_compatibility_summary"]),
        "dry_run_apply_plan_compatibility_proof": bool(summary["dryrun_apply_plan_compatibility_summary"]),
        "no_checkpoint_created_proof": _proof(praise)["no_checkpoint_created"],
        "no_rollback_created_proof": _proof(praise)["no_rollback_created"],
        "no_audit_written_proof": _proof(praise)["no_audit_written"],
        "no_checkpoint_write_proof": _proof(praise)["no_checkpoint_write"],
        "no_rollback_write_proof": _proof(praise)["no_rollback_write"],
        "no_audit_write_proof": _proof(praise)["no_audit_write"],
        "no_apply_permission_proof": _proof(praise)["no_apply_permission"],
        "no_live_apply_proof": _proof(praise)["no_live_apply"],
        "no_runtime_mutation_proof": _proof(praise)["no_runtime_mutation"],
        "no_persistence_proof": _proof(praise)["no_persistence"],
        "no_memory_write_proof": _proof(praise)["no_memory_write"],
        "no_vector_content_read_load_proof": _proof(praise)["no_vector_content_read_load"],
        "no_artifact_creation_staging_proof": _proof(praise)["no_artifact_creation_staging"],
        "hardware_non_panic_proof": _proof(traces["hardware_low_power_non_panic_trace_result"])["hardware_non_panic"],
        "anti_global_synchrony_proof": _proof(praise)["anti_global_synchrony"],
        "agp_fallback_non_bypass_proof": _proof(praise)["agp_fallback_non_bypass"],
        "exactly_one_next_implementation_recommendation": "Round921 may add operator-reviewed in-memory checkpoint object schema preview only; keep creation, rollback records, audit writes, emotion/hormone apply, memory, persistence, and vector reads disabled until a separate explicit authorization round.",
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
