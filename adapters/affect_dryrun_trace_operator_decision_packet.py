"""Round921-940 read-only operator decision packet for affect dry-run traces.

This module represents an operator decision over an existing affect
checkpoint/rollback execution dry-run trace as detached data only.  It never
creates checkpoints, rollback records, audit files, operator artifacts, vectors,
or persistence records; never grants dry-run/live apply permission; never
applies emotion or hormone transitions; and never mutates runtime, memory,
AGP, fallback, or vector state.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MEMORY_SELF_UPDATE_EVENTS,
    OPERATIONAL_HARDWARE_AXIS_SET,
    REQUIRED_EVENT_CATEGORIES,
    SOCIAL_SELF_IDENTITY_AXIS_SET,
    SPEECH_PRESSURE_EVENTS,
)
from adapters.affect_transition_checkpoint_rollback_dryrun_trace import (
    AUDIT_EVENT_SEQUENCE,
    CHECKPOINT_CAPTURE_SEQUENCE,
    DETERMINISTIC_SEQUENCE_ID_POLICY,
    FAILURE_HANDLING_SEQUENCE,
    GUARD_PRESERVATION_SEQUENCE,
    INTEGRITY_HASH_POLICY,
    NO_APPLY_GUARANTEE,
    NO_WRITE_GUARANTEE,
    PREFLIGHT_VALIDATION_SEQUENCE,
    ROLLBACK_RESTORE_SEQUENCE,
    affect_checkpoint_rollback_execution_dryrun_trace_summary,
    build_affect_checkpoint_rollback_execution_dryrun_trace,
    validate_affect_checkpoint_rollback_execution_dryrun_trace,
)

ROUND921_940_VERSION = "v3_round921_940_affect_dryrun_trace_operator_decision_packet"
FEATURE_TRACK = "read_only_operator_decision_packet_for_affect_dryrun_trace"

SUPPORTED_OPERATOR_DECISIONS: tuple[str, ...] = (
    "approve_for_future_dryrun_simulation",
    "reject",
    "hold_for_review",
    "request_revision",
)

_FALSE_PACKET_FLAGS: dict[str, bool] = {
    "dryrun_apply_allowed": False,
    "live_apply_allowed": False,
    "apply_permission_granted": False,
    "checkpoint_created": False,
    "rollback_created": False,
    "audit_written": False,
    "checkpoint_write_allowed": False,
    "rollback_write_allowed": False,
    "audit_write_allowed": False,
    "state_mutation_performed": False,
    "runtime_mutation_performed": False,
    "memory_write_performed": False,
    "persistence_write_performed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "artifact_created_or_staged": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
    "runtime_mapping_enabled_default": False,
    "enforcement_enabled_default": False,
    "production_persistence_enabled": False,
    "vector_contents_read": False,
    "vectors_loaded": False,
}

EXPLICIT_NON_PERMISSIONS: tuple[str, ...] = (
    "approve_for_future_dryrun_simulation_is_review_data_only",
    "no_dryrun_apply_permission",
    "no_live_apply_permission",
    "no_emotion_transition_apply",
    "no_hormone_transition_apply",
    "no_checkpoint_creation",
    "no_rollback_creation",
    "no_audit_write",
    "no_runtime_mutation",
    "no_persistence_enablement",
    "no_runtime_mapping_enablement",
    "no_enforcement_enablement",
    "no_memory_or_self_model_write",
    "no_vector_content_read_or_load",
    "no_operator_artifact_creation_or_staging",
    "no_agp_bypass",
    "no_fallback_bypass",
    "no_autonomous_runtime_scheduling",
)

REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY: tuple[str, ...] = (
    "separate_explicit_operator_authorization_round",
    "fresh_checkpoint_and_rollback_design_review",
    "fresh_emotion_transition_validator_and_gate_review",
    "fresh_dryrun_apply_plan_review",
    "focused_and_full_test_suite_before_apply",
    "artifact_safety_check_confirming_no_operator_artifacts_or_vectors_staged",
)

APPROVAL_SCOPE = (
    "operator decision may approve only a future request to simulate another dry-run trace; "
    "it grants no dry-run apply, live apply, checkpoint, rollback, audit, memory, persistence, runtime, AGP, fallback, or vector permission"
)

DECISION_REASON_DEFAULTS: dict[str, str] = {
    "approve_for_future_dryrun_simulation": "operator may later request another read-only dry-run simulation after followup review",
    "reject": "operator rejects the packet as data only; no mutation or apply path is opened",
    "hold_for_review": "operator holds the packet for additional review; no mutation or apply path is opened",
    "request_revision": "operator requests a revised packet; no mutation or apply path is opened",
}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _deltas(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _summary_sequence(names: tuple[str, ...]) -> dict[str, Any]:
    return {"count": len(names), "first": names[0], "last": names[-1], "steps": tuple(names)}


def _decision_flags(operator_decision: str) -> dict[str, bool]:
    return {
        "approve_for_future_dryrun_simulation": operator_decision == "approve_for_future_dryrun_simulation",
        "reject_requested": operator_decision == "reject",
        "hold_for_review_requested": operator_decision == "hold_for_review",
        "revision_requested": operator_decision == "request_revision",
    }


def _schema_summary() -> dict[str, Any]:
    return {
        "packet_version": ROUND921_940_VERSION,
        "required_fields": (
            "packet_version",
            "event_category",
            "operator_decision",
            "decision_reason",
            "dryrun_trace_summary",
            "preflight_validation_sequence_summary",
            "checkpoint_capture_sequence_summary",
            "bounded_apply_candidate_sequence_summary",
            "rollback_restore_sequence_summary",
            "audit_event_sequence_summary",
            "failure_handling_sequence_summary",
            "guard_preservation_sequence_summary",
            "approval_scope",
            "explicit_non_permissions",
            "required_followup_before_any_apply",
            "deterministic_sequence_id_policy",
            "integrity_hash_policy",
            "no_write_guarantee",
            "no_apply_guarantee",
            "notes",
        ),
        "supported_operator_decisions": SUPPORTED_OPERATOR_DECISIONS,
        "approval_scope": APPROVAL_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE,
        "no_apply_guarantee": NO_APPLY_GUARANTEE,
    }


def _event_safety_result(event_category: str, deltas: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, bool]:
    guard = trace.get("guard_preservation_result", {}) if isinstance(trace.get("guard_preservation_result"), Mapping) else {}
    delta_axes = set(str(axis) for axis in deltas)
    hostile = event_category in HOSTILE_SOCIAL_EVENTS
    hardware = event_category in HARDWARE_EVENTS
    return {
        "hostile_social_preserves_quarantine_appraisal_gate": (not hostile) or guard.get("hostile_social_preserves_quarantine_appraisal_gate") is True,
        "hostile_social_blocks_identity_self_memory_update": (not hostile) or guard.get("hostile_social_blocks_identity_self_memory_update") is True,
        "useful_criticism_preserves_appraisal_before_memory_self_model": event_category != "useful_criticism" or guard.get("useful_criticism_preserves_appraisal_before_memory_self_model") is True,
        "hardware_normal_zero_delta": event_category != "hardware_normal" or not delta_axes,
        "hardware_events_do_not_target_social_self_identity_axes": (not hardware) or delta_axes.isdisjoint(SOCIAL_SELF_IDENTITY_AXIS_SET),
        "hardware_low_power_and_below_non_panic_operational_only": event_category not in {"hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"} or (delta_axes.issubset(OPERATIONAL_HARDWARE_AXIS_SET) and guard.get("hardware_non_panic_operational_only") is True),
        "hardware_prediction_error_diagnostic_operational_only": event_category != "hardware_prediction_error" or guard.get("hardware_prediction_error_diagnostic_operational_only") is True,
        "hardware_polling_tick_no_recursive_concern_loop": event_category != "hardware_polling_tick" or guard.get("hardware_polling_tick_no_recursive_concern_loop") is True,
        "speech_pressure_preserves_agp_fallback_gates": event_category not in SPEECH_PRESSURE_EVENTS or guard.get("speech_pressure_preserves_agp_fallback_gates") is True,
        "listening_uncertainty_does_not_relabel_neutral_as_hostile": event_category != "listening_uncertainty" or guard.get("listening_uncertainty_does_not_relabel_neutral_as_hostile") is True,
        "imagination_negative_spiral_preserves_budget_cooldown_reality_check": event_category != "imagination_negative_spiral" or guard.get("imagination_negative_spiral_preserves_budget_cooldown_reality_check") is True,
        "memory_self_update_preserves_appraisal_quarantine": event_category not in MEMORY_SELF_UPDATE_EVENTS or guard.get("memory_self_update_preserves_appraisal_quarantine") is True,
        "one_event_cannot_build_all_axis_decision_packets": len(delta_axes) < len(ALL_AXIS_GROUPS),
        "global_synchrony_blocked": trace.get("global_synchrony_blocked") is True,
        "agp_fallback_non_bypass": trace.get("agp_bypass_allowed") is False and trace.get("fallback_bypass_allowed") is False,
    }


def build_affect_dryrun_trace_operator_decision_packet(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    operator_decision: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached operator decision packet over a dry-run trace."""

    meta = _copy_mapping(metadata)
    deltas = _deltas(proposed_axis_deltas)
    trace = build_affect_checkpoint_rollback_execution_dryrun_trace(event_category, deltas, meta)
    trace_validation = validate_affect_checkpoint_rollback_execution_dryrun_trace(trace)
    trace_passed = trace.get("trace_passed") is True and trace_validation.get("validation_passed") is True
    valid_decision = operator_decision in SUPPORTED_OPERATOR_DECISIONS
    event_known = event_category in REQUIRED_EVENT_CATEGORIES
    approval_requested = operator_decision == "approve_for_future_dryrun_simulation"

    blocked_reasons: list[str] = []
    blocked_reasons.extend(str(reason) for reason in trace.get("blocked_reasons", ()))
    blocked_reasons.extend(str(reason) for reason in trace_validation.get("blocked_reasons", ()))
    if not event_known:
        blocked_reasons.append("unknown_event_category_fail_closed")
    if not valid_decision:
        blocked_reasons.append("invalid_operator_decision_fail_closed")
    if approval_requested and not trace_passed:
        blocked_reasons.append("trace_failure_blocks_approve_for_future_dryrun_simulation")

    safety_result = _event_safety_result(event_category, deltas, trace)
    for key, preserved in safety_result.items():
        if preserved is not True:
            blocked_reasons.append(f"{key}_must_be_preserved")

    decision_recorded = valid_decision and event_known
    decision_packet_built = decision_recorded
    packet_passed = decision_packet_built and trace_passed and not blocked_reasons
    packet_status = "valid_read_only_operator_decision_packet_no_apply_no_write" if packet_passed else "blocked_fail_closed_operator_decision_packet"

    decision_reason = str(meta.get("decision_reason") or DECISION_REASON_DEFAULTS.get(operator_decision, "invalid operator decision rejected fail-closed"))
    warnings = sorted(
        set(str(warning) for warning in trace.get("warnings", ()))
        | {
            "decision_recorded_does_not_write_audit",
            "decision_packet_built_does_not_create_checkpoint",
            "decision_packet_built_does_not_create_rollback",
            "approve_for_future_dryrun_simulation_does_not_grant_dryrun_apply",
            "approve_for_future_dryrun_simulation_does_not_grant_live_apply",
            "all_decisions_remain_data_only_no_mutation",
        }
    )

    return {
        "packet_version": ROUND921_940_VERSION,
        "feature_track": FEATURE_TRACK,
        "packet_passed": packet_passed,
        "packet_status": packet_status,
        "event_category": event_category,
        "operator_decision": operator_decision,
        "decision_reason": decision_reason,
        "decision_recorded": decision_recorded,
        "decision_packet_built": decision_packet_built,
        "trace_passed": trace_passed,
        "trace_accepted": trace_passed,
        **_decision_flags(operator_decision),
        **_FALSE_PACKET_FLAGS,
        "global_synchrony_blocked": True,
        "approval_scope": APPROVAL_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE,
        "no_apply_guarantee": NO_APPLY_GUARANTEE,
        "dryrun_trace_summary": {
            "trace_version": trace.get("trace_version"),
            "trace_status": trace.get("trace_status"),
            "trace_passed": trace_passed,
            "trace_validation_status": trace_validation.get("validation_status"),
            "checkpoint_plan_passed": trace.get("checkpoint_plan_passed"),
            "rollback_plan_passed": trace.get("rollback_plan_passed"),
        },
        "preflight_validation_sequence_summary": _summary_sequence(PREFLIGHT_VALIDATION_SEQUENCE),
        "checkpoint_capture_sequence_summary": _summary_sequence(CHECKPOINT_CAPTURE_SEQUENCE),
        "bounded_apply_candidate_sequence_summary": {
            "count": len(trace.get("bounded_apply_candidate_sequence", ())),
            "candidate_apply_performed_any": any(row.get("candidate_apply_performed") is True for row in trace.get("bounded_apply_candidate_sequence", ())),
            "state_mutation_performed_any": any(row.get("state_mutation_performed") is True for row in trace.get("bounded_apply_candidate_sequence", ())),
            "axes": tuple(row.get("axis") for row in trace.get("bounded_apply_candidate_sequence", ())),
        },
        "rollback_restore_sequence_summary": _summary_sequence(ROLLBACK_RESTORE_SEQUENCE),
        "audit_event_sequence_summary": _summary_sequence(AUDIT_EVENT_SEQUENCE),
        "failure_handling_sequence_summary": _summary_sequence(FAILURE_HANDLING_SEQUENCE),
        "guard_preservation_sequence_summary": _summary_sequence(GUARD_PRESERVATION_SEQUENCE),
        "event_safety_result": safety_result,
        "trace_validation": trace_validation,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "warnings": warnings,
        "notes": (
            "operator_decision_packet_is_data_only",
            "decision_success_is_not_apply_authorization",
            "no_checkpoint_rollback_audit_or_artifact_created",
            "no_live_runtime_state_mutated",
        ),
    }


def validate_affect_dryrun_trace_operator_decision_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a decision packet preserves read-only/non-permission guards."""

    p = _copy_mapping(packet)
    reasons: list[str] = []
    if p.get("packet_version") != ROUND921_940_VERSION:
        reasons.append("packet_version_mismatch")
    if p.get("event_category") not in REQUIRED_EVENT_CATEGORIES:
        reasons.append("unknown_event_category_fail_closed")
    if p.get("operator_decision") not in SUPPORTED_OPERATOR_DECISIONS:
        reasons.append("invalid_operator_decision_fail_closed")
    if p.get("operator_decision") == "approve_for_future_dryrun_simulation" and p.get("trace_passed") is not True:
        reasons.append("trace_failure_blocks_approve_for_future_dryrun_simulation")
    for key, expected in _FALSE_PACKET_FLAGS.items():
        if p.get(key) is not expected:
            reasons.append(f"{key}_must_remain_false")
    if p.get("global_synchrony_blocked") is not True:
        reasons.append("global_synchrony_must_remain_blocked")
    if p.get("decision_recorded") is True and p.get("audit_written") is not False:
        reasons.append("decision_recorded_must_not_imply_audit_write")
    if p.get("decision_packet_built") is True and p.get("checkpoint_created") is not False:
        reasons.append("decision_packet_built_must_not_imply_checkpoint_creation")
    if p.get("decision_packet_built") is True and p.get("rollback_created") is not False:
        reasons.append("decision_packet_built_must_not_imply_rollback_creation")
    for key in (
        "approval_scope",
        "explicit_non_permissions",
        "required_followup_before_any_apply",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
        "no_write_guarantee",
        "no_apply_guarantee",
    ):
        if not p.get(key):
            reasons.append(f"{key}_required")
    if set(p.get("explicit_non_permissions", ())) != set(EXPLICIT_NON_PERMISSIONS):
        reasons.append("explicit_non_permissions_must_match_round921_940_scope")
    safety_result = p.get("event_safety_result", {}) if isinstance(p.get("event_safety_result"), Mapping) else {}
    for key, value in safety_result.items():
        if value is not True:
            reasons.append(f"{key}_must_be_preserved")
    if p.get("bounded_apply_candidate_sequence_summary", {}).get("candidate_apply_performed_any") is True:
        reasons.append("bounded_apply_candidates_must_not_apply")
    if p.get("bounded_apply_candidate_sequence_summary", {}).get("state_mutation_performed_any") is True:
        reasons.append("bounded_apply_candidates_must_not_mutate_state")

    passed = not reasons and p.get("packet_passed") is True
    return {
        "packet_version": ROUND921_940_VERSION,
        "validation_passed": passed,
        "validation_status": "valid_read_only_operator_decision_packet" if passed else "blocked_invalid_operator_decision_packet",
        "event_category": p.get("event_category"),
        "operator_decision": p.get("operator_decision"),
        "blocked_reasons": sorted(set(reasons)),
        "warnings": ["validation_success_does_not_grant_apply_write_checkpoint_rollback_audit_memory_persistence_runtime_or_vector_permission"],
        **_FALSE_PACKET_FLAGS,
    }


def affect_dryrun_trace_operator_decision_packet_summary() -> dict[str, Any]:
    """Return compact operator-facing summary for the decision packet surface."""

    trace_summary = affect_checkpoint_rollback_execution_dryrun_trace_summary()
    return {
        "packet_version": ROUND921_940_VERSION,
        "feature_track": FEATURE_TRACK,
        "scope": "operator_decision_packet_review_surface_only_no_apply_no_write_no_mutation",
        "decision_packet_schema_summary": _schema_summary(),
        "supported_decisions": SUPPORTED_OPERATOR_DECISIONS,
        "decision_safety_rules": {
            "unknown_event_category_fails_closed": True,
            "invalid_operator_decision_fails_closed": True,
            "trace_failure_blocks_approve_for_future_dryrun_simulation": True,
            "approval_only_allows_future_dryrun_simulation_request_later": True,
            "approval_does_not_grant_dryrun_apply_permission": True,
            "approval_does_not_grant_live_apply_permission": True,
            "reject_hold_and_revision_grant_no_apply_permission": True,
            "decision_recorded_does_not_imply_audit_write": True,
            "decision_packet_built_does_not_imply_checkpoint_creation": True,
            "decision_packet_built_does_not_imply_rollback_creation": True,
            "no_runtime_persistence_memory_vector_agp_or_fallback_bypass_request_allowed": True,
            "hostile_social_preserves_quarantine_appraisal_gate": True,
            "useful_criticism_preserves_appraisal_before_memory_self_model": True,
            "hardware_normal_zero_delta": True,
            "hardware_low_power_and_below_non_panic_operational_only": True,
            "hardware_prediction_error_diagnostic_operational_only": True,
            "hardware_polling_tick_no_recursive_concern_loop": True,
            "speech_pressure_preserves_agp_fallback_gates": True,
            "listening_uncertainty_does_not_relabel_neutral_as_hostile": True,
            "imagination_negative_spiral_preserves_budget_cooldown_reality_check": True,
            "memory_self_update_candidates_preserve_appraisal_quarantine": True,
            "one_event_cannot_build_all_axis_decision_packets": True,
            "global_synchrony_blocked": True,
        },
        "approval_scope": APPROVAL_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy_summary": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy_summary": INTEGRITY_HASH_POLICY,
        "no_write_guarantee_summary": NO_WRITE_GUARANTEE,
        "no_apply_guarantee_summary": NO_APPLY_GUARANTEE,
        "execution_trace_compatibility_summary": trace_summary,
        **_FALSE_PACKET_FLAGS,
        "global_synchrony_blocked": True,
        "notes": (
            "operator_report_command_scripts_operator_decision_round921_940_affect_dryrun_trace_packet_py",
            "no operator artifacts are written by this module",
        ),
    }
