"""Round941-960 read-only future dry-run simulation request packet.

This module derives an intent-only future dry-run simulation request packet from
an operator decision packet.  It records that a later read-only simulation may be
requested only when the source operator decision is
``approve_for_future_dryrun_simulation`` and the source decision/trace packets
are valid.  It never executes a dry-run apply, never authorizes live apply,
never creates checkpoint/rollback/audit artifacts, and never mutates runtime,
memory, persistence, emotion, hormone, AGP, fallback, or vector state.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_dryrun_trace_operator_decision_packet import (
    APPROVAL_SCOPE as SOURCE_APPROVAL_SCOPE,
    EXPLICIT_NON_PERMISSIONS as SOURCE_EXPLICIT_NON_PERMISSIONS,
    REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY as SOURCE_REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
    SUPPORTED_OPERATOR_DECISIONS,
    affect_dryrun_trace_operator_decision_packet_summary,
    build_affect_dryrun_trace_operator_decision_packet,
    validate_affect_dryrun_trace_operator_decision_packet,
)
from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS
from adapters.affect_event_to_axis_proposal_map import REQUIRED_EVENT_CATEGORIES
from adapters.affect_transition_checkpoint_rollback_dryrun_trace import (
    DETERMINISTIC_SEQUENCE_ID_POLICY,
    INTEGRITY_HASH_POLICY,
    NO_APPLY_GUARANTEE,
    NO_WRITE_GUARANTEE,
    affect_checkpoint_rollback_execution_dryrun_trace_summary,
)
from adapters.emotion_transition_dryrun_apply_plan import future_live_apply_policy
from adapters.emotion_transition_gate import gate_required_for_future_apply_round
from adapters.emotion_transition_validator import validator_contract_summary
from adapters.affect_transition_checkpoint_rollback_plan import affect_transition_checkpoint_rollback_plan_summary
from adapters.affect_reviewed_payload_dryrun_bridge import affect_reviewed_payload_dryrun_bridge_summary
from adapters.affect_transition_payload_operator_handoff import affect_transition_payload_operator_handoff_summary
from adapters.affect_proposal_transition_payload_builder import affect_proposal_transition_payload_builder_summary
from adapters.affect_event_proposal_validator import event_proposal_validator_summary

ROUND941_960_VERSION = "v3_round941_960_affect_future_dryrun_simulation_request_packet"
FEATURE_TRACK = "read_only_future_dryrun_simulation_request_packet_from_operator_decision"
REQUESTED_FUTURE_ACTION = "future_dryrun_simulation_request_only"
PASSING_OPERATOR_DECISION = "approve_for_future_dryrun_simulation"

_FALSE_REQUEST_FLAGS: dict[str, bool] = {
    "future_dryrun_simulation_requested_now": False,
    "dryrun_apply_executed": False,
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
    "emotion_transition_applied": False,
    "hormone_transition_applied": False,
}

EXPLICIT_NON_PERMISSIONS: tuple[str, ...] = SOURCE_EXPLICIT_NON_PERMISSIONS + (
    "future_dryrun_simulation_request_is_intent_only",
    "request_packet_success_does_not_execute_simulation_now",
    "request_packet_success_does_not_schedule_autonomous_runtime_work",
    "request_packet_success_does_not_write_audit_record",
)

REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION: tuple[str, ...] = (
    "separate_explicit_operator_simulation_request_round",
    "fresh_read_only_simulation_scope_review",
    "fresh_source_decision_packet_validation",
    "fresh_source_trace_validation",
    "confirm_no_checkpoint_rollback_audit_or_operator_artifact_write_paths",
    "focused_request_packet_tests_and_artifact_safety_check",
)

REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY: tuple[str, ...] = SOURCE_REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY + (
    "separate_apply_authorization_after_future_simulation_review",
    "new_checkpoint_rollback_artifact_policy_review_before_any_write",
)

APPROVAL_SCOPE = (
    "a valid approve_for_future_dryrun_simulation operator decision may create only an intent packet "
    "for a later read-only dry-run simulation request; it grants no execution-now, dry-run apply, "
    "live apply, checkpoint, rollback, audit, memory, persistence, runtime, AGP, fallback, or vector permission"
)

NO_WRITE_GUARANTEE_REQUEST = NO_WRITE_GUARANTEE + " request packet construction is pure data and creates no files."
NO_APPLY_GUARANTEE_REQUEST = NO_APPLY_GUARANTEE + " request packet construction never executes dry-run or live apply."


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _deltas(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _source_summary(decision_packet: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_version": decision_packet.get("packet_version"),
        "packet_passed": decision_packet.get("packet_passed") is True,
        "validation_passed": validation.get("validation_passed") is True,
        "event_category": decision_packet.get("event_category"),
        "operator_decision": decision_packet.get("operator_decision"),
        "decision_packet_built": decision_packet.get("decision_packet_built") is True,
        "trace_passed": decision_packet.get("trace_passed") is True,
        "trace_accepted": decision_packet.get("trace_accepted") is True,
        "blocked_reasons": tuple(decision_packet.get("blocked_reasons", ())) + tuple(validation.get("blocked_reasons", ())),
    }


def _schema_summary() -> dict[str, Any]:
    return {
        "packet_version": ROUND941_960_VERSION,
        "required_fields": (
            "packet_version",
            "event_category",
            "source_operator_decision",
            "source_decision_packet_summary",
            "dryrun_trace_summary",
            "requested_future_action",
            "requested_now",
            "execution_allowed_now",
            "dryrun_apply_allowed",
            "live_apply_allowed",
            "approval_scope",
            "explicit_non_permissions",
            "required_followup_before_any_simulation",
            "required_followup_before_any_apply",
            "deterministic_sequence_id_policy",
            "integrity_hash_policy",
            "no_write_guarantee",
            "no_apply_guarantee",
            "notes",
        ),
        "supported_source_decisions": SUPPORTED_OPERATOR_DECISIONS,
        "passing_source_decision": PASSING_OPERATOR_DECISION,
        "requested_future_action": REQUESTED_FUTURE_ACTION,
        "approval_scope": APPROVAL_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_simulation": REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE_REQUEST,
        "no_apply_guarantee": NO_APPLY_GUARANTEE_REQUEST,
    }


def build_affect_future_dryrun_simulation_request_packet(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    operator_decision: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pure intent request packet by first building a source decision packet."""

    source_packet = build_affect_dryrun_trace_operator_decision_packet(
        event_category,
        _deltas(proposed_axis_deltas),
        operator_decision,
        _copy_mapping(metadata),
    )
    return build_affect_future_dryrun_simulation_request_from_decision_packet(source_packet, metadata)


def build_affect_future_dryrun_simulation_request_from_decision_packet(
    decision_packet: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a future simulation request intent packet from a source decision packet."""

    meta = _copy_mapping(metadata)
    source = _copy_mapping(decision_packet)
    source_validation = validate_affect_dryrun_trace_operator_decision_packet(source)
    event_category = str(source.get("event_category") or "")
    operator_decision = str(source.get("operator_decision") or "")
    source_decision_packet_passed = source.get("packet_passed") is True and source_validation.get("validation_passed") is True
    source_trace_passed = source.get("trace_passed") is True
    event_known = event_category in REQUIRED_EVENT_CATEGORIES
    decision_valid = operator_decision in SUPPORTED_OPERATOR_DECISIONS
    approval_decision = operator_decision == PASSING_OPERATOR_DECISION

    blocked_reasons: list[str] = []
    blocked_reasons.extend(str(reason) for reason in source.get("blocked_reasons", ()))
    blocked_reasons.extend(str(reason) for reason in source_validation.get("blocked_reasons", ()))
    if not event_known:
        blocked_reasons.append("unknown_event_category_fail_closed")
    if not decision_valid:
        blocked_reasons.append("invalid_operator_decision_fail_closed")
    if not source_decision_packet_passed:
        blocked_reasons.append("source_decision_packet_failure_blocks_request_packet")
    if not source_trace_passed:
        blocked_reasons.append("source_trace_failure_blocks_request_packet")
    if decision_valid and not approval_decision:
        blocked_reasons.append(f"{operator_decision}_must_not_create_future_simulation_request")
    if approval_decision and (not source_decision_packet_passed or not source_trace_passed):
        blocked_reasons.append("approve_for_future_dryrun_simulation_requires_valid_source_decision_and_trace")

    safety_result = source.get("event_safety_result", {}) if isinstance(source.get("event_safety_result"), Mapping) else {}
    for key, value in safety_result.items():
        if value is not True:
            blocked_reasons.append(f"{key}_must_be_preserved")

    delta_axes = tuple(source.get("bounded_apply_candidate_sequence_summary", {}).get("axes", ())) if isinstance(source.get("bounded_apply_candidate_sequence_summary"), Mapping) else ()
    if len(tuple(axis for axis in delta_axes if axis != "no_axis_delta_candidate")) >= len(ALL_AXIS_GROUPS):
        blocked_reasons.append("one_event_cannot_build_all_axis_request_packets_must_be_preserved")

    request_passed = approval_decision and source_decision_packet_passed and source_trace_passed and not blocked_reasons
    request_status = (
        "valid_read_only_future_dryrun_simulation_request_packet_no_execution_no_write"
        if request_passed
        else "blocked_no_future_dryrun_simulation_request_fail_closed"
    )
    request_recorded = request_passed
    request_allowed = request_passed

    warnings = sorted(
        set(str(warning) for warning in source.get("warnings", ()))
        | {
            "future_dryrun_simulation_request_allowed_does_not_execute_now",
            "future_dryrun_simulation_request_allowed_does_not_grant_dryrun_apply",
            "future_dryrun_simulation_request_allowed_does_not_grant_live_apply",
            "future_dryrun_simulation_request_recorded_does_not_write_audit",
            "request_packet_built_does_not_create_checkpoint_or_rollback",
            "request_packet_is_intent_only_no_runtime_scheduling",
        }
    )

    return {
        "packet_version": ROUND941_960_VERSION,
        "feature_track": FEATURE_TRACK,
        "request_packet_passed": request_passed,
        "request_packet_status": request_status,
        "event_category": event_category,
        "source_operator_decision": operator_decision,
        "source_decision_packet_passed": source_decision_packet_passed,
        "source_trace_passed": source_trace_passed,
        "future_dryrun_simulation_request_recorded": request_recorded,
        "future_dryrun_simulation_request_allowed": request_allowed,
        **_FALSE_REQUEST_FLAGS,
        "requested_future_action": REQUESTED_FUTURE_ACTION,
        "requested_now": False,
        "execution_allowed_now": False,
        "source_decision_packet_summary": _source_summary(source, source_validation),
        "dryrun_trace_summary": source.get("dryrun_trace_summary", {}),
        "approval_scope": APPROVAL_SCOPE,
        "source_approval_scope": SOURCE_APPROVAL_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_simulation": REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE_REQUEST,
        "no_apply_guarantee": NO_APPLY_GUARANTEE_REQUEST,
        "event_safety_result": safety_result,
        "global_synchrony_blocked": True,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "warnings": warnings,
        "metadata": meta,
        "notes": (
            "request_packet_is_pure_intent_data_only",
            "no_dryrun_apply_executed_now",
            "no_checkpoint_rollback_audit_or_operator_artifact_created",
            "no_live_emotion_hormone_runtime_memory_persistence_or_vector_state_mutated",
            "future_simulation_requires_separate_followup_authorization",
        ),
    }


def validate_affect_future_dryrun_simulation_request_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate request-packet read-only intent and fail-closed rules."""

    p = _copy_mapping(packet)
    reasons: list[str] = []
    if p.get("packet_version") != ROUND941_960_VERSION:
        reasons.append("packet_version_mismatch")
    if p.get("event_category") not in REQUIRED_EVENT_CATEGORIES:
        reasons.append("unknown_event_category_fail_closed")
    if p.get("source_operator_decision") not in SUPPORTED_OPERATOR_DECISIONS:
        reasons.append("invalid_operator_decision_fail_closed")
    if p.get("source_decision_packet_passed") is not True:
        reasons.append("source_decision_packet_failure_blocks_request_packet")
    if p.get("source_trace_passed") is not True:
        reasons.append("source_trace_failure_blocks_request_packet")
    if p.get("source_operator_decision") != PASSING_OPERATOR_DECISION:
        reasons.append(f"{p.get('source_operator_decision')}_must_not_create_future_simulation_request")
    for key, expected in _FALSE_REQUEST_FLAGS.items():
        if p.get(key) is not expected:
            reasons.append(f"{key}_must_remain_false")
    if p.get("future_dryrun_simulation_request_allowed") is True and p.get("execution_allowed_now") is not False:
        reasons.append("future_request_allowed_must_not_imply_execution_now")
    if p.get("future_dryrun_simulation_request_allowed") is True and p.get("dryrun_apply_allowed") is not False:
        reasons.append("future_request_allowed_must_not_imply_dryrun_apply_permission")
    if p.get("future_dryrun_simulation_request_allowed") is True and p.get("live_apply_allowed") is not False:
        reasons.append("future_request_allowed_must_not_imply_live_apply_permission")
    if p.get("future_dryrun_simulation_request_recorded") is True and p.get("audit_written") is not False:
        reasons.append("future_request_recorded_must_not_imply_audit_write")
    if p.get("request_packet_passed") is True and p.get("checkpoint_created") is not False:
        reasons.append("request_packet_built_must_not_imply_checkpoint_creation")
    if p.get("request_packet_passed") is True and p.get("rollback_created") is not False:
        reasons.append("request_packet_built_must_not_imply_rollback_creation")
    if p.get("requested_future_action") != REQUESTED_FUTURE_ACTION:
        reasons.append("requested_future_action_must_be_request_only")
    if p.get("requested_now") is not False or p.get("execution_allowed_now") is not False:
        reasons.append("request_packet_must_not_execute_now")
    for key in (
        "approval_scope",
        "explicit_non_permissions",
        "required_followup_before_any_simulation",
        "required_followup_before_any_apply",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
        "no_write_guarantee",
        "no_apply_guarantee",
    ):
        if not p.get(key):
            reasons.append(f"{key}_required")
    if set(p.get("explicit_non_permissions", ())) != set(EXPLICIT_NON_PERMISSIONS):
        reasons.append("explicit_non_permissions_must_match_round941_960_scope")
    safety_result = p.get("event_safety_result", {}) if isinstance(p.get("event_safety_result"), Mapping) else {}
    for key, value in safety_result.items():
        if value is not True:
            reasons.append(f"{key}_must_be_preserved")
    if p.get("global_synchrony_blocked") is not True:
        reasons.append("global_synchrony_must_remain_blocked")

    passed = not reasons and p.get("request_packet_passed") is True
    return {
        "packet_version": ROUND941_960_VERSION,
        "validation_passed": passed,
        "validation_status": "valid_read_only_future_dryrun_simulation_request_packet" if passed else "blocked_invalid_future_dryrun_simulation_request_packet",
        "event_category": p.get("event_category"),
        "source_operator_decision": p.get("source_operator_decision"),
        "blocked_reasons": sorted(set(reasons)),
        "warnings": ["validation_success_does_not_grant_execution_apply_write_checkpoint_rollback_audit_memory_persistence_runtime_or_vector_permission"],
        **_FALSE_REQUEST_FLAGS,
    }


def affect_future_dryrun_simulation_request_packet_summary() -> dict[str, Any]:
    """Return a compact operator-facing summary for the request packet surface."""

    return {
        "packet_version": ROUND941_960_VERSION,
        "feature_track": FEATURE_TRACK,
        "scope": "operator_request_packet_intent_surface_only_no_execution_no_apply_no_write_no_mutation",
        "request_packet_schema_summary": _schema_summary(),
        "supported_source_decision_summary": {
            "supported_operator_decisions": SUPPORTED_OPERATOR_DECISIONS,
            "only_passing_decision": PASSING_OPERATOR_DECISION,
            "reject_hold_revision_are_non_request_fail_closed": True,
        },
        "request_safety_rules": {
            "unknown_event_category_fails_closed": True,
            "invalid_operator_decision_fails_closed": True,
            "source_decision_packet_failure_blocks_request_packet": True,
            "source_trace_failure_blocks_request_packet": True,
            "only_approve_for_future_dryrun_simulation_can_pass": True,
            "reject_hold_request_revision_do_not_create_future_request": True,
            "future_request_allowed_does_not_imply_execution_now": True,
            "future_request_allowed_does_not_imply_dryrun_apply_permission": True,
            "future_request_allowed_does_not_imply_live_apply_permission": True,
            "future_request_recorded_does_not_imply_audit_write": True,
            "request_packet_built_does_not_create_checkpoint_or_rollback": True,
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
            "one_event_cannot_build_all_axis_request_packets": True,
            "global_synchrony_blocked": True,
        },
        "approval_scope": APPROVAL_SCOPE,
        "explicit_non_permission_summary": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_any_simulation_summary": REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION,
        "required_followup_before_any_apply_summary": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy_summary": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy_summary": INTEGRITY_HASH_POLICY,
        "no_write_guarantee_summary": NO_WRITE_GUARANTEE_REQUEST,
        "no_apply_guarantee_summary": NO_APPLY_GUARANTEE_REQUEST,
        "decision_packet_compatibility_summary": affect_dryrun_trace_operator_decision_packet_summary(),
        "execution_dryrun_trace_compatibility_summary": affect_checkpoint_rollback_execution_dryrun_trace_summary(),
        "checkpoint_rollback_plan_compatibility_summary": affect_transition_checkpoint_rollback_plan_summary(),
        "dryrun_bridge_compatibility_summary": affect_reviewed_payload_dryrun_bridge_summary(),
        "handoff_compatibility_summary": affect_transition_payload_operator_handoff_summary(),
        "builder_compatibility_summary": affect_proposal_transition_payload_builder_summary(),
        "proposal_validator_compatibility_summary": event_proposal_validator_summary(),
        "emotion_transition_validator_compatibility_summary": validator_contract_summary(),
        "emotion_transition_gate_compatibility_summary": gate_required_for_future_apply_round(),
        "dryrun_apply_plan_compatibility_summary": future_live_apply_policy(),
        "exactly_one_next_implementation_recommendation": "add_a_separate_read_only_future_simulation_runner_that_consumes_this_request_packet_without_apply_or_artifact_writes",
        **_FALSE_REQUEST_FLAGS,
        "global_synchrony_blocked": True,
        "notes": (
            "operator_report_command_scripts_operator_request_round941_960_affect_future_dryrun_simulation_py",
            "no operator artifacts are written by this module",
        ),
    }
