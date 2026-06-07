"""Round961-980 read-only future dry-run simulation runner preflight.

This module consumes a valid Round941-960 future dry-run simulation request
packet and produces runner-readiness data only.  It does not execute the future
simulation, does not execute dry-run apply, does not authorize live apply, does
not create checkpoint/rollback/audit artifacts, and does not mutate runtime,
memory, persistence, emotion, hormone, AGP, fallback, or vector state.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_dryrun_trace_operator_decision_packet import affect_dryrun_trace_operator_decision_packet_summary
from adapters.affect_event_proposal_validator import ALL_AXIS_GROUPS, event_proposal_validator_summary
from adapters.affect_event_to_axis_proposal_map import REQUIRED_EVENT_CATEGORIES
from adapters.affect_future_dryrun_simulation_request_packet import (
    EXPLICIT_NON_PERMISSIONS as REQUEST_EXPLICIT_NON_PERMISSIONS,
    PASSING_OPERATOR_DECISION,
    REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY as REQUEST_REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
    REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION as REQUEST_REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION,
    affect_future_dryrun_simulation_request_packet_summary,
    build_affect_future_dryrun_simulation_request_packet,
    validate_affect_future_dryrun_simulation_request_packet,
)
from adapters.affect_proposal_transition_payload_builder import affect_proposal_transition_payload_builder_summary
from adapters.affect_reviewed_payload_dryrun_bridge import affect_reviewed_payload_dryrun_bridge_summary
from adapters.affect_transition_checkpoint_rollback_dryrun_trace import (
    DETERMINISTIC_SEQUENCE_ID_POLICY,
    INTEGRITY_HASH_POLICY,
    NO_APPLY_GUARANTEE,
    NO_WRITE_GUARANTEE,
    affect_checkpoint_rollback_execution_dryrun_trace_summary,
)
from adapters.affect_transition_checkpoint_rollback_plan import affect_transition_checkpoint_rollback_plan_summary
from adapters.affect_transition_payload_operator_handoff import affect_transition_payload_operator_handoff_summary
from adapters.emotion_transition_dryrun_apply_plan import future_live_apply_policy
from adapters.emotion_transition_gate import gate_required_for_future_apply_round
from adapters.emotion_transition_validator import validator_contract_summary

ROUND961_980_VERSION = "v3_round961_980_affect_future_dryrun_simulation_runner_preflight"
FEATURE_TRACK = "read_only_future_dryrun_simulation_runner_preflight_from_request_packet"
RUNNER_ACTION = "future_dryrun_simulation_runner_preflight_only"
REQUESTED_FUTURE_ACTION = "future_dryrun_simulation_request_only"

_FALSE_PREFLIGHT_FLAGS: dict[str, bool] = {
    "future_simulation_execution_requested_now": False,
    "future_simulation_executed": False,
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
    "execution_requested_now": False,
    "execution_allowed_now": False,
}

EXPLICIT_NON_PERMISSIONS: tuple[str, ...] = REQUEST_EXPLICIT_NON_PERMISSIONS + (
    "runner_preflight_is_readiness_data_only",
    "runner_preflight_success_does_not_execute_future_simulation",
    "runner_preflight_success_does_not_execute_or_authorize_dryrun_apply",
    "runner_preflight_success_does_not_authorize_live_apply",
    "runner_preflight_success_does_not_create_checkpoint_rollback_or_audit",
    "runner_preflight_success_does_not_schedule_autonomous_runtime_work",
)

REQUIRED_FOLLOWUP_BEFORE_RUNNER_EXECUTION: tuple[str, ...] = REQUEST_REQUIRED_FOLLOWUP_BEFORE_ANY_SIMULATION + (
    "separate_explicit_operator_runner_execution_round",
    "fresh_runner_preflight_validation",
    "fresh_no_write_no_apply_artifact_safety_check",
    "confirm_runner_execution_remains_read_only_before_calling_simulation",
)

REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY: tuple[str, ...] = REQUEST_REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY + (
    "separate_post_simulation_operator_review_before_any_apply",
    "fresh_apply_gate_validation_after_simulation_results",
)

RUNNER_ELIGIBILITY_SCOPE = (
    "a passed preflight means only that a later explicitly requested read-only simulation runner may be eligible; "
    "it grants no execution-now, dry-run apply, live apply, checkpoint, rollback, audit, memory, persistence, "
    "runtime, AGP, fallback, or vector permission"
)

NO_WRITE_GUARANTEE_PREFLIGHT = NO_WRITE_GUARANTEE + " runner preflight construction is pure data and creates no files."
NO_APPLY_GUARANTEE_PREFLIGHT = NO_APPLY_GUARANTEE + " runner preflight construction never executes simulation, dry-run apply, or live apply."


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _source_request_summary(packet: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packet_version": packet.get("packet_version"),
        "request_packet_passed": packet.get("request_packet_passed") is True,
        "validation_passed": validation.get("validation_passed") is True,
        "event_category": packet.get("event_category"),
        "source_operator_decision": packet.get("source_operator_decision"),
        "future_request_allowed": packet.get("future_dryrun_simulation_request_allowed") is True,
        "requested_now": packet.get("requested_now") is True or packet.get("future_dryrun_simulation_requested_now") is True,
        "dryrun_apply_allowed": packet.get("dryrun_apply_allowed") is True,
        "live_apply_allowed": packet.get("live_apply_allowed") is True,
        "dryrun_apply_executed": packet.get("dryrun_apply_executed") is True,
        "checkpoint_created": packet.get("checkpoint_created") is True,
        "rollback_created": packet.get("rollback_created") is True,
        "audit_written": packet.get("audit_written") is True,
        "blocked_reasons": tuple(packet.get("blocked_reasons", ())) + tuple(validation.get("blocked_reasons", ())),
    }


def _schema_summary() -> dict[str, Any]:
    required_fields = (
        "preflight_version",
        "event_category",
        "source_request_packet_summary",
        "source_operator_decision",
        "dryrun_trace_summary",
        "requested_future_action",
        "runner_action",
        "execution_requested_now",
        "execution_allowed_now",
        "dryrun_apply_allowed",
        "live_apply_allowed",
        "runner_eligibility_scope",
        "explicit_non_permissions",
        "required_followup_before_runner_execution",
        "required_followup_before_any_apply",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
        "no_write_guarantee",
        "no_apply_guarantee",
        "notes",
    )
    return {
        "preflight_version": ROUND961_980_VERSION,
        "required_fields": required_fields,
        "runner_action": RUNNER_ACTION,
        "requested_future_action": REQUESTED_FUTURE_ACTION,
        "runner_eligibility_scope": RUNNER_ELIGIBILITY_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_runner_execution": REQUIRED_FOLLOWUP_BEFORE_RUNNER_EXECUTION,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE_PREFLIGHT,
        "no_apply_guarantee": NO_APPLY_GUARANTEE_PREFLIGHT,
    }


def build_affect_future_dryrun_simulation_runner_preflight(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    operator_decision: str = PASSING_OPERATOR_DECISION,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a runner preflight by first building a source request packet."""

    request_packet = build_affect_future_dryrun_simulation_request_packet(
        event_category,
        deepcopy(dict(proposed_axis_deltas or {})),
        operator_decision,
        _copy_mapping(metadata),
    )
    return build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(request_packet, metadata)


def build_affect_future_dryrun_simulation_runner_preflight_from_request_packet(
    request_packet: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Consume a future dry-run simulation request packet into read-only runner readiness data."""

    meta = _copy_mapping(metadata)
    source = _copy_mapping(request_packet)
    validation = validate_affect_future_dryrun_simulation_request_packet(source)
    event_category = str(source.get("event_category") or "")
    source_request_packet_passed = source.get("request_packet_passed") is True and validation.get("validation_passed") is True
    source_future_request_allowed = source.get("future_dryrun_simulation_request_allowed") is True
    source_requested_now = source.get("requested_now") is True or source.get("future_dryrun_simulation_requested_now") is True
    source_dryrun_apply_allowed = source.get("dryrun_apply_allowed") is True
    source_live_apply_allowed = source.get("live_apply_allowed") is True

    blocked_reasons: list[str] = []
    blocked_reasons.extend(str(reason) for reason in source.get("blocked_reasons", ()))
    blocked_reasons.extend(str(reason) for reason in validation.get("blocked_reasons", ()))
    if event_category not in REQUIRED_EVENT_CATEGORIES:
        blocked_reasons.append("unknown_event_category_fail_closed")
    if not source_request_packet_passed:
        blocked_reasons.append("source_request_packet_failure_blocks_runner_preflight")
    if not source_future_request_allowed:
        blocked_reasons.append("source_future_request_allowed_false_blocks_runner_preflight")
    if source_requested_now:
        blocked_reasons.append("source_requested_now_true_blocks_runner_preflight")
    if source_dryrun_apply_allowed:
        blocked_reasons.append("source_dryrun_apply_allowed_true_blocks_runner_preflight")
    if source_live_apply_allowed:
        blocked_reasons.append("source_live_apply_allowed_true_blocks_runner_preflight")
    for key in (
        "checkpoint_created",
        "rollback_created",
        "audit_written",
        "checkpoint_write_allowed",
        "rollback_write_allowed",
        "audit_write_allowed",
        "runtime_mutation_performed",
        "persistence_write_performed",
        "memory_write_performed",
        "vector_read_performed",
        "vector_load_performed",
        "vector_contents_read",
        "vectors_loaded",
        "artifact_created_or_staged",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
    ):
        if source.get(key) is True:
            blocked_reasons.append(f"source_{key}_true_blocks_runner_preflight")

    safety_result = source.get("event_safety_result", {}) if isinstance(source.get("event_safety_result"), Mapping) else {}
    for key, value in safety_result.items():
        if value is not True:
            blocked_reasons.append(f"{key}_must_be_preserved")

    delta_axes = tuple(source.get("bounded_apply_candidate_sequence_summary", {}).get("axes", ())) if isinstance(source.get("bounded_apply_candidate_sequence_summary"), Mapping) else ()
    if len(tuple(axis for axis in delta_axes if axis != "no_axis_delta_candidate")) >= len(ALL_AXIS_GROUPS):
        blocked_reasons.append("one_event_cannot_build_all_axis_runner_preflights_must_be_preserved")

    runner_passed = source_request_packet_passed and source_future_request_allowed and not source_requested_now and not source_dryrun_apply_allowed and not source_live_apply_allowed and not blocked_reasons
    runner_status = (
        "valid_read_only_future_dryrun_simulation_runner_preflight_no_execution_no_write"
        if runner_passed
        else "blocked_future_dryrun_simulation_runner_preflight_fail_closed"
    )
    warnings = sorted(
        set(str(warning) for warning in source.get("warnings", ()))
        | {
            "runner_preflight_success_does_not_execute_future_simulation_now",
            "runner_preflight_success_does_not_grant_dryrun_apply_permission",
            "runner_preflight_success_does_not_grant_live_apply_permission",
            "runner_preflight_success_does_not_write_audit_checkpoint_or_rollback",
            "runner_preflight_is_not_autonomous_runtime_scheduling",
        }
    )

    return {
        "preflight_version": ROUND961_980_VERSION,
        "feature_track": FEATURE_TRACK,
        "runner_preflight_passed": runner_passed,
        "runner_preflight_status": runner_status,
        "event_category": event_category,
        "source_request_packet_passed": source_request_packet_passed,
        "source_future_request_allowed": source_future_request_allowed,
        "source_requested_now": source_requested_now,
        "source_dryrun_apply_allowed": source_dryrun_apply_allowed,
        "source_live_apply_allowed": source_live_apply_allowed,
        "runner_preflight_built": True,
        "request_packet_accepted": runner_passed,
        "future_simulation_runner_eligible": runner_passed,
        **_FALSE_PREFLIGHT_FLAGS,
        "source_operator_decision": source.get("source_operator_decision"),
        "source_request_packet_summary": _source_request_summary(source, validation),
        "dryrun_trace_summary": deepcopy(source.get("dryrun_trace_summary", {})),
        "requested_future_action": source.get("requested_future_action", REQUESTED_FUTURE_ACTION),
        "runner_action": RUNNER_ACTION,
        "runner_eligibility_scope": RUNNER_ELIGIBILITY_SCOPE,
        "explicit_non_permissions": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_runner_execution": REQUIRED_FOLLOWUP_BEFORE_RUNNER_EXECUTION,
        "required_followup_before_any_apply": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy": INTEGRITY_HASH_POLICY,
        "no_write_guarantee": NO_WRITE_GUARANTEE_PREFLIGHT,
        "no_apply_guarantee": NO_APPLY_GUARANTEE_PREFLIGHT,
        "event_safety_result": deepcopy(safety_result),
        "global_synchrony_blocked": True,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "warnings": warnings,
        "metadata": meta,
        "notes": (
            "runner_preflight_is_read_only_readiness_data_only",
            "no_future_dryrun_simulation_executed_now",
            "no_dryrun_apply_or_live_apply_executed_or_authorized",
            "no_checkpoint_rollback_audit_or_operator_artifact_created",
            "no_live_emotion_hormone_runtime_memory_persistence_or_vector_state_mutated",
            "runner_execution_requires_separate_followup_authorization",
        ),
    }


def validate_affect_future_dryrun_simulation_runner_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Validate runner preflight readiness and non-permission invariants."""

    p = _copy_mapping(preflight)
    reasons: list[str] = []
    if p.get("preflight_version") != ROUND961_980_VERSION:
        reasons.append("preflight_version_mismatch")
    if p.get("event_category") not in REQUIRED_EVENT_CATEGORIES:
        reasons.append("unknown_event_category_fail_closed")
    if p.get("source_request_packet_passed") is not True:
        reasons.append("source_request_packet_failure_blocks_runner_preflight")
    if p.get("source_future_request_allowed") is not True:
        reasons.append("source_future_request_allowed_false_blocks_runner_preflight")
    if p.get("source_requested_now") is not False:
        reasons.append("source_requested_now_true_blocks_runner_preflight")
    if p.get("source_dryrun_apply_allowed") is not False:
        reasons.append("source_dryrun_apply_allowed_true_blocks_runner_preflight")
    if p.get("source_live_apply_allowed") is not False:
        reasons.append("source_live_apply_allowed_true_blocks_runner_preflight")
    if p.get("runner_action") != RUNNER_ACTION:
        reasons.append("runner_action_must_be_preflight_only")
    if p.get("requested_future_action") != REQUESTED_FUTURE_ACTION:
        reasons.append("requested_future_action_must_remain_request_only")
    for key, expected in _FALSE_PREFLIGHT_FLAGS.items():
        if p.get(key) is not expected:
            reasons.append(f"{key}_must_remain_false")
    if p.get("runner_preflight_passed") is True:
        for key in ("future_simulation_executed", "dryrun_apply_executed", "audit_written", "checkpoint_created", "rollback_created"):
            if p.get(key) is not False:
                reasons.append(f"runner_preflight_success_must_not_imply_{key}")
    for key in (
        "runner_eligibility_scope",
        "explicit_non_permissions",
        "required_followup_before_runner_execution",
        "required_followup_before_any_apply",
        "deterministic_sequence_id_policy",
        "integrity_hash_policy",
        "no_write_guarantee",
        "no_apply_guarantee",
    ):
        if not p.get(key):
            reasons.append(f"{key}_required")
    if set(p.get("explicit_non_permissions", ())) != set(EXPLICIT_NON_PERMISSIONS):
        reasons.append("explicit_non_permissions_must_match_round961_980_scope")
    safety_result = p.get("event_safety_result", {}) if isinstance(p.get("event_safety_result"), Mapping) else {}
    for key, value in safety_result.items():
        if value is not True:
            reasons.append(f"{key}_must_be_preserved")
    if p.get("global_synchrony_blocked") is not True:
        reasons.append("global_synchrony_must_remain_blocked")

    passed = not reasons and p.get("runner_preflight_passed") is True
    return {
        "preflight_version": ROUND961_980_VERSION,
        "validation_passed": passed,
        "validation_status": "valid_read_only_future_dryrun_simulation_runner_preflight" if passed else "blocked_invalid_future_dryrun_simulation_runner_preflight",
        "event_category": p.get("event_category"),
        "source_operator_decision": p.get("source_operator_decision"),
        "blocked_reasons": sorted(set(reasons)),
        "warnings": ["validation_success_does_not_grant_execution_apply_write_checkpoint_rollback_audit_memory_persistence_runtime_or_vector_permission"],
        **_FALSE_PREFLIGHT_FLAGS,
    }


def affect_future_dryrun_simulation_runner_preflight_summary() -> dict[str, Any]:
    """Return a compact operator-facing summary for the runner preflight surface."""

    return {
        "preflight_version": ROUND961_980_VERSION,
        "feature_track": FEATURE_TRACK,
        "scope": "runner_preflight_request_consumption_readiness_only_no_execution_no_apply_no_write_no_mutation",
        "runner_preflight_schema_summary": _schema_summary(),
        "runner_safety_rules": {
            "unknown_event_category_fails_closed": True,
            "source_request_packet_failure_blocks_runner_preflight": True,
            "source_future_request_allowed_false_blocks_runner_preflight": True,
            "source_requested_now_true_blocks_runner_preflight": True,
            "source_dryrun_apply_allowed_true_blocks_runner_preflight": True,
            "source_live_apply_allowed_true_blocks_runner_preflight": True,
            "runner_preflight_success_does_not_imply_execution_now": True,
            "runner_preflight_success_does_not_imply_future_simulation_execution": True,
            "runner_preflight_success_does_not_imply_dryrun_apply_permission": True,
            "runner_preflight_success_does_not_imply_live_apply_permission": True,
            "runner_preflight_success_does_not_imply_audit_write": True,
            "runner_preflight_success_does_not_imply_checkpoint_creation": True,
            "runner_preflight_success_does_not_imply_rollback_creation": True,
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
            "one_event_cannot_build_all_axis_runner_preflights": True,
            "global_synchrony_blocked": True,
        },
        "runner_eligibility_scope_summary": RUNNER_ELIGIBILITY_SCOPE,
        "explicit_non_permission_summary": EXPLICIT_NON_PERMISSIONS,
        "required_followup_before_runner_execution_summary": REQUIRED_FOLLOWUP_BEFORE_RUNNER_EXECUTION,
        "required_followup_before_any_apply_summary": REQUIRED_FOLLOWUP_BEFORE_ANY_APPLY,
        "deterministic_sequence_id_policy_summary": DETERMINISTIC_SEQUENCE_ID_POLICY,
        "integrity_hash_policy_summary": INTEGRITY_HASH_POLICY,
        "no_write_guarantee_summary": NO_WRITE_GUARANTEE_PREFLIGHT,
        "no_apply_guarantee_summary": NO_APPLY_GUARANTEE_PREFLIGHT,
        "future_request_packet_compatibility_summary": affect_future_dryrun_simulation_request_packet_summary(),
        "operator_decision_packet_compatibility_summary": affect_dryrun_trace_operator_decision_packet_summary(),
        "execution_dryrun_trace_compatibility_summary": affect_checkpoint_rollback_execution_dryrun_trace_summary(),
        "checkpoint_rollback_plan_compatibility_summary": affect_transition_checkpoint_rollback_plan_summary(),
        "dryrun_bridge_compatibility_summary": affect_reviewed_payload_dryrun_bridge_summary(),
        "handoff_compatibility_summary": affect_transition_payload_operator_handoff_summary(),
        "builder_compatibility_summary": affect_proposal_transition_payload_builder_summary(),
        "proposal_validator_compatibility_summary": event_proposal_validator_summary(),
        "emotion_transition_validator_compatibility_summary": validator_contract_summary(),
        "emotion_transition_gate_compatibility_summary": gate_required_for_future_apply_round(),
        "dryrun_apply_plan_compatibility_summary": future_live_apply_policy(),
        "exactly_one_next_implementation_recommendation": "add_a_separate_read_only_future_simulation_runner_execution_surface_that_revalidates_this_preflight_without_apply_or_artifact_writes",
        **_FALSE_PREFLIGHT_FLAGS,
        "global_synchrony_blocked": True,
        "notes": (
            "operator_report_command_scripts_operator_preflight_round961_980_affect_future_dryrun_runner_py",
            "no operator artifacts are written by this module",
        ),
    }
