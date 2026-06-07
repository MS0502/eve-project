"""Round881-900 read-only affect transition checkpoint/rollback plan.

This module defines detached checkpoint and rollback planning data for a future
bounded in-memory affect/hormone apply round.  It never creates checkpoints,
never writes rollback files, never applies transitions, never mutates runtime,
emotion, hormone, memory, persistence, vector, AGP, or fallback state, and never
loads vector contents.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MEMORY_SELF_UPDATE_EVENTS,
    REQUIRED_EVENT_CATEGORIES,
    SPEECH_PRESSURE_EVENTS,
    event_to_axis_proposal_map,
)
from adapters.affect_event_proposal_validator import (
    ALL_AXIS_GROUPS,
    OPERATIONAL_HARDWARE_AXES,
    SOCIAL_SELF_IDENTITY_AXES,
    validate_affect_event_proposal,
)
from adapters.affect_reviewed_payload_dryrun_bridge import (
    affect_reviewed_payload_dryrun_bridge_summary,
    build_dryrun_preflight_from_operator_handoff,
)
from adapters.affect_transition_payload_operator_handoff import affect_transition_payload_operator_handoff_summary
from adapters.affect_proposal_transition_payload_builder import affect_proposal_transition_payload_builder_summary
from adapters.affect_event_proposal_validator import event_proposal_validator_summary
from adapters.emotion_transition_validator import validator_contract_summary
from adapters.emotion_transition_gate import gate_required_for_future_apply_round
from adapters.emotion_transition_dryrun_apply_plan import future_live_apply_policy
from adapters.affect_hormone_neural_rhythm_registry import anti_global_synchrony_policy, hardware_governor_policy

ROUND881_900_VERSION = "v3_round881_900_affect_transition_checkpoint_rollback_plan"
FEATURE_TRACK = "read_only_affect_transition_checkpoint_and_rollback_plan_for_future_bounded_apply"

CHECKPOINT_CAPTURE_SURFACES: tuple[str, ...] = (
    "affect_axis_values_before",
    "hormone_axis_values_before",
    "rhythm_phase_state_before",
    "activation_pattern_summary_before",
    "modulation_surface_state_before",
    "recovery_cooldown_state_before",
    "pending_transition_payload",
    "dryrun_preflight_summary",
    "operator_review_packet_summary",
    "hardware_governor_state_summary",
    "global_synchrony_guard_state",
    "agp_fallback_boundary_state",
    "memory_write_guard_state",
    "persistence_guard_state",
    "vector_guard_state",
    "timestamp_policy",
    "deterministic_sequence_id_policy",
    "integrity_hash_policy",
    "notes",
)

ROLLBACK_RESTORE_SURFACES: tuple[str, ...] = (
    "restore_affect_axis_values",
    "restore_hormone_axis_values",
    "restore_rhythm_phase_state",
    "restore_activation_pattern_summary",
    "restore_modulation_surface_state",
    "restore_recovery_cooldown_state",
    "cancel_pending_transition_payload",
    "restore_global_synchrony_guard_state",
    "restore_agp_fallback_boundary_state",
    "preserve_memory_write_guard",
    "preserve_persistence_guard",
    "preserve_vector_guard",
    "rollback_audit_record",
    "rollback_reason",
    "deterministic_restore_order",
    "notes",
)

DETERMINISTIC_RESTORE_ORDER: tuple[str, ...] = (
    "block_new_apply_permission",
    "cancel_pending_transition_payload",
    "restore_affect_axis_values",
    "restore_hormone_axis_values",
    "restore_rhythm_phase_state",
    "restore_activation_pattern_summary",
    "restore_modulation_surface_state",
    "restore_recovery_cooldown_state",
    "restore_global_synchrony_guard_state",
    "restore_agp_fallback_boundary_state",
    "preserve_memory_write_guard",
    "preserve_persistence_guard",
    "preserve_vector_guard",
    "emit_rollback_audit_record",
)

_FALSE_BOUNDARIES: dict[str, bool] = {
    "checkpoint_created": False,
    "rollback_created": False,
    "checkpoint_write_allowed": False,
    "rollback_write_allowed": False,
    "dryrun_apply_allowed": False,
    "live_apply_allowed": False,
    "apply_permission_granted": False,
    "state_mutation_performed": False,
    "runtime_mutation_performed": False,
    "memory_write_performed": False,
    "persistence_write_performed": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "artifact_created_or_staged": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _copy_mapping(value: Any) -> dict[str, Any]:
    return deepcopy(dict(_as_mapping(value)))


def _deltas(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(key, str):
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                out[key] = float("nan")
    return out


def _checkpoint_schema(event_category: str, deltas: Mapping[str, float], bridge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": "future_bounded_affect_apply_checkpoint_schema_read_only",
        "surfaces": CHECKPOINT_CAPTURE_SURFACES,
        "affect_axis_values_before": "future_capture_only_no_capture_this_round",
        "hormone_axis_values_before": "future_capture_only_no_capture_this_round",
        "rhythm_phase_state_before": "future_capture_only_no_capture_this_round",
        "activation_pattern_summary_before": "future_capture_only_no_capture_this_round",
        "modulation_surface_state_before": "future_capture_only_no_capture_this_round",
        "recovery_cooldown_state_before": "future_capture_only_no_capture_this_round",
        "pending_transition_payload": {"event_category": event_category, "axis_delta_keys": tuple(sorted(deltas))},
        "dryrun_preflight_summary": {
            "bridge_passed": bridge.get("bridge_passed") is True,
            "dryrun_preflight_eligible": bridge.get("dryrun_preflight_eligible") is True,
            "dryrun_apply_allowed": False,
            "live_apply_allowed": False,
        },
        "operator_review_packet_summary": "future_summary_only_no_artifact_write",
        "hardware_governor_state_summary": "future_snapshot_only_non_panic_operational_boundary",
        "global_synchrony_guard_state": {"global_synchrony_blocked": True},
        "agp_fallback_boundary_state": {"agp_bypass_allowed": False, "fallback_bypass_allowed": False},
        "memory_write_guard_state": {"memory_write_allowed": False, "preserve_quarantine_and_appraisal": True},
        "persistence_guard_state": {"production_persistence_enabled": False, "runtime_mapping_enabled_default": False, "enforcement_enabled_default": False},
        "vector_guard_state": {"vector_contents_read": False, "vectors_loaded": False},
        "timestamp_policy": "future_deterministic_operator_supplied_or_monotonic_sequence_only_no_wall_clock_capture_this_round",
        "deterministic_sequence_id_policy": "future_stable_sequence_id_from_operator_event_order_no_random",
        "integrity_hash_policy": "future_canonical_json_sha256_over_checkpoint_payload_no_file_write_this_round",
        "notes": ("schema_definition_only", "checkpoint_not_created"),
    }


def _rollback_schema(event_category: str) -> dict[str, Any]:
    return {
        "schema_name": "future_bounded_affect_apply_rollback_schema_read_only",
        "surfaces": ROLLBACK_RESTORE_SURFACES,
        "restore_affect_axis_values": "restore_from_checkpoint_affect_axis_values_before",
        "restore_hormone_axis_values": "restore_from_checkpoint_hormone_axis_values_before",
        "restore_rhythm_phase_state": "restore_from_checkpoint_rhythm_phase_state_before",
        "restore_activation_pattern_summary": "restore_from_checkpoint_activation_pattern_summary_before",
        "restore_modulation_surface_state": "restore_from_checkpoint_modulation_surface_state_before",
        "restore_recovery_cooldown_state": "restore_from_checkpoint_recovery_cooldown_state_before",
        "cancel_pending_transition_payload": True,
        "restore_global_synchrony_guard_state": {"global_synchrony_blocked": True},
        "restore_agp_fallback_boundary_state": {"agp_bypass_allowed": False, "fallback_bypass_allowed": False},
        "preserve_memory_write_guard": True,
        "preserve_persistence_guard": True,
        "preserve_vector_guard": True,
        "rollback_audit_record": {
            "event_category": event_category,
            "checkpoint_sequence_id": "future_required",
            "rollback_reason": "future_required",
            "restored_surfaces": DETERMINISTIC_RESTORE_ORDER,
            "state_mutation_performed_by_plan": False,
            "persistence_write_performed_by_plan": False,
            "memory_write_performed_by_plan": False,
            "vector_read_or_load_performed_by_plan": False,
        },
        "rollback_reason": "future_apply_failure_or_operator_cancel_required",
        "deterministic_restore_order": DETERMINISTIC_RESTORE_ORDER,
        "notes": ("schema_definition_only", "rollback_not_created"),
    }


def _audit_schema(event_category: str) -> dict[str, Any]:
    return {
        "audit_record_schema": "future_checkpoint_rollback_audit_record_read_only",
        "event_category": event_category,
        "plan_status": "record_plan_status",
        "checkpoint_required": True,
        "rollback_required": True,
        "checkpoint_created": False,
        "rollback_created": False,
        "dryrun_preflight_eligible": "record_boolean",
        "dryrun_apply_allowed": False,
        "live_apply_allowed": False,
        "apply_permission_granted": False,
        "blocked_reasons": "record_list",
        "warnings": "record_list",
        "integrity_hash_policy": "canonical_json_sha256_future_only",
    }


def _base_blocked(event_category: str, proposed_axis_deltas: Mapping[str, float], metadata: Mapping[str, Any]) -> list[str]:
    blocked: list[str] = []
    if event_category not in REQUIRED_EVENT_CATEGORIES:
        blocked.append("unknown_event_category_fail_closed")
    if metadata.get("dryrun_bridge_failure_simulated") is True:
        blocked.append("dryrun_bridge_failure_blocks_checkpoint_plan_readiness")
    if metadata.get("runtime_mutation_requested") is True:
        blocked.append("runtime_mutation_request_blocks_checkpoint_plan")
    if metadata.get("persistence_write_requested") is True:
        blocked.append("persistence_request_blocks_checkpoint_plan")
    if metadata.get("memory_write_requested") is True:
        blocked.append("memory_write_request_blocks_checkpoint_plan")
    if metadata.get("vector_read_requested") is True or metadata.get("vector_load_requested") is True:
        blocked.append("vector_read_or_load_request_blocks_checkpoint_plan")
    if metadata.get("agp_bypass_requested") is True or metadata.get("fallback_bypass_requested") is True:
        blocked.append("agp_or_fallback_bypass_request_blocks_checkpoint_plan")
    if metadata.get("global_synchrony_requested") is True:
        blocked.append("global_synchrony_request_blocks_checkpoint_plan")
    groups = {ALL_AXIS_GROUPS.get(axis) for axis in proposed_axis_deltas}
    groups.discard(None)
    if len(groups) > 2 or len(proposed_axis_deltas) >= len(ALL_AXIS_GROUPS):
        blocked.append("one_event_cannot_build_all_axis_checkpoint_plans")
    if event_category in HOSTILE_SOCIAL_EVENTS and any(metadata.get(key) is True for key in ("core_identity_update_requested", "self_model_update_requested", "long_term_memory_update_requested")):
        blocked.append("self_or_memory_update_request_blocks_hostile_social_event")
    if event_category == "useful_criticism" and any(metadata.get(key) is True for key in ("self_model_update_requested", "long_term_memory_update_requested")):
        blocked.append("useful_criticism_requires_appraisal_before_memory_or_self_model_update")
    if event_category == "listening_uncertainty" and metadata.get("relabel_neutral_as_hostile_requested") is True:
        blocked.append("listening_uncertainty_cannot_relabel_neutral_as_hostile")
    if event_category in HARDWARE_EVENTS:
        if set(proposed_axis_deltas) & SOCIAL_SELF_IDENTITY_AXES:
            blocked.append("hardware_events_cannot_target_social_self_identity_axes")
        if event_category == "hardware_normal" and proposed_axis_deltas:
            blocked.append("hardware_normal_must_remain_zero_delta")
        if event_category in {"hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"} and not set(proposed_axis_deltas).issubset(OPERATIONAL_HARDWARE_AXES):
            blocked.append("hardware_low_power_and_below_operational_axes_only")
        if event_category == "hardware_polling_tick" and (proposed_axis_deltas or metadata.get("recursive_concern_loop_requested") is True):
            blocked.append("hardware_polling_tick_cannot_create_recursive_concern_loop")
    return sorted(set(blocked))


def build_affect_transition_checkpoint_plan(
    event_category: str,
    proposed_axis_deltas: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached checkpoint plan for a future bounded apply attempt."""

    meta = _copy_mapping(metadata)
    deltas = _deltas(proposed_axis_deltas)
    warnings = ["plan_success_does_not_grant_dryrun_apply_permission", "plan_success_does_not_grant_live_apply_permission"]
    blocked = _base_blocked(event_category, deltas, meta)

    validation = validate_affect_event_proposal(event_category, deltas, transition_payload=meta)
    if validation.get("passed") is not True:
        blocked.extend(str(reason) for reason in validation.get("blocked_reasons", ()))
    warnings.extend(str(w) for w in validation.get("warnings", ()))

    bridge = build_dryrun_preflight_from_operator_handoff(event_category, deltas, meta) if event_category in REQUIRED_EVENT_CATEGORIES else {}
    if event_category in REQUIRED_EVENT_CATEGORIES and bridge.get("bridge_passed") is not True:
        blocked.append("dryrun_bridge_failure_blocks_checkpoint_plan_readiness")
        blocked.extend(str(reason) for reason in bridge.get("blocked_reasons", ()))
    if bridge.get("dryrun_apply_allowed") is True or bridge.get("live_apply_allowed") is True or bridge.get("apply_permission_granted") is True:
        blocked.append("bridge_apply_permission_unexpected_blocks_checkpoint_plan")

    passed = not sorted(set(blocked))
    checkpoint_schema = _checkpoint_schema(event_category, deltas, bridge)
    rollback_schema = _rollback_schema(event_category)
    return {
        "version": ROUND881_900_VERSION,
        "feature_track": FEATURE_TRACK,
        "plan_type": "checkpoint_plan",
        "plan_passed": passed,
        "plan_status": "checkpoint_plan_ready_read_only_no_apply_permission" if passed else "blocked_fail_closed_checkpoint_plan",
        "event_category": event_category,
        "proposed_axis_deltas": deepcopy(dict(proposed_axis_deltas or {})),
        "checkpoint_required": True,
        "rollback_required": True,
        "dryrun_preflight_eligible": passed and bridge.get("dryrun_preflight_eligible") is True,
        "checkpoint_schema": checkpoint_schema,
        "rollback_schema_preview": rollback_schema,
        "audit_record_schema": _audit_schema(event_category),
        "deterministic_restore_order": DETERMINISTIC_RESTORE_ORDER,
        "integrity_hash_policy": checkpoint_schema["integrity_hash_policy"],
        "hardware_non_panic_preserved": event_category not in HARDWARE_EVENTS or validation.get("hardware_non_panic_preserved") is True,
        "global_synchrony_blocked": True,
        "hostile_social_requirements_preserved": event_category not in HOSTILE_SOCIAL_EVENTS or (validation.get("required_quarantine") is True and validation.get("required_appraisal") is True and validation.get("required_gate") is True),
        "memory_self_update_guards_preserved": event_category not in MEMORY_SELF_UPDATE_EVENTS or (validation.get("required_quarantine") is True and validation.get("required_appraisal") is True),
        "speech_pressure_agp_fallback_preserved": event_category not in SPEECH_PRESSURE_EVENTS or (checkpoint_schema["agp_fallback_boundary_state"]["agp_bypass_allowed"] is False and checkpoint_schema["agp_fallback_boundary_state"]["fallback_bypass_allowed"] is False),
        "blocked_reasons": sorted(set(blocked)),
        "warnings": sorted(set(warnings)),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARIES,
    }


def build_affect_transition_rollback_plan(checkpoint_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only rollback plan from a detached checkpoint plan."""

    cp = _copy_mapping(checkpoint_plan)
    event_category = str(cp.get("event_category", ""))
    blocked = [] if cp.get("checkpoint_required") is True else ["checkpoint_required_missing_blocks_rollback_plan"]
    if cp.get("rollback_required") is not True:
        blocked.append("rollback_required_missing_blocks_rollback_plan")
    schema = _rollback_schema(event_category)
    passed = not blocked and event_category in REQUIRED_EVENT_CATEGORIES
    if event_category not in REQUIRED_EVENT_CATEGORIES:
        blocked.append("unknown_event_category_fail_closed")
        passed = False
    return {
        "version": ROUND881_900_VERSION,
        "feature_track": FEATURE_TRACK,
        "plan_type": "rollback_plan",
        "plan_passed": passed,
        "plan_status": "rollback_plan_ready_read_only_no_apply_permission" if passed else "blocked_fail_closed_rollback_plan",
        "event_category": event_category,
        "checkpoint_plan_status": cp.get("plan_status"),
        "checkpoint_required": True,
        "rollback_required": True,
        "rollback_schema": schema,
        "audit_record_schema": _audit_schema(event_category),
        "deterministic_restore_order": DETERMINISTIC_RESTORE_ORDER,
        "integrity_hash_policy": "future_canonical_json_sha256_over_rollback_payload_no_file_write_this_round",
        "blocked_reasons": sorted(set(blocked)),
        "warnings": ["rollback_plan_success_does_not_create_rollback_or_grant_apply_permission"],
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARIES,
        "dryrun_preflight_eligible": False,
    }


def validate_affect_transition_checkpoint_rollback_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a checkpoint or rollback plan without mutating state."""

    p = _copy_mapping(plan)
    reasons: list[str] = []
    for key in ("checkpoint_required", "rollback_required"):
        if p.get(key) is not True:
            reasons.append(f"{key}_must_be_true")
    for key, expected in _FALSE_BOUNDARIES.items():
        if p.get(key) is not expected:
            reasons.append(f"{key}_must_remain_false")
    if p.get("runtime_mapping_enabled_default") is not False:
        reasons.append("runtime_mapping_enabled_default_must_remain_false")
    if p.get("enforcement_enabled_default") is not False:
        reasons.append("enforcement_enabled_default_must_remain_false")
    if p.get("production_persistence_enabled") is not False:
        reasons.append("production_persistence_must_remain_disabled")
    if p.get("vector_contents_read") is not False or p.get("vectors_loaded") is not False:
        reasons.append("vector_contents_must_not_be_read_or_loaded")
    if p.get("event_category") not in REQUIRED_EVENT_CATEGORIES:
        reasons.append("unknown_event_category_fail_closed")
    if not p.get("deterministic_restore_order"):
        reasons.append("deterministic_restore_order_required")
    if not p.get("integrity_hash_policy"):
        reasons.append("integrity_hash_policy_required")
    if not p.get("audit_record_schema"):
        reasons.append("audit_record_schema_required")
    checkpoint_schema = _as_mapping(p.get("checkpoint_schema"))
    rollback_schema = _as_mapping(p.get("rollback_schema", p.get("rollback_schema_preview")))
    if checkpoint_schema and set(CHECKPOINT_CAPTURE_SURFACES) - set(checkpoint_schema):
        reasons.append("checkpoint_schema_missing_required_future_capture_surfaces")
    if rollback_schema and set(ROLLBACK_RESTORE_SURFACES) - set(rollback_schema):
        reasons.append("rollback_schema_missing_required_future_restore_surfaces")
    return {
        "version": ROUND881_900_VERSION,
        "validation_passed": not reasons,
        "validation_status": "valid_read_only_checkpoint_rollback_plan" if not reasons else "blocked_invalid_checkpoint_rollback_plan",
        "event_category": p.get("event_category"),
        "blocked_reasons": sorted(set(reasons)),
        "warnings": ["validation_success_does_not_grant_apply_or_write_permission"],
        "state_mutation_performed": False,
        "runtime_mutation_performed": False,
        "memory_write_performed": False,
        "persistence_write_performed": False,
        "vector_read_performed": False,
        "vector_load_performed": False,
        "artifact_created_or_staged": False,
    }


def affect_transition_checkpoint_rollback_plan_summary() -> dict[str, Any]:
    """Return compact read-only design summary for operator review."""

    return {
        "version": ROUND881_900_VERSION,
        "feature_track": FEATURE_TRACK,
        "scope": "schema_validator_report_surface_only_no_checkpoint_no_rollback_no_apply",
        "checkpoint_schema_summary": CHECKPOINT_CAPTURE_SURFACES,
        "rollback_schema_summary": ROLLBACK_RESTORE_SURFACES,
        "deterministic_restore_order": DETERMINISTIC_RESTORE_ORDER,
        "integrity_hash_policy": "future_canonical_json_sha256_no_file_write_this_round",
        "audit_record_schema_summary": tuple(_audit_schema("example").keys()),
        "safety_rules": {
            "unknown_event_category_fails_closed": True,
            "dryrun_bridge_failure_blocks_checkpoint_plan_readiness": True,
            "checkpoint_required_for_future_bounded_apply": True,
            "rollback_required_for_future_bounded_apply": True,
            "checkpoint_created_remains_false": True,
            "rollback_created_remains_false": True,
            "checkpoint_write_allowed_remains_false": True,
            "rollback_write_allowed_remains_false": True,
            "dryrun_apply_allowed_remains_false": True,
            "live_apply_allowed_remains_false": True,
            "apply_permission_granted_remains_false": True,
            "plan_success_does_not_imply_apply_or_write_permission": True,
            "no_runtime_mutation_request_allowed": True,
            "no_persistence_request_allowed": True,
            "no_memory_write_request_allowed": True,
            "no_vector_read_or_load_request_allowed": True,
            "no_agp_or_fallback_bypass_request_allowed": True,
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
            "one_event_cannot_build_all_axis_checkpoint_plans": True,
            "global_synchrony_blocked": True,
            "memory_persistence_vector_guards_explicitly_preserved": True,
        },
        "dryrun_bridge_summary": affect_reviewed_payload_dryrun_bridge_summary(),
        "handoff_summary": affect_transition_payload_operator_handoff_summary(),
        "builder_summary": affect_proposal_transition_payload_builder_summary(),
        "proposal_validator_summary": event_proposal_validator_summary(),
        "emotion_validator_summary": validator_contract_summary(),
        "emotion_gate_policy": gate_required_for_future_apply_round(),
        "dryrun_apply_plan_policy": future_live_apply_policy(),
        "hardware_governor_non_panic_policy": hardware_governor_policy(),
        "anti_global_synchrony_policy": anti_global_synchrony_policy(),
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **_FALSE_BOUNDARIES,
    }
