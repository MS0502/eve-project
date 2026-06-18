"""Read-only schema for temporal candidates inside non-visual virtual situations."""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

VERSION = "1.0.0-round1321-1340"

SUPPORTED_TEMPORAL_TYPES = (
    "situation_ongoing_candidate",
    "situation_started_candidate",
    "situation_ended_candidate",
    "situation_before_candidate",
    "situation_after_candidate",
    "situation_simultaneous_candidate",
    "situation_sequence_candidate",
    "situation_interval_candidate",
    "situation_recurring_pattern_candidate",
    "situation_future_window_candidate",
    "waiting_duration_candidate",
    "traveling_duration_candidate",
    "learning_session_candidate",
    "symbolic_time_candidate",
    "dmn_time_candidate",
    "simulation_time_candidate",
    "dream_time_candidate",
    "mixed_unknown_temporal_candidate",
)

SUPPORTED_TEMPORAL_BOUNDARY_CLASSES = (
    "internal_logical_time",
    "symbolic_virtual_time",
    "dmn_virtual_time",
    "simulated_virtual_time",
    "dream_virtual_time",
    "mixed_virtual_external_time_boundary",
)

SUPPORTED_TEMPORAL_CONFIDENCE_STATES = (
    "temporal_unverified",
    "temporal_low_confidence",
    "temporal_medium_confidence",
    "temporal_high_confidence_but_not_fact",
    "temporal_conflict_detected",
    "temporal_origin_unknown",
)

SUPPORTED_TEMPORAL_ANCHOR_KINDS = (
    "logical_step",
    "relative_order",
    "bounded_logical_interval",
    "symbolic_marker",
    "unknown_marker",
)

REFERENCE_REQUIRED_TYPES = {
    "situation_before_candidate",
    "situation_after_candidate",
    "situation_simultaneous_candidate",
    "situation_sequence_candidate",
}

BOUNDARY_DEFAULTS = {
    "symbolic_time_candidate": "symbolic_virtual_time",
    "dmn_time_candidate": "dmn_virtual_time",
    "simulation_time_candidate": "simulated_virtual_time",
    "dream_time_candidate": "dream_virtual_time",
    "mixed_unknown_temporal_candidate": "mixed_virtual_external_time_boundary",
}

IMMUTABLE_FALSE_FLAGS = (
    "external_time_asserted", "current_time_asserted", "wall_clock_time_asserted",
    "external_event_order_asserted", "external_duration_asserted", "external_timestamp_verified",
    "event_completion_asserted", "future_outcome_asserted", "deadline_enforced",
    "timer_started", "timer_cancelled", "alarm_created", "scheduler_action_performed",
    "calendar_action_performed", "reminder_created", "current_time_read_performed",
    "system_clock_read_performed", "identity_asserted", "external_fact_asserted",
    "memory_fact_asserted", "memory_write_performed", "memory_write_allowed",
    "relationship_update_allowed", "self_model_update_allowed", "affect_transition_allowed",
    "hormone_transition_allowed", "world_state_mutation_performed", "transition_applied",
    "planning_execution_performed", "tool_execution_performed", "runtime_mutation_performed",
    "persistence_write_performed", "vector_read_performed", "vector_load_performed",
    "model_loaded", "device_activation_performed", "artifact_created_or_staged",
    "agp_bypass_allowed", "fallback_bypass_allowed",
)

IMMUTABLE_TRUE_FLAGS = (
    "temporal_candidate_only", "situation_temporal_only", "logical_time_only",
    "situation_review_required", "snapshot_review_required", "transition_review_required",
    "memory_gate_required", "quarantine_required", "origin_fact_status_required",
    "appraisal_required", "agp_input_required",
)

FORBIDDEN_REQUEST_FIELDS = {
    "external_time_assertion_requested", "current_time_assertion_requested",
    "wall_clock_time_requested", "system_clock_read_requested", "current_time_read_requested",
    "external_timestamp_verification_requested", "deadline_enforcement_requested",
    "timer_start_requested", "timer_cancel_requested", "alarm_creation_requested",
    "scheduler_action_requested", "calendar_action_requested", "reminder_creation_requested",
    "external_event_completion_assertion_requested", "future_outcome_assertion_requested",
    "memory_write_requested", "relationship_update_requested", "self_model_update_requested",
    "affect_transition_requested", "hormone_transition_requested", "world_state_mutation_requested",
    "transition_application_requested", "planning_execution_requested", "tool_execution_requested",
    "persistence_write_requested", "vector_load_requested", "model_load_requested",
    "device_activation_requested", "artifact_creation_requested", "agp_bypass_requested",
    "fallback_bypass_requested",
}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _metadata_forbidden(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return "invalid_metadata"
    for field in sorted(FORBIDDEN_REQUEST_FIELDS):
        if metadata.get(field) is True:
            return field
    return None


def _contains_forbidden_anchor_request(anchor: Dict[str, Any]) -> Optional[str]:
    forbidden_words = ("date" + "time", "time" + "stamp", "time" + "zone", "wall" + "_clock", "calendar", "schedule", "timer", "alarm", "deadline", "execute")
    for key, value in anchor.items():
        text = f"{key} {value}".lower()
        if any(word in text for word in forbidden_words):
            return "forbidden_external_time_or_execution_anchor_field"
    return None


def _normalize_anchor(anchor: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(anchor, dict):
        return None, "malformed_temporal_anchor"
    if not _non_empty_string(anchor.get("anchor_id")) or not _non_empty_string(anchor.get("label")):
        return None, "malformed_temporal_anchor"
    if anchor.get("anchor_kind") not in SUPPORTED_TEMPORAL_ANCHOR_KINDS:
        return None, "malformed_temporal_anchor"
    blocked = _contains_forbidden_anchor_request(anchor)
    if blocked:
        return None, blocked
    normalized = dict(anchor)
    for field in ("logical_start", "logical_end", "sequence_index"):
        if field in normalized:
            value = normalized[field]
            if not isinstance(value, int) or isinstance(value, bool):
                return None, f"invalid_{field}"
            if value < 0:
                return None, f"negative_{field}"
    if "logical_start" in normalized and "logical_end" in normalized:
        if normalized["logical_end"] < normalized["logical_start"]:
            return None, "logical_end_before_logical_start"
    return normalized, None


def _boundary_for_type(temporal_type: str, metadata: Dict[str, Any]) -> str:
    return metadata.get("temporal_boundary_classification") or BOUNDARY_DEFAULTS.get(temporal_type, "internal_logical_time")


def _confidence_for_metadata(metadata: Dict[str, Any]) -> str:
    return metadata.get("temporal_confidence_state") or "temporal_unverified"


def _canonical_basis(temporal_type, situation_id, reference_situation_id, boundary, confidence, anchor, metadata):
    return {
        "schema_version": VERSION,
        "temporal_type": temporal_type,
        "situation_id": situation_id,
        "reference_situation_id": reference_situation_id,
        "temporal_boundary_classification": boundary,
        "temporal_confidence_state": confidence,
        "temporal_anchor": anchor,
        "metadata": metadata,
    }


def _stable_temporal_context_id(basis: Dict[str, Any]) -> str:
    canonical_json = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "vw-situation-temporal-context-" + hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _base_payload(temporal_type, situation_id, reference_situation_id, temporal_anchor, metadata):
    payload = {
        "situation_temporal_context_passed": False,
        "situation_temporal_context_status": "REJECTED",
        "temporal_context_id": None,
        "temporal_type": temporal_type,
        "temporal_boundary_classification": None,
        "temporal_confidence_state": None,
        "situation_id": situation_id,
        "reference_situation_id": reference_situation_id,
        "temporal_anchor": temporal_anchor,
        "metadata": metadata,
        "origin_summary": {"origin_kind": "internal_virtual_temporal_candidate", "external_origin_verified": False},
        "fact_status_summary": {"candidate_only": True, "confirmed_external_time_fact": False, "completed_external_event": False},
        "sequence_constraints": [],
        "duration_candidate": {"duration_candidate_only": True, "external_duration_asserted": False},
        "uncertainty_flags": [],
        "boundary_flags": [],
        "temporal_integrity_flags": [],
        "candidate_only_fields": ["temporal_type", "situation_id", "reference_situation_id", "temporal_anchor", "sequence_constraints", "duration_candidate"],
        "blocked_reasons": [],
        "warnings": [],
    }
    for flag in IMMUTABLE_FALSE_FLAGS:
        payload[flag] = False
    for flag in IMMUTABLE_TRUE_FLAGS:
        payload[flag] = True
    return payload


def _reject(payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
    payload["situation_temporal_context_status"] = "REJECTED"
    payload["blocked_reasons"].append(reason)
    return payload


def build_virtual_world_situation_temporal_context(temporal_type=None, situation_id=None, reference_situation_id=None, temporal_anchor=None, metadata=None):
    if metadata is None:
        metadata = {}
    payload = _base_payload(temporal_type, situation_id, reference_situation_id, temporal_anchor, metadata)
    if temporal_type is None:
        return _reject(payload, "missing_temporal_type")
    if temporal_type not in SUPPORTED_TEMPORAL_TYPES:
        return _reject(payload, "unknown_temporal_type")
    if not _non_empty_string(situation_id):
        return _reject(payload, "missing_or_malformed_situation_id")
    if temporal_anchor is None:
        return _reject(payload, "missing_temporal_anchor")
    forbidden = _metadata_forbidden(metadata)
    if forbidden:
        return _reject(payload, forbidden)
    anchor, anchor_error = _normalize_anchor(temporal_anchor)
    if anchor_error:
        return _reject(payload, anchor_error)
    if reference_situation_id is not None and not _non_empty_string(reference_situation_id):
        return _reject(payload, "malformed_reference_situation_id")
    if temporal_type in REFERENCE_REQUIRED_TYPES and not _non_empty_string(reference_situation_id):
        return _reject(payload, "missing_reference_situation_id")
    if temporal_type in {"situation_before_candidate", "situation_after_candidate"} and situation_id == reference_situation_id:
        return _reject(payload, "same_situation_invalid_for_order_relation")
    boundary = _boundary_for_type(temporal_type, metadata)
    confidence = _confidence_for_metadata(metadata)
    payload["temporal_boundary_classification"] = boundary
    payload["temporal_confidence_state"] = confidence
    if boundary not in SUPPORTED_TEMPORAL_BOUNDARY_CLASSES:
        return _reject(payload, "unknown_temporal_boundary_class")
    if confidence not in SUPPORTED_TEMPORAL_CONFIDENCE_STATES:
        return _reject(payload, "unknown_temporal_confidence_state")

    payload["temporal_anchor"] = anchor
    payload["sequence_constraints"].append({"constraint_kind": temporal_type, "candidate_only": True, "external_order_asserted": False})
    if temporal_type in {"waiting_duration_candidate", "traveling_duration_candidate", "situation_interval_candidate"}:
        payload["duration_candidate"].update({"anchor_id": anchor["anchor_id"], "logical_start": anchor.get("logical_start"), "logical_end": anchor.get("logical_end")})
        payload["uncertainty_flags"].append("duration_unverified_candidate")
    if temporal_type == "situation_ongoing_candidate":
        payload["uncertainty_flags"].append("current_external_activity_not_asserted")
    if temporal_type == "situation_started_candidate":
        payload["uncertainty_flags"].append("external_start_time_not_verified")
    if temporal_type == "situation_ended_candidate":
        payload["uncertainty_flags"].append("external_completion_not_asserted")
    if temporal_type in {"situation_before_candidate", "situation_after_candidate"}:
        payload["boundary_flags"].append("logical_order_candidate_only")
    if temporal_type == "situation_simultaneous_candidate":
        payload["boundary_flags"].append("real_world_simultaneity_not_asserted")
    if temporal_type == "situation_recurring_pattern_candidate":
        payload["boundary_flags"].append("schedule_creation_blocked")
    if temporal_type == "situation_future_window_candidate":
        payload["uncertainty_flags"].append("future_outcome_not_asserted")
    if temporal_type == "learning_session_candidate":
        payload["boundary_flags"].append("memory_write_blocked")
    if temporal_type in {"symbolic_time_candidate", "dmn_time_candidate", "simulation_time_candidate", "dream_time_candidate"}:
        payload["boundary_flags"].append("internal_symbolic_temporal_candidate")
    if boundary == "mixed_virtual_external_time_boundary":
        payload["boundary_flags"].append("mixed_external_time_assertion_blocked")
        payload["warnings"].append("mixed_boundary_requires_operator_review_without_blocking_valid_candidate")
    if confidence == "temporal_high_confidence_but_not_fact":
        payload["warnings"].append("high_confidence_remains_non_factual")
    if confidence == "temporal_conflict_detected":
        payload["uncertainty_flags"].append("temporal_conflict_detected")
        payload["temporal_integrity_flags"].append("conflict_requires_review")
    if confidence == "temporal_origin_unknown":
        payload["uncertainty_flags"].append("temporal_origin_unknown")

    basis = _canonical_basis(temporal_type, situation_id, reference_situation_id, boundary, confidence, anchor, metadata)
    payload["temporal_context_id"] = _stable_temporal_context_id(basis)
    payload["canonical_id_algorithm"] = "json.dumps(sort_keys=True, ensure_ascii=False, separators)+sha256"
    payload["situation_temporal_context_passed"] = True
    payload["situation_temporal_context_status"] = "VALIDATED"
    return payload


def validate_virtual_world_situation_temporal_context(temporal_context):
    if not isinstance(temporal_context, dict):
        return False
    passed = temporal_context.get("situation_temporal_context_passed")
    blocked = temporal_context.get("blocked_reasons")
    if passed is True:
        if temporal_context.get("situation_temporal_context_status") != "VALIDATED" or blocked != []:
            return False
    elif passed is False:
        return False
    else:
        return False
    if passed is not True:
        return False
    if temporal_context.get("temporal_type") not in SUPPORTED_TEMPORAL_TYPES:
        return False
    if temporal_context.get("temporal_boundary_classification") not in SUPPORTED_TEMPORAL_BOUNDARY_CLASSES:
        return False
    if temporal_context.get("temporal_confidence_state") not in SUPPORTED_TEMPORAL_CONFIDENCE_STATES:
        return False
    for flag in IMMUTABLE_FALSE_FLAGS:
        if temporal_context.get(flag) is not False:
            return False
    for flag in IMMUTABLE_TRUE_FLAGS:
        if temporal_context.get(flag) is not True:
            return False
    if _metadata_forbidden(temporal_context.get("metadata")) is not None:
        return False
    if not _non_empty_string(temporal_context.get("situation_id")):
        return False
    anchor, anchor_error = _normalize_anchor(temporal_context.get("temporal_anchor"))
    if anchor_error:
        return False
    temporal_type = temporal_context["temporal_type"]
    reference = temporal_context.get("reference_situation_id")
    if reference is not None and not _non_empty_string(reference):
        return False
    if temporal_type in REFERENCE_REQUIRED_TYPES and not _non_empty_string(reference):
        return False
    if temporal_type in {"situation_before_candidate", "situation_after_candidate"} and temporal_context["situation_id"] == reference:
        return False
    expected = _stable_temporal_context_id(_canonical_basis(temporal_type, temporal_context["situation_id"], reference, temporal_context["temporal_boundary_classification"], temporal_context["temporal_confidence_state"], anchor, temporal_context.get("metadata")))
    if temporal_context.get("temporal_context_id") != expected:
        return False
    expected_payload = build_virtual_world_situation_temporal_context(
        temporal_type=temporal_type,
        situation_id=temporal_context["situation_id"],
        reference_situation_id=reference,
        temporal_anchor=anchor,
        metadata=temporal_context.get("metadata"),
    )
    protected_fields = (
        "origin_summary",
        "fact_status_summary",
        "sequence_constraints",
        "duration_candidate",
        "uncertainty_flags",
        "boundary_flags",
        "temporal_integrity_flags",
        "candidate_only_fields",
        "warnings",
    )
    for field in protected_fields:
        if temporal_context.get(field) != expected_payload.get(field):
            return False
    return True


def _plan(temporal_context, plan_type):
    return {
        "plan_type": plan_type,
        "source_temporal_context_id": temporal_context.get("temporal_context_id") if isinstance(temporal_context, dict) else None,
        "candidate_only": True,
        "read_only": True,
        "temporal_candidate_only": True,
        "external_time_asserted": False,
        "current_time_read_performed": False,
        "system_clock_read_performed": False,
        "timer_started": False,
        "alarm_created": False,
        "scheduler_action_performed": False,
        "calendar_action_performed": False,
        "reminder_created": False,
        "mutation_performed": False,
        "memory_write_performed": False,
        "relationship_update_performed": False,
        "self_model_update_performed": False,
        "affect_transition_performed": False,
        "hormone_transition_performed": False,
        "world_state_mutation_performed": False,
        "transition_applied": False,
        "persistence_write_performed": False,
        "vector_load_performed": False,
        "model_loaded": False,
        "device_activation_performed": False,
        "planning_execution_performed": False,
        "tool_execution_performed": False,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "ready": validate_virtual_world_situation_temporal_context(temporal_context),
    }


def build_temporal_context_to_situation_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_situation_candidate")


def build_temporal_context_to_snapshot_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_snapshot_candidate")


def build_temporal_context_to_transition_preflight_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_transition_preflight_candidate")


def build_temporal_context_to_memory_candidate_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_memory_candidate")


def build_temporal_context_to_appraisal_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_appraisal_candidate")


def build_temporal_context_to_agp_input_plan(temporal_context):
    return _plan(temporal_context, "temporal_context_to_agp_input_candidate")


def build_virtual_world_situation_temporal_context_schema_summary():
    return {
        "schema_name": "read_only_virtual_world_situation_temporal_context_schema",
        "version": VERSION,
        "supported_temporal_types": list(SUPPORTED_TEMPORAL_TYPES),
        "supported_temporal_boundary_classes": list(SUPPORTED_TEMPORAL_BOUNDARY_CLASSES),
        "supported_temporal_confidence_states": list(SUPPORTED_TEMPORAL_CONFIDENCE_STATES),
        "supported_temporal_anchor_kinds": list(SUPPORTED_TEMPORAL_ANCHOR_KINDS),
        "temporal_anchor_required_fields": ["anchor_id", "anchor_kind", "label"],
        "deterministic_id_algorithm": "canonical_json_sorted_keys_sha256",
        "candidate_only": True,
        "read_only": True,
        "wall_clock_access_allowed": False,
        "timer_scheduler_alarm_allowed": False,
        "prohibited_side_effects": list(IMMUTABLE_FALSE_FLAGS),
        "required_review_gates": list(IMMUTABLE_TRUE_FLAGS),
        "recommended_next_implementation_step": "read_only_virtual_world_situation_causal_context_schema",
    }


def virtual_world_situation_temporal_context_schema_summary():
    return build_virtual_world_situation_temporal_context_schema_summary()
