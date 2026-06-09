import uuid
from typing import Dict, Any, Optional

VERSION = "1.0.0"

SUPPORTED_TRANSITION_SOURCE_TYPES = {
    "internal_virtual_action_transition_candidate",
    "internal_virtual_object_transition_candidate",
    "internal_virtual_agent_transition_candidate",
    "internal_virtual_location_transition_candidate",
    "internal_virtual_relationship_scene_transition_candidate",
    "symbolic_virtual_transition_candidate",
    "dmn_virtual_transition_candidate",
    "simulation_virtual_transition_candidate",
    "dream_virtual_transition_candidate",
    "operator_supplied_virtual_transition_candidate"
}

SUPPORTED_TRANSITION_BOUNDARY_CLASSES = {
    "internal_virtual_transition",
    "symbolic_virtual_transition",
    "dmn_virtual_transition",
    "simulated_virtual_transition",
    "dream_virtual_transition",
    "mixed_virtual_external_transition_boundary"
}

SUPPORTED_TRANSITION_CONFIDENCE_STATES = {
    "transition_unverified",
    "transition_low_confidence",
    "transition_medium_confidence",
    "transition_high_confidence_but_not_external_fact",
    "transition_conflict_detected",
    "transition_origin_unknown"
}

SUPPORTED_TRANSITION_DECISIONS = {
    "transition_future_review_allowed",
    "transition_review_required",
    "transition_blocked_missing_snapshot",
    "transition_blocked_missing_transition",
    "transition_blocked_unknown_source",
    "transition_blocked_unknown_boundary",
    "transition_blocked_unknown_confidence",
    "transition_blocked_external_fact_assertion",
    "transition_blocked_identity_assertion",
    "transition_blocked_relationship_assertion",
    "transition_blocked_self_model_update",
    "transition_blocked_memory_write",
    "transition_blocked_persistence_write",
    "transition_blocked_virtual_state_mutation",
    "transition_blocked_privacy_risk",
    "transition_not_applicable"
}

def build_virtual_world_transition_preflight_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "track": "read_only_virtual_world_transition_preflight_schema",
        "supported_transition_source_types": list(SUPPORTED_TRANSITION_SOURCE_TYPES),
        "supported_transition_boundary_classes": list(SUPPORTED_TRANSITION_BOUNDARY_CLASSES),
        "supported_transition_confidence_states": list(SUPPORTED_TRANSITION_CONFIDENCE_STATES),
        "supported_transition_decisions": list(SUPPORTED_TRANSITION_DECISIONS)
    }

def virtual_world_transition_preflight_schema_summary() -> Dict[str, Any]:
    return build_virtual_world_transition_preflight_schema_summary()

def build_virtual_world_transition_preflight(
    current_snapshot: Optional[Dict[str, Any]] = None,
    proposed_transition: Optional[Dict[str, Any]] = None,
    transition_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    if current_snapshot is None:
        current_snapshot = {}
    if proposed_transition is None:
        proposed_transition = {}
    if transition_context is None:
        transition_context = {}
    if metadata is None:
        metadata = {}

    transition_source_type = metadata.get("transition_source_type", "")
    transition_boundary_classification = metadata.get("transition_boundary_classification", "")
    transition_confidence_state = metadata.get("transition_confidence_state", "")

    preflight = {
        "virtual_world_transition_preflight_passed": False,
        "virtual_world_transition_preflight_status": "failed",
        "transition_preflight_id": str(uuid.uuid4()),
        "transition_source_type": transition_source_type,
        "transition_boundary_classification": transition_boundary_classification,
        "transition_confidence_state": transition_confidence_state,
        "transition_decision": "transition_not_applicable",
        "current_snapshot": current_snapshot,
        "proposed_transition": proposed_transition,
        "transition_context": transition_context,
        "metadata": metadata,
        "origin_summary": {},
        "fact_status_summary": {},
        "internal_virtual_transition_candidate": True,
        "preflight_only": True,
        "transition_applied": False,
        "virtual_state_mutation_performed": False,
        "virtual_world_write_allowed": False,
        "virtual_world_persistence_allowed": False,
        "persistent_world_state_created": False,
        "external_reality_asserted": False,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "memory_write_performed": False,
        "memory_write_allowed": False,
        "quarantine_promotion_allowed": False,
        "self_model_update_allowed": False,
        "relationship_update_allowed": False,
        "relationship_state_asserted": False,
        "identity_asserted": False,
        "user_emotion_asserted": False,
        "user_intent_asserted": False,
        "affect_transition_allowed": False,
        "hormone_transition_allowed": False,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "vector_read_performed": False,
        "vector_load_performed": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
        "artifact_created_or_staged": False,
        "candidate_only_fields": [],
        "uncertainty_flags": [],
        "privacy_flags": [],
        "virtual_boundary_flags": [],
        "transition_integrity_flags": [],
        "blocked_reasons": [],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }

    # Fail closed checks
    if not current_snapshot:
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_empty_snapshot"
        preflight["transition_decision"] = "transition_blocked_missing_snapshot"
        preflight["blocked_reasons"].append("empty_current_snapshot")
        return preflight

    if not proposed_transition:
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_empty_transition"
        preflight["transition_decision"] = "transition_blocked_missing_transition"
        preflight["blocked_reasons"].append("empty_proposed_transition")
        return preflight

    if transition_source_type not in SUPPORTED_TRANSITION_SOURCE_TYPES:
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_unknown_source"
        preflight["transition_decision"] = "transition_blocked_unknown_source"
        preflight["blocked_reasons"].append("unknown_transition_source_type")
        return preflight

    if transition_boundary_classification not in SUPPORTED_TRANSITION_BOUNDARY_CLASSES:
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_unknown_boundary"
        preflight["transition_decision"] = "transition_blocked_unknown_boundary"
        preflight["blocked_reasons"].append("unknown_transition_boundary_classification")
        return preflight

    if transition_confidence_state not in SUPPORTED_TRANSITION_CONFIDENCE_STATES:
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_unknown_confidence"
        preflight["transition_decision"] = "transition_blocked_unknown_confidence"
        preflight["blocked_reasons"].append("unknown_transition_confidence_state")
        return preflight

    # Mixed boundary
    if transition_boundary_classification == "mixed_virtual_external_transition_boundary":
        preflight["virtual_boundary_flags"].append("mixed_boundary_detected")
        preflight["blocked_reasons"].append("mixed_virtual_external_transition_boundary_blocks_external_assertion")

    # Conflict
    if transition_confidence_state == "transition_conflict_detected":
        preflight["uncertainty_flags"].append("conflict_in_transition")
        preflight["transition_integrity_flags"].append("integrity_conflict")
        preflight["virtual_boundary_flags"].append("boundary_conflict")

    # Unknown Origin
    if transition_confidence_state == "transition_origin_unknown":
        preflight["uncertainty_flags"].append("origin_unknown")

    # Source specific rules
    if transition_source_type == "internal_virtual_action_transition_candidate":
        preflight["candidate_only_fields"].append("internal_virtual_action_transition_remains_preflight_only")

    if transition_source_type == "internal_virtual_object_transition_candidate":
        preflight["candidate_only_fields"].append("object_transition_remains_preflight_only")

    if transition_source_type == "internal_virtual_agent_transition_candidate":
        preflight["candidate_only_fields"].append("virtual_agent_transition_cannot_assert_real_person_identity")

    if transition_source_type == "internal_virtual_relationship_scene_transition_candidate":
        preflight["candidate_only_fields"].append("virtual_relationship_scene_transition_cannot_assert_relationship_state")

    if transition_source_type == "symbolic_virtual_transition_candidate":
        preflight["candidate_only_fields"].append("symbolic_transition_remains_symbolic_candidate_only")

    if transition_source_type == "dmn_virtual_transition_candidate":
        preflight["candidate_only_fields"].append("dmn_transition_remains_internal_candidate_only")

    if transition_source_type == "simulation_virtual_transition_candidate":
        preflight["candidate_only_fields"].append("simulation_transition_cannot_become_current_fact")

    if transition_source_type == "dream_virtual_transition_candidate":
        preflight["candidate_only_fields"].append("dream_transition_cannot_become_external_fact")

    if transition_source_type == "operator_supplied_virtual_transition_candidate":
        preflight["candidate_only_fields"].append("operator_supplied_transition_remains_candidate_only")

    # Check for requested blocks in metadata
    if metadata.get("privacy_risk", False):
        preflight["privacy_flags"].append("privacy_risk_detected")
        preflight["transition_decision"] = "transition_blocked_privacy_risk"
        preflight["blocked_reasons"].append("privacy_risk_blocks_future_review")
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_privacy_risk"
        return preflight

    if metadata.get("self_model_update_request", False):
        preflight["transition_decision"] = "transition_blocked_self_model_update"
        preflight["blocked_reasons"].append("self_model_update_request_blocks_future_review")
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_self_model_update_request"
        return preflight

    if metadata.get("relationship_update_request", False):
        preflight["transition_decision"] = "transition_blocked_relationship_assertion"
        preflight["blocked_reasons"].append("relationship_update_request_blocks_future_review")
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_relationship_update_request"
        return preflight

    if metadata.get("memory_write_request", False):
        preflight["transition_decision"] = "transition_blocked_memory_write"
        preflight["blocked_reasons"].append("memory_write_request_blocks_future_review")
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_memory_write_request"
        return preflight

    if metadata.get("persistence_write_request", False):
        preflight["transition_decision"] = "transition_blocked_persistence_write"
        preflight["blocked_reasons"].append("persistence_write_request_blocks_future_review")
        preflight["virtual_world_transition_preflight_passed"] = False
        preflight["virtual_world_transition_preflight_status"] = "failed_persistence_write_request"
        return preflight

    # If it reached here, review is allowed or required
    preflight["virtual_world_transition_preflight_passed"] = True
    preflight["virtual_world_transition_preflight_status"] = "passed"
    preflight["transition_decision"] = "transition_future_review_allowed"

    return preflight

def validate_virtual_world_transition_preflight(preflight: Dict[str, Any]) -> bool:
    if not isinstance(preflight, dict):
        return False
    if preflight.get("virtual_world_transition_preflight_passed") is not True:
        return False

    invariants = [
        "transition_applied",
        "virtual_state_mutation_performed",
        "virtual_world_write_allowed",
        "virtual_world_persistence_allowed",
        "persistent_world_state_created",
        "external_reality_asserted",
        "current_external_fact_asserted",
        "real_world_event_asserted",
        "memory_write_performed",
        "memory_write_allowed",
        "quarantine_promotion_allowed",
        "self_model_update_allowed",
        "relationship_update_allowed",
        "relationship_state_asserted",
        "identity_asserted",
        "user_emotion_asserted",
        "user_intent_asserted",
        "affect_transition_allowed",
        "hormone_transition_allowed",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
        "vector_read_performed",
        "vector_load_performed",
        "runtime_mutation_performed",
        "persistence_write_performed",
        "artifact_created_or_staged"
    ]
    for inv in invariants:
        if preflight.get(inv) is not False:
            return False

    positive_invariants = [
        "internal_virtual_transition_candidate",
        "preflight_only",
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "quarantine_required",
        "origin_fact_status_required"
    ]
    for inv in positive_invariants:
        if preflight.get(inv) is not True:
            return False

    return True

def build_virtual_world_transition_to_origin_fact_status_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_origin_fact_status",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "origin": "virtual_world_visual",
        "fact_status": "observed_internal_virtual",
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }

def build_virtual_world_transition_to_snapshot_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_snapshot",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }

def build_virtual_world_transition_to_memory_candidate_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_memory_candidate",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }

def build_virtual_world_transition_to_quarantine_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_quarantine",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }

def build_virtual_world_transition_to_appraisal_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_appraisal",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }

def build_virtual_world_transition_to_agp_input_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_transition_to_agp_input",
        "source_preflight_id": preflight.get("transition_preflight_id"),
        "ready": validate_virtual_world_transition_preflight(preflight) and preflight.get("virtual_world_transition_preflight_passed", False)
    }
