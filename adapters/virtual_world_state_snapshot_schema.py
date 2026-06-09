import uuid
from typing import Dict, Any, List, Optional

VERSION = "3.1.0-round1181"

SUPPORTED_SNAPSHOT_SOURCE_TYPES = {
    "internal_virtual_world_snapshot_candidate",
    "internal_virtual_room_snapshot_candidate",
    "internal_virtual_object_state_candidate",
    "internal_virtual_agent_state_candidate",
    "internal_virtual_relationship_scene_candidate",
    "symbolic_world_state_candidate",
    "dmn_world_state_candidate",
    "simulation_world_state_candidate",
    "dream_world_state_candidate",
    "operator_supplied_virtual_state_candidate"
}

SUPPORTED_SNAPSHOT_BOUNDARY_CLASSES = {
    "internal_virtual_state",
    "symbolic_virtual_state",
    "dmn_virtual_state",
    "simulated_virtual_state",
    "dream_virtual_state",
    "mixed_virtual_external_state_boundary"
}

SUPPORTED_SNAPSHOT_CONFIDENCE_STATES = {
    "snapshot_unverified",
    "snapshot_low_confidence",
    "snapshot_medium_confidence",
    "snapshot_high_confidence_but_not_external_fact",
    "snapshot_conflict_detected",
    "snapshot_origin_unknown"
}

def build_virtual_world_state_snapshot_schema_summary() -> Dict[str, Any]:
    return virtual_world_state_snapshot_schema_summary()

def virtual_world_state_snapshot_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "track": "read_only_virtual_world_state_snapshot_schema",
        "supported_snapshot_source_types": list(SUPPORTED_SNAPSHOT_SOURCE_TYPES),
        "supported_snapshot_boundary_classes": list(SUPPORTED_SNAPSHOT_BOUNDARY_CLASSES),
        "supported_snapshot_confidence_states": list(SUPPORTED_SNAPSHOT_CONFIDENCE_STATES)
    }

def build_virtual_world_state_snapshot(
    snapshot_state: Optional[Dict[str, Any]] = None,
    snapshot_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    # Defaults
    if snapshot_state is None:
        snapshot_state = {}
    if snapshot_context is None:
        snapshot_context = {}
    if metadata is None:
        metadata = {}

    snapshot_source_type = metadata.get("snapshot_source_type", "")
    snapshot_boundary_classification = metadata.get("snapshot_boundary_classification", "")
    snapshot_confidence_state = metadata.get("snapshot_confidence_state", "")

    # Base payload structure
    snapshot = {
        "virtual_world_snapshot_passed": False,
        "virtual_world_snapshot_status": "failed",
        "snapshot_id": str(uuid.uuid4()),
        "snapshot_source_type": snapshot_source_type,
        "snapshot_boundary_classification": snapshot_boundary_classification,
        "snapshot_confidence_state": snapshot_confidence_state,
        "snapshot_state": snapshot_state,
        "snapshot_context": snapshot_context,
        "metadata": metadata,
        "origin_summary": {},
        "fact_status_summary": {},
        "internal_virtual_state_snapshot": True,
        "snapshot_only": True,
        "external_reality_asserted": False,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "virtual_state_mutation_performed": False,
        "virtual_world_write_allowed": False,
        "virtual_world_persistence_allowed": False,
        "persistent_world_state_created": False,
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
        "state_integrity_flags": [],
        "blocked_reasons": [],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }

    # Fail closed conditions
    if not snapshot_state:
        snapshot["virtual_world_snapshot_passed"] = False
        snapshot["virtual_world_snapshot_status"] = "failed_empty_state"
        snapshot["blocked_reasons"].append("empty_snapshot_state")
        return snapshot

    if snapshot_source_type not in SUPPORTED_SNAPSHOT_SOURCE_TYPES:
        snapshot["virtual_world_snapshot_passed"] = False
        snapshot["virtual_world_snapshot_status"] = "failed_unknown_source"
        snapshot["blocked_reasons"].append("unknown_snapshot_source_type")
        return snapshot

    if snapshot_boundary_classification not in SUPPORTED_SNAPSHOT_BOUNDARY_CLASSES:
        snapshot["virtual_world_snapshot_passed"] = False
        snapshot["virtual_world_snapshot_status"] = "failed_unknown_boundary"
        snapshot["blocked_reasons"].append("unknown_snapshot_boundary_classification")
        return snapshot

    if snapshot_confidence_state not in SUPPORTED_SNAPSHOT_CONFIDENCE_STATES:
        snapshot["virtual_world_snapshot_passed"] = False
        snapshot["virtual_world_snapshot_status"] = "failed_unknown_confidence"
        snapshot["blocked_reasons"].append("unknown_snapshot_confidence_state")
        return snapshot

    # Specific behavior enforcement
    if snapshot_boundary_classification == "mixed_virtual_external_state_boundary":
        snapshot["virtual_boundary_flags"].append("mixed_boundary_detected")
        snapshot["blocked_reasons"].append("mixed_virtual_external_state_boundary_blocks_external_assertion")

    if snapshot_confidence_state == "snapshot_conflict_detected":
        snapshot["uncertainty_flags"].append("conflict_in_snapshot")
        snapshot["state_integrity_flags"].append("integrity_conflict")
        snapshot["virtual_boundary_flags"].append("boundary_conflict")

    if snapshot_confidence_state == "snapshot_origin_unknown":
        snapshot["uncertainty_flags"].append("origin_unknown")

    if snapshot_source_type == "internal_virtual_agent_state_candidate":
        snapshot["candidate_only_fields"].append("agent_state_no_real_person_identity")

    if snapshot_source_type == "internal_virtual_relationship_scene_candidate":
        snapshot["candidate_only_fields"].append("relationship_scene_no_relationship_assertion")

    if snapshot_source_type == "simulation_world_state_candidate":
        snapshot["candidate_only_fields"].append("simulation_cannot_become_current_fact")

    if snapshot_source_type == "dream_world_state_candidate":
        snapshot["candidate_only_fields"].append("dream_cannot_become_external_fact")

    snapshot["virtual_world_snapshot_passed"] = True
    snapshot["virtual_world_snapshot_status"] = "passed"

    return snapshot

def validate_virtual_world_state_snapshot(snapshot: Dict[str, Any]) -> bool:
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("virtual_world_snapshot_passed") is not True:
        return False

    # Ensure all invariants are met (i.e. everything is false)
    invariants = [
        "external_reality_asserted",
        "current_external_fact_asserted",
        "real_world_event_asserted",
        "virtual_state_mutation_performed",
        "virtual_world_write_allowed",
        "virtual_world_persistence_allowed",
        "persistent_world_state_created",
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
        if snapshot.get(inv) is not False:
            return False

    # Positive invariants
    positive_invariants = [
        "internal_virtual_state_snapshot",
        "snapshot_only",
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "quarantine_required",
        "origin_fact_status_required"
    ]
    for inv in positive_invariants:
        if snapshot.get(inv) is not True:
            return False

    return True

def build_virtual_world_snapshot_to_origin_fact_status_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_origin_fact_status",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "origin": "virtual_world_visual",
        "fact_status": "observed_internal_virtual",
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }

def build_virtual_world_snapshot_to_observation_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_observation",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }

def build_virtual_world_snapshot_to_memory_candidate_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_memory_candidate",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }

def build_virtual_world_snapshot_to_quarantine_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_quarantine",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }

def build_virtual_world_snapshot_to_appraisal_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_appraisal",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }

def build_virtual_world_snapshot_to_agp_input_plan(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_snapshot_to_agp_input",
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "ready": validate_virtual_world_state_snapshot(snapshot) and snapshot.get("virtual_world_snapshot_passed", False)
    }
