import uuid
from typing import Dict, Any, Optional

VERSION = "1.0.0"

SUPPORTED_AUDIT_SOURCE_TYPES = {
    "observation_snapshot_audit_candidate",
    "snapshot_transition_audit_candidate",
    "observation_transition_audit_candidate",
    "full_virtual_world_audit_candidate",
    "symbolic_virtual_audit_candidate",
    "dmn_virtual_audit_candidate",
    "simulation_virtual_audit_candidate",
    "dream_virtual_audit_candidate",
    "operator_supplied_virtual_audit_candidate"
}

SUPPORTED_AUDIT_BOUNDARY_CLASSES = {
    "internal_virtual_consistency_audit",
    "symbolic_virtual_consistency_audit",
    "dmn_virtual_consistency_audit",
    "simulated_virtual_consistency_audit",
    "dream_virtual_consistency_audit",
    "mixed_virtual_external_consistency_boundary"
}

SUPPORTED_AUDIT_DECISIONS = {
    "consistency_future_review_allowed",
    "consistency_review_required",
    "consistency_blocked_missing_observation",
    "consistency_blocked_missing_snapshot",
    "consistency_blocked_missing_transition",
    "consistency_blocked_origin_conflict",
    "consistency_blocked_fact_status_conflict",
    "consistency_blocked_boundary_conflict",
    "consistency_blocked_external_fact_assertion",
    "consistency_blocked_identity_assertion",
    "consistency_blocked_relationship_assertion",
    "consistency_blocked_self_model_update",
    "consistency_blocked_memory_write",
    "consistency_blocked_persistence_write",
    "consistency_blocked_virtual_state_mutation",
    "consistency_blocked_privacy_risk",
    "consistency_not_applicable"
}

def build_virtual_world_consistency_audit_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "track": "read_only_virtual_world_consistency_audit_schema",
        "supported_audit_source_types": list(SUPPORTED_AUDIT_SOURCE_TYPES),
        "supported_audit_boundary_classes": list(SUPPORTED_AUDIT_BOUNDARY_CLASSES),
        "supported_audit_decisions": list(SUPPORTED_AUDIT_DECISIONS)
    }

def virtual_world_consistency_audit_schema_summary() -> Dict[str, Any]:
    return build_virtual_world_consistency_audit_schema_summary()

def build_virtual_world_consistency_audit(
    observation_candidate: Optional[Dict[str, Any]] = None,
    snapshot_candidate: Optional[Dict[str, Any]] = None,
    transition_preflight: Optional[Dict[str, Any]] = None,
    audit_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    if observation_candidate is None:
        observation_candidate = {}
    if snapshot_candidate is None:
        snapshot_candidate = {}
    if transition_preflight is None:
        transition_preflight = {}
    if audit_context is None:
        audit_context = {}
    if metadata is None:
        metadata = {}

    audit_source_type = metadata.get("audit_source_type", "")
    audit_boundary_classification = metadata.get("audit_boundary_classification", "")

    # Check for direct update requests in metadata/context to block future review
    update_requests = metadata.get("update_requests", {})
    external_fact_assertion = update_requests.get("external_fact_assertion", False)
    identity_assertion = update_requests.get("identity_assertion", False)
    relationship_assertion = update_requests.get("relationship_assertion", False)
    self_model_update = update_requests.get("self_model_update", False)
    memory_write_request = update_requests.get("memory_write", False)
    persistence_write_request = update_requests.get("persistence_write", False)
    virtual_state_mutation_request = update_requests.get("virtual_state_mutation", False)
    privacy_risk = update_requests.get("privacy_risk", False)

    audit = {
        "virtual_world_consistency_audit_passed": False,
        "virtual_world_consistency_audit_status": "failed",
        "consistency_audit_id": str(uuid.uuid4()),
        "audit_source_type": audit_source_type,
        "audit_boundary_classification": audit_boundary_classification,
        "audit_decision": "consistency_not_applicable",
        "observation_candidate": observation_candidate,
        "snapshot_candidate": snapshot_candidate,
        "transition_preflight": transition_preflight,
        "audit_context": audit_context,
        "metadata": metadata,
        "origin_summary": {},
        "fact_status_summary": {},
        "consistency_audit_only": True,
        "audit_applied": False,
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
        "consistency_flags": [],
        "blocked_reasons": [],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }

    # Fail closed conditions
    if audit_source_type not in SUPPORTED_AUDIT_SOURCE_TYPES:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_unknown_source"
        audit["blocked_reasons"].append("unknown_audit_source_type")
        return audit

    if audit_boundary_classification not in SUPPORTED_AUDIT_BOUNDARY_CLASSES:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_unknown_boundary"
        audit["blocked_reasons"].append("unknown_audit_boundary_classification")
        return audit

    # Check required inputs
    obs_req = audit_source_type in [
        "observation_snapshot_audit_candidate",
        "observation_transition_audit_candidate",
        "full_virtual_world_audit_candidate"
    ]
    snap_req = audit_source_type in [
        "observation_snapshot_audit_candidate",
        "snapshot_transition_audit_candidate",
        "full_virtual_world_audit_candidate"
    ]
    tran_req = audit_source_type in [
        "snapshot_transition_audit_candidate",
        "observation_transition_audit_candidate",
        "full_virtual_world_audit_candidate"
    ]

    if obs_req and not observation_candidate:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_missing_observation"
        audit["audit_decision"] = "consistency_blocked_missing_observation"
        audit["blocked_reasons"].append("missing_observation_candidate")
        return audit

    if snap_req and not snapshot_candidate:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_missing_snapshot"
        audit["audit_decision"] = "consistency_blocked_missing_snapshot"
        audit["blocked_reasons"].append("missing_snapshot_candidate")
        return audit

    if tran_req and not transition_preflight:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_missing_transition"
        audit["audit_decision"] = "consistency_blocked_missing_transition"
        audit["blocked_reasons"].append("missing_transition_preflight")
        return audit

    # Mismatch checking
    if metadata.get("mismatch_detected", False):
        audit["consistency_flags"].append("mismatch_detected")

    # Conflict checking
    has_conflict = False
    if metadata.get("origin_conflict", False):
        audit["audit_decision"] = "consistency_blocked_origin_conflict"
        audit["blocked_reasons"].append("origin_conflict")
        has_conflict = True
    if metadata.get("fact_status_conflict", False):
        audit["audit_decision"] = "consistency_blocked_fact_status_conflict"
        audit["blocked_reasons"].append("fact_status_conflict")
        has_conflict = True
    if metadata.get("boundary_conflict", False) or audit_boundary_classification == "mixed_virtual_external_consistency_boundary":
        audit["audit_decision"] = "consistency_blocked_boundary_conflict"
        audit["blocked_reasons"].append("boundary_conflict")
        audit["virtual_boundary_flags"].append("boundary_conflict")
        has_conflict = True

    if external_fact_assertion:
        audit["audit_decision"] = "consistency_blocked_external_fact_assertion"
        audit["blocked_reasons"].append("external_fact_assertion")
        has_conflict = True
    if identity_assertion:
        audit["audit_decision"] = "consistency_blocked_identity_assertion"
        audit["blocked_reasons"].append("identity_assertion")
        has_conflict = True
    if relationship_assertion:
        audit["audit_decision"] = "consistency_blocked_relationship_assertion"
        audit["blocked_reasons"].append("relationship_assertion")
        has_conflict = True
    if self_model_update:
        audit["audit_decision"] = "consistency_blocked_self_model_update"
        audit["blocked_reasons"].append("self_model_update")
        has_conflict = True
    if memory_write_request:
        audit["audit_decision"] = "consistency_blocked_memory_write"
        audit["blocked_reasons"].append("memory_write")
        has_conflict = True
    if persistence_write_request:
        audit["audit_decision"] = "consistency_blocked_persistence_write"
        audit["blocked_reasons"].append("persistence_write")
        has_conflict = True
    if virtual_state_mutation_request:
        audit["audit_decision"] = "consistency_blocked_virtual_state_mutation"
        audit["blocked_reasons"].append("virtual_state_mutation")
        has_conflict = True
    if privacy_risk:
        audit["audit_decision"] = "consistency_blocked_privacy_risk"
        audit["blocked_reasons"].append("privacy_risk")
        audit["privacy_flags"].append("privacy_risk")
        has_conflict = True

    if has_conflict:
        audit["virtual_world_consistency_audit_passed"] = False
        audit["virtual_world_consistency_audit_status"] = "failed_conflict"
        return audit

    # Candidate only preservation
    if audit_source_type == "symbolic_virtual_audit_candidate":
        audit["candidate_only_fields"].append("symbolic_remains_candidate")
    elif audit_source_type == "dmn_virtual_audit_candidate":
        audit["candidate_only_fields"].append("dmn_remains_candidate")
    elif audit_source_type == "simulation_virtual_audit_candidate":
        audit["candidate_only_fields"].append("simulation_cannot_become_current_fact")
        audit["blocked_reasons"].append("simulation_cannot_become_current_fact")
    elif audit_source_type == "dream_virtual_audit_candidate":
        audit["candidate_only_fields"].append("dream_cannot_become_external_fact")
        audit["blocked_reasons"].append("dream_cannot_become_external_fact")
    elif audit_source_type == "operator_supplied_virtual_audit_candidate":
        audit["candidate_only_fields"].append("operator_supplied_remains_candidate")

    audit["virtual_world_consistency_audit_passed"] = True
    audit["virtual_world_consistency_audit_status"] = "passed"
    audit["audit_decision"] = "consistency_future_review_allowed"

    return audit

def validate_virtual_world_consistency_audit(audit: Dict[str, Any]) -> bool:
    if not isinstance(audit, dict):
        return False
    if audit.get("virtual_world_consistency_audit_passed") is not True:
        return False

    invariants = [
        "audit_applied",
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
        if audit.get(inv) is not False:
            return False

    positive_invariants = [
        "consistency_audit_only",
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "quarantine_required",
        "origin_fact_status_required"
    ]
    for inv in positive_invariants:
        if audit.get(inv) is not True:
            return False

    return True

def build_virtual_world_consistency_to_origin_fact_status_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_origin_fact_status",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_snapshot_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_snapshot",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_transition_preflight_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_transition_preflight",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_memory_candidate_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_memory_candidate",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_quarantine_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_quarantine",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_appraisal_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_appraisal",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }

def build_virtual_world_consistency_to_agp_input_plan(audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "virtual_world_consistency_to_agp_input",
        "audit_id": audit.get("consistency_audit_id"),
        "ready": validate_virtual_world_consistency_audit(audit)
    }
