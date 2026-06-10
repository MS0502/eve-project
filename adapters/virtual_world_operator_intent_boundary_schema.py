import datetime
import uuid
from typing import Dict, Any, List, Optional

def build_virtual_world_operator_intent_boundary_schema_summary() -> Dict[str, Any]:
    return virtual_world_operator_intent_boundary_schema_summary()

def virtual_world_operator_intent_boundary_schema_summary() -> Dict[str, Any]:
    return {
        "schema_type": "virtual_world_operator_intent_boundary_schema",
        "description": "Pure read-only boundary schema separating operator instruction, user utterance, and virtual intent.",
        "supported_sources": [
            "explicit_operator_instruction_candidate",
            "explicit_user_utterance_intent_candidate",
            "inferred_user_intent_candidate",
            "virtual_world_intent_candidate",
            "virtual_agent_intent_candidate",
            "symbolic_intent_candidate",
            "dmn_intent_candidate",
            "simulation_intent_candidate",
            "dream_intent_candidate",
            "operator_supplied_intent_boundary_candidate"
        ],
        "supported_boundaries": [
            "explicit_operator_boundary",
            "explicit_user_utterance_boundary",
            "inferred_user_intent_boundary",
            "internal_virtual_intent_boundary",
            "symbolic_intent_boundary",
            "dmn_intent_boundary",
            "simulated_intent_boundary",
            "dream_intent_boundary",
            "mixed_operator_user_virtual_boundary"
        ],
        "supported_decisions": [
            "intent_future_review_allowed",
            "intent_review_required",
            "intent_blocked_missing_candidate",
            "intent_blocked_unknown_source",
            "intent_blocked_unknown_boundary",
            "intent_blocked_unknown_decision",
            "intent_blocked_operator_user_conflict",
            "intent_blocked_inferred_user_intent_as_fact",
            "intent_blocked_virtual_intent_as_external_fact",
            "intent_blocked_identity_assertion",
            "intent_blocked_relationship_assertion",
            "intent_blocked_self_model_update",
            "intent_blocked_memory_write",
            "intent_blocked_persistence_write",
            "intent_blocked_virtual_state_mutation",
            "intent_blocked_transition_application",
            "intent_blocked_privacy_risk",
            "intent_not_applicable"
        ],
        "supported_confidence_states": [
            "intent_explicit",
            "intent_inferred_low_confidence",
            "intent_inferred_medium_confidence",
            "intent_inferred_high_confidence_but_not_fact",
            "intent_conflict_detected",
            "intent_origin_unknown"
        ]
    }

def build_virtual_world_operator_intent_boundary(intent_candidate: Optional[Dict[str, Any]] = None, boundary_context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if intent_candidate is None:
        intent_candidate = {}
    if boundary_context is None:
        boundary_context = {}
    if metadata is None:
        metadata = {}

    schema_info = virtual_world_operator_intent_boundary_schema_summary()

    # Initialize output fields
    boundary = {
        "virtual_world_operator_intent_boundary_passed": False,
        "virtual_world_operator_intent_boundary_status": "initialized",
        "intent_boundary_id": str(uuid.uuid4()),
        "intent_candidate_source_type": intent_candidate.get("candidate_source_type", "unknown"),
        "intent_boundary_classification": boundary_context.get("boundary_classification", "unknown"),
        "intent_decision": "intent_review_required",
        "intent_confidence_state": boundary_context.get("confidence_state", "intent_origin_unknown"),
        "intent_candidate": intent_candidate,
        "boundary_context": boundary_context,
        "metadata": metadata,
        "origin_summary": boundary_context.get("origin_summary", {}),
        "fact_status_summary": boundary_context.get("fact_status_summary", {}),

        # Mandatory read-only true flags
        "intent_boundary_only": True,
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True,

        # Dynamic bool flags
        "operator_intent_explicit": False,
        "user_intent_explicit": False,
        "user_intent_inferred": False,

        # Mandatory read-only false flags
        "inferred_intent_treated_as_fact": False,
        "virtual_intent_treated_as_external_fact": False,
        "boundary_applied": False,
        "policy_applied": False,
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

        # Dynamic lists
        "candidate_only_fields": [],
        "uncertainty_flags": [],
        "privacy_flags": [],
        "virtual_boundary_flags": [],
        "intent_boundary_flags": [],
        "blocked_reasons": [],
        "warnings": []
    }

    # Extract flags from candidate
    flags = intent_candidate.get("flags", [])
    if isinstance(flags, list):
        for f in flags:
            boundary["candidate_only_fields"].append(str(f))

    # Evaluate closed conditions first
    if not intent_candidate:
        boundary["intent_decision"] = "intent_blocked_missing_candidate"
        boundary["virtual_world_operator_intent_boundary_status"] = "failed_closed"
        boundary["blocked_reasons"].append("empty intent_candidate fails closed")
        return boundary

    if boundary["intent_candidate_source_type"] not in schema_info["supported_sources"]:
        boundary["intent_decision"] = "intent_blocked_unknown_source"
        boundary["virtual_world_operator_intent_boundary_status"] = "failed_closed"
        boundary["blocked_reasons"].append(f"unknown intent source type {boundary['intent_candidate_source_type']} fails closed")
        return boundary

    if boundary["intent_boundary_classification"] not in schema_info["supported_boundaries"]:
        boundary["intent_decision"] = "intent_blocked_unknown_boundary"
        boundary["virtual_world_operator_intent_boundary_status"] = "failed_closed"
        boundary["blocked_reasons"].append(f"unknown boundary class {boundary['intent_boundary_classification']} fails closed")
        return boundary

    if boundary["intent_confidence_state"] not in schema_info["supported_confidence_states"]:
        boundary["intent_decision"] = "intent_blocked_unknown_decision" # Using unknown decision as a proxy for unknown confidence failure
        boundary["virtual_world_operator_intent_boundary_status"] = "failed_closed"
        boundary["blocked_reasons"].append(f"unknown confidence state {boundary['intent_confidence_state']} fails closed")
        return boundary

    # Set explicit/inferred flags
    if boundary["intent_candidate_source_type"] == "explicit_operator_instruction_candidate":
        boundary["operator_intent_explicit"] = True
    elif boundary["intent_candidate_source_type"] == "explicit_user_utterance_intent_candidate":
        boundary["user_intent_explicit"] = True
    elif boundary["intent_candidate_source_type"] == "inferred_user_intent_candidate":
        boundary["user_intent_inferred"] = True
        if intent_candidate.get("treated_as_fact", False):
            boundary["intent_decision"] = "intent_blocked_inferred_user_intent_as_fact"
            boundary["blocked_reasons"].append("inferred user intent as fact blocks future review")

    # Virtual intent rules
    if boundary["intent_candidate_source_type"] in ["virtual_world_intent_candidate", "virtual_agent_intent_candidate", "symbolic_intent_candidate", "dmn_intent_candidate", "simulation_intent_candidate", "dream_intent_candidate"]:
         if intent_candidate.get("treated_as_external_fact", False):
            boundary["intent_decision"] = "intent_blocked_virtual_intent_as_external_fact"
            boundary["blocked_reasons"].append("virtual intent as external fact blocks future review")

    # Simulation/Dream/Symbolic/DMN rules
    if boundary["intent_candidate_source_type"] == "simulation_intent_candidate":
        if intent_candidate.get("assert_current_fact", False):
            boundary["intent_decision"] = "intent_blocked_virtual_intent_as_external_fact" # Proxy for cannot become current fact
            boundary["blocked_reasons"].append("simulation intent cannot become current fact")

    if boundary["intent_candidate_source_type"] == "dream_intent_candidate":
        if intent_candidate.get("assert_external_fact", False):
            boundary["intent_decision"] = "intent_blocked_virtual_intent_as_external_fact"
            boundary["blocked_reasons"].append("dream intent cannot become external fact")

    if boundary["intent_candidate_source_type"] in ["symbolic_intent_candidate", "dmn_intent_candidate"]:
        boundary["intent_boundary_flags"].append("remains_candidate_only")

    # Mixed boundary rules
    if boundary["intent_boundary_classification"] == "mixed_operator_user_virtual_boundary":
        boundary["intent_boundary_flags"].append("mixed_boundary")
        boundary["intent_decision"] = "intent_review_required"
        boundary["blocked_reasons"].append("mixed operator/user/virtual boundary requires review")

    # Conflict rules
    if boundary_context.get("operator_user_conflict", False):
        boundary["intent_decision"] = "intent_blocked_operator_user_conflict"
        boundary["blocked_reasons"].append("operator-user conflict blocks future review")

    # Analyze requests/assertions to block
    requests = intent_candidate.get("requests", [])
    assertions = intent_candidate.get("assertions", [])

    if "external_reality_assertion" in assertions:
        boundary["intent_decision"] = "intent_blocked_virtual_intent_as_external_fact" # Proxy
        boundary["blocked_reasons"].append("external fact assertion blocks future review")

    if "identity_assertion" in assertions:
        boundary["intent_decision"] = "intent_blocked_identity_assertion"
        boundary["blocked_reasons"].append("identity assertion blocks future review")

    if "relationship_assertion" in assertions:
        boundary["intent_decision"] = "intent_blocked_relationship_assertion"
        boundary["blocked_reasons"].append("relationship assertion blocks future review")

    if "self_model_update" in requests:
        boundary["intent_decision"] = "intent_blocked_self_model_update"
        boundary["blocked_reasons"].append("self_model_update request blocks future review")

    if "memory_write" in requests:
        boundary["intent_decision"] = "intent_blocked_memory_write"
        boundary["blocked_reasons"].append("memory_write request blocks future review")

    if "persistence_write" in requests:
        boundary["intent_decision"] = "intent_blocked_persistence_write"
        boundary["blocked_reasons"].append("persistence_write request blocks future review")

    if "virtual_state_mutation" in requests:
        boundary["intent_decision"] = "intent_blocked_virtual_state_mutation"
        boundary["blocked_reasons"].append("virtual_state_mutation request blocks future review")

    if "transition_application" in requests:
        boundary["intent_decision"] = "intent_blocked_transition_application"
        boundary["blocked_reasons"].append("transition_application request blocks future review")

    if "privacy_risk" in boundary_context.get("risks", []):
        boundary["intent_decision"] = "intent_blocked_privacy_risk"
        boundary["privacy_flags"].append("detected")
        boundary["blocked_reasons"].append("privacy risk blocks future review")

    # If no blocks and recognized, allow future review (unless review required)
    if not boundary["blocked_reasons"]:
        boundary["intent_decision"] = "intent_future_review_allowed"
        boundary["virtual_world_operator_intent_boundary_passed"] = True
        boundary["virtual_world_operator_intent_boundary_status"] = "passed"
    else:
        boundary["virtual_world_operator_intent_boundary_passed"] = False
        if boundary["virtual_world_operator_intent_boundary_status"] != "failed_closed":
             boundary["virtual_world_operator_intent_boundary_status"] = "blocked"
             if "requires review" in "".join(boundary["blocked_reasons"]):
                 boundary["intent_decision"] = "intent_review_required"

    # Enforce decision valid
    if boundary["intent_decision"] not in schema_info["supported_decisions"]:
        boundary["intent_decision"] = "intent_blocked_unknown_decision"
        boundary["virtual_world_operator_intent_boundary_passed"] = False
        boundary["virtual_world_operator_intent_boundary_status"] = "failed_closed"
        boundary["blocked_reasons"].append("unknown intent decision fails closed")

    return boundary

def validate_virtual_world_operator_intent_boundary(boundary: Dict[str, Any]) -> bool:
    if not isinstance(boundary, dict):
        return False

    # Check required fields
    required_fields = [
        "virtual_world_operator_intent_boundary_passed", "virtual_world_operator_intent_boundary_status",
        "intent_boundary_id", "intent_candidate_source_type", "intent_boundary_classification",
        "intent_decision", "intent_confidence_state", "intent_candidate", "boundary_context",
        "metadata", "origin_summary", "fact_status_summary", "intent_boundary_only",
        "operator_intent_explicit", "user_intent_explicit", "user_intent_inferred",
        "inferred_intent_treated_as_fact", "virtual_intent_treated_as_external_fact",
        "boundary_applied", "policy_applied", "audit_applied", "transition_applied",
        "virtual_state_mutation_performed", "virtual_world_write_allowed",
        "virtual_world_persistence_allowed", "persistent_world_state_created",
        "external_reality_asserted", "current_external_fact_asserted", "real_world_event_asserted",
        "memory_write_performed", "memory_write_allowed", "quarantine_promotion_allowed",
        "self_model_update_allowed", "relationship_update_allowed", "relationship_state_asserted",
        "identity_asserted", "user_emotion_asserted", "user_intent_asserted", "affect_transition_allowed",
        "hormone_transition_allowed", "agp_bypass_allowed", "fallback_bypass_allowed",
        "vector_read_performed", "vector_load_performed", "runtime_mutation_performed",
        "persistence_write_performed", "artifact_created_or_staged", "candidate_only_fields",
        "uncertainty_flags", "privacy_flags", "virtual_boundary_flags", "intent_boundary_flags",
        "blocked_reasons", "warnings", "appraisal_required", "agp_input_required",
        "memory_gate_required", "quarantine_required", "origin_fact_status_required"
    ]

    for f in required_fields:
        if f not in boundary:
            return False

    # Check false fields
    false_fields = [
        "inferred_intent_treated_as_fact", "virtual_intent_treated_as_external_fact",
        "boundary_applied", "policy_applied", "audit_applied", "transition_applied",
        "virtual_state_mutation_performed", "virtual_world_write_allowed",
        "virtual_world_persistence_allowed", "persistent_world_state_created",
        "external_reality_asserted", "current_external_fact_asserted", "real_world_event_asserted",
        "memory_write_performed", "memory_write_allowed", "quarantine_promotion_allowed",
        "self_model_update_allowed", "relationship_update_allowed", "relationship_state_asserted",
        "identity_asserted", "user_emotion_asserted", "user_intent_asserted", "affect_transition_allowed",
        "hormone_transition_allowed", "agp_bypass_allowed", "fallback_bypass_allowed",
        "vector_read_performed", "vector_load_performed", "runtime_mutation_performed",
        "persistence_write_performed", "artifact_created_or_staged"
    ]
    for f in false_fields:
        if boundary.get(f) is not False:
            return False

    # Check true fields
    true_fields = [
        "intent_boundary_only", "appraisal_required", "agp_input_required",
        "memory_gate_required", "quarantine_required", "origin_fact_status_required"
    ]
    for f in true_fields:
        if boundary.get(f) is not True:
            return False

    return True

def build_operator_intent_boundary_to_origin_fact_status_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "origin_fact_status", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_policy_gate_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "policy_gate", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_consistency_audit_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "consistency_audit", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_memory_candidate_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "memory_candidate", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_quarantine_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "quarantine", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_appraisal_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "appraisal", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}

def build_operator_intent_boundary_to_agp_input_plan(boundary: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "agp_input", "boundary_id": boundary.get("intent_boundary_id"), "status": "pending"}
