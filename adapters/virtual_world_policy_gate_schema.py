import datetime
import uuid
from typing import Dict, Any, List, Optional

def build_virtual_world_policy_gate_schema_summary() -> Dict[str, Any]:
    return virtual_world_policy_gate_schema_summary()

def virtual_world_policy_gate_schema_summary() -> Dict[str, Any]:
    return {
        "schema_type": "virtual_world_policy_gate_schema",
        "description": "Pure read-only policy gate schema evaluating virtual world candidates.",
        "supported_sources": [
            "virtual_world_observation_candidate",
            "virtual_world_state_snapshot_candidate",
            "virtual_world_transition_preflight_candidate",
            "virtual_world_consistency_audit_candidate",
            "symbolic_virtual_candidate",
            "dmn_virtual_candidate",
            "simulation_virtual_candidate",
            "dream_virtual_candidate",
            "operator_supplied_virtual_policy_candidate"
        ],
        "supported_boundaries": [
            "internal_virtual_policy_boundary",
            "symbolic_virtual_policy_boundary",
            "dmn_virtual_policy_boundary",
            "simulated_virtual_policy_boundary",
            "dream_virtual_policy_boundary",
            "mixed_virtual_external_policy_boundary"
        ],
        "supported_decisions": [
            "policy_future_review_allowed",
            "policy_review_required",
            "policy_blocked_missing_candidate",
            "policy_blocked_origin_conflict",
            "policy_blocked_fact_status_conflict",
            "policy_blocked_boundary_conflict",
            "policy_blocked_external_fact_assertion",
            "policy_blocked_identity_assertion",
            "policy_blocked_relationship_assertion",
            "policy_blocked_self_model_update",
            "policy_blocked_memory_write",
            "policy_blocked_persistence_write",
            "policy_blocked_virtual_state_mutation",
            "policy_blocked_transition_application",
            "policy_blocked_privacy_risk",
            "policy_not_applicable"
        ],
        "supported_risks": [
            "policy_risk_none",
            "policy_risk_low",
            "policy_risk_medium",
            "policy_risk_high",
            "policy_risk_unknown"
        ]
    }

def build_virtual_world_policy_gate(candidate: Optional[Dict[str, Any]] = None, policy_context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if candidate is None:
        candidate = {}
    if policy_context is None:
        policy_context = {}
    if metadata is None:
        metadata = {}

    schema_info = virtual_world_policy_gate_schema_summary()

    # Initialize output fields
    gate = {
        "virtual_world_policy_gate_passed": False,
        "virtual_world_policy_gate_status": "initialized",
        "policy_gate_id": str(uuid.uuid4()),
        "policy_candidate_source_type": candidate.get("candidate_source_type", "unknown"),
        "policy_boundary_classification": policy_context.get("boundary_classification", "unknown"),
        "policy_decision": "policy_review_required",
        "policy_risk_level": policy_context.get("risk_level", "policy_risk_unknown"),
        "candidate": candidate,
        "policy_context": policy_context,
        "metadata": metadata,
        "origin_summary": policy_context.get("origin_summary", {}),
        "fact_status_summary": policy_context.get("fact_status_summary", {}),

        # Mandatory read-only true flags
        "policy_gate_only": True,
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True,

        # Mandatory read-only false flags
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
        "policy_flags": [],
        "blocked_reasons": [],
        "warnings": []
    }

    # Extract flags/requests from candidate
    flags = candidate.get("flags", [])
    if isinstance(flags, list):
        for f in flags:
            gate["candidate_only_fields"].append(str(f))

    # Evaluate closed conditions first
    if not candidate:
        gate["policy_decision"] = "policy_blocked_missing_candidate"
        gate["virtual_world_policy_gate_status"] = "failed_closed"
        gate["blocked_reasons"].append("empty candidate fails closed")
        return gate

    if gate["policy_candidate_source_type"] not in schema_info["supported_sources"]:
        gate["policy_decision"] = "policy_blocked_missing_candidate" # Default blocked mapping for unknown source
        gate["virtual_world_policy_gate_status"] = "failed_closed"
        gate["blocked_reasons"].append(f"unknown source type {gate['policy_candidate_source_type']} fails closed")
        return gate

    if gate["policy_boundary_classification"] not in schema_info["supported_boundaries"]:
        gate["policy_decision"] = "policy_blocked_boundary_conflict" # Default blocked mapping for unknown boundary
        gate["virtual_world_policy_gate_status"] = "failed_closed"
        gate["blocked_reasons"].append(f"unknown boundary class {gate['policy_boundary_classification']} fails closed")
        return gate

    if gate["policy_risk_level"] not in schema_info["supported_risks"]:
        gate["policy_decision"] = "policy_review_required" # Unknown risk doesn't fail decision outright, but blocks passed
        gate["virtual_world_policy_gate_status"] = "failed_closed"
        gate["blocked_reasons"].append(f"unknown risk level {gate['policy_risk_level']} fails closed")
        return gate

    # Analyze requests/assertions to block
    requests = candidate.get("requests", [])
    assertions = candidate.get("assertions", [])

    if "external_fact_assertion" in assertions or gate["policy_candidate_source_type"] == "dream_virtual_candidate":
        gate["policy_decision"] = "policy_blocked_external_fact_assertion"
        gate["blocked_reasons"].append("external fact assertion blocks future review")

    if "identity_assertion" in assertions:
        gate["policy_decision"] = "policy_blocked_identity_assertion"
        gate["blocked_reasons"].append("identity assertion blocks future review")

    if "relationship_assertion" in assertions:
        gate["policy_decision"] = "policy_blocked_relationship_assertion"
        gate["blocked_reasons"].append("relationship assertion blocks future review")

    if "self_model_update" in requests:
        gate["policy_decision"] = "policy_blocked_self_model_update"
        gate["blocked_reasons"].append("self-model update request blocks future review")

    if "memory_write" in requests:
        gate["policy_decision"] = "policy_blocked_memory_write"
        gate["blocked_reasons"].append("memory write request blocks future review")

    if "persistence_write" in requests:
        gate["policy_decision"] = "policy_blocked_persistence_write"
        gate["blocked_reasons"].append("persistence write request blocks future review")

    if "virtual_state_mutation" in requests:
        gate["policy_decision"] = "policy_blocked_virtual_state_mutation"
        gate["blocked_reasons"].append("virtual state mutation request blocks future review")

    if "transition_application" in requests:
        gate["policy_decision"] = "policy_blocked_transition_application"
        gate["blocked_reasons"].append("transition application request blocks future review")

    # Analyze context
    if gate["policy_boundary_classification"] == "mixed_virtual_external_policy_boundary":
        gate["virtual_boundary_flags"].append("mixed_boundary")
        gate["policy_decision"] = "policy_blocked_boundary_conflict"
        gate["blocked_reasons"].append("mixed virtual/external boundary blocks external assertion")

    if gate["policy_risk_level"] == "policy_risk_high":
        gate["policy_decision"] = "policy_review_required"
        gate["blocked_reasons"].append("high risk blocks future review")

    if "privacy_risk" in policy_context.get("risks", []):
        gate["policy_decision"] = "policy_blocked_privacy_risk"
        gate["privacy_flags"].append("detected")
        gate["blocked_reasons"].append("privacy risk blocks future review")

    if policy_context.get("origin_conflict", False):
        gate["policy_decision"] = "policy_blocked_origin_conflict"
        gate["blocked_reasons"].append("origin conflict blocks future review")

    if policy_context.get("fact_status_conflict", False):
        gate["policy_decision"] = "policy_blocked_fact_status_conflict"
        gate["blocked_reasons"].append("fact status conflict blocks future review")

    if policy_context.get("boundary_conflict", False):
        gate["policy_decision"] = "policy_blocked_boundary_conflict"
        gate["blocked_reasons"].append("boundary conflict blocks future review")

    # Specific type rules
    if gate["policy_candidate_source_type"] in ["symbolic_virtual_candidate", "dmn_virtual_candidate", "operator_supplied_virtual_policy_candidate"]:
        gate["policy_flags"].append("remains_candidate_only")

    if gate["policy_candidate_source_type"] == "simulation_virtual_candidate":
        gate["policy_flags"].append("simulation_cannot_become_current_fact")

    # If no blocks and recognized, allow future review
    if not gate["blocked_reasons"]:
        gate["policy_decision"] = "policy_future_review_allowed"
        gate["virtual_world_policy_gate_passed"] = True
        gate["virtual_world_policy_gate_status"] = "passed"
    else:
        gate["virtual_world_policy_gate_passed"] = False
        if gate["virtual_world_policy_gate_status"] != "failed_closed":
             gate["virtual_world_policy_gate_status"] = "blocked"

    # Enforce decision valid
    if gate["policy_decision"] not in schema_info["supported_decisions"]:
        gate["policy_decision"] = "policy_review_required"
        gate["virtual_world_policy_gate_passed"] = False
        gate["virtual_world_policy_gate_status"] = "failed_closed"
        gate["blocked_reasons"].append("unknown policy decision fails closed")

    return gate

def validate_virtual_world_policy_gate(gate: Dict[str, Any]) -> bool:
    if not isinstance(gate, dict):
        return False

    # Check required fields
    required_fields = [
        "virtual_world_policy_gate_passed", "virtual_world_policy_gate_status", "policy_gate_id",
        "policy_candidate_source_type", "policy_boundary_classification", "policy_decision",
        "policy_risk_level", "candidate", "policy_context", "metadata", "origin_summary",
        "fact_status_summary", "policy_gate_only", "policy_applied", "audit_applied",
        "transition_applied", "virtual_state_mutation_performed", "virtual_world_write_allowed",
        "virtual_world_persistence_allowed", "persistent_world_state_created", "external_reality_asserted",
        "current_external_fact_asserted", "real_world_event_asserted", "memory_write_performed",
        "memory_write_allowed", "quarantine_promotion_allowed", "self_model_update_allowed",
        "relationship_update_allowed", "relationship_state_asserted", "identity_asserted",
        "user_emotion_asserted", "user_intent_asserted", "affect_transition_allowed",
        "hormone_transition_allowed", "agp_bypass_allowed", "fallback_bypass_allowed",
        "vector_read_performed", "vector_load_performed", "runtime_mutation_performed",
        "persistence_write_performed", "artifact_created_or_staged", "candidate_only_fields",
        "uncertainty_flags", "privacy_flags", "virtual_boundary_flags", "policy_flags",
        "blocked_reasons", "warnings", "appraisal_required", "agp_input_required",
        "memory_gate_required", "quarantine_required", "origin_fact_status_required"
    ]

    for f in required_fields:
        if f not in gate:
            return False

    # Check false fields
    false_fields = [
        "policy_applied", "audit_applied", "transition_applied", "virtual_state_mutation_performed",
        "virtual_world_write_allowed", "virtual_world_persistence_allowed", "persistent_world_state_created",
        "external_reality_asserted", "current_external_fact_asserted", "real_world_event_asserted",
        "memory_write_performed", "memory_write_allowed", "quarantine_promotion_allowed",
        "self_model_update_allowed", "relationship_update_allowed", "relationship_state_asserted",
        "identity_asserted", "user_emotion_asserted", "user_intent_asserted", "affect_transition_allowed",
        "hormone_transition_allowed", "agp_bypass_allowed", "fallback_bypass_allowed",
        "vector_read_performed", "vector_load_performed", "runtime_mutation_performed",
        "persistence_write_performed", "artifact_created_or_staged"
    ]
    for f in false_fields:
        if gate.get(f) is not False:
            return False

    # Check true fields
    true_fields = [
        "policy_gate_only", "appraisal_required", "agp_input_required", "memory_gate_required",
        "quarantine_required", "origin_fact_status_required"
    ]
    for f in true_fields:
        if gate.get(f) is not True:
            return False

    return True

def build_virtual_world_policy_to_origin_fact_status_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "origin_fact_status", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_consistency_audit_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "consistency_audit", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_transition_preflight_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "transition_preflight", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_memory_candidate_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "memory_candidate", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_quarantine_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "quarantine", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_appraisal_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "appraisal", "gate_id": gate.get("policy_gate_id"), "status": "pending"}

def build_virtual_world_policy_to_agp_input_plan(gate: Dict[str, Any]) -> Dict[str, Any]:
    return {"plan_type": "agp_input", "gate_id": gate.get("policy_gate_id"), "status": "pending"}
