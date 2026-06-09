from typing import Dict, Any, Optional, List

VERSION = "3.1.0-round1141"

SUPPORTED_MEMORY_CANDIDATE_SOURCE_TYPES = [
    "sensory_observation_candidate",
    "multimodal_event_candidate",
    "memory_replay_observation_candidate",
    "cross_modal_binding_preflight_candidate",
    "operator_supplied_memory_candidate",
    "narrative_summary_candidate",
    "self_model_candidate",
    "relationship_memory_candidate",
    "simulation_candidate",
    "dream_candidate"
]

SUPPORTED_PROVENANCE_STATES = [
    "provenance_complete_unverified",
    "provenance_partial",
    "provenance_missing",
    "provenance_conflict_detected",
    "provenance_origin_unknown",
    "provenance_external_observed_candidate",
    "provenance_internal_virtual_candidate",
    "provenance_reconstructed_memory_candidate",
    "provenance_imagined_candidate",
    "provenance_simulated_candidate"
]

SUPPORTED_QUARANTINE_DECISIONS = [
    "quarantine_review_required",
    "quarantine_blocked_missing_provenance",
    "quarantine_blocked_origin_conflict",
    "quarantine_blocked_fact_status_conflict",
    "quarantine_blocked_identity_assertion",
    "quarantine_blocked_privacy_risk",
    "quarantine_blocked_memory_truth_assertion",
    "quarantine_blocked_self_model_update",
    "quarantine_blocked_relationship_update",
    "quarantine_blocked_external_fact_assertion",
    "quarantine_future_review_allowed",
    "quarantine_not_applicable"
]

def memory_provenance_quarantine_preflight_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "feature_track": "read_only_memory_provenance_and_quarantine_preflight_schema",
        "supported_memory_candidate_source_types": sorted(SUPPORTED_MEMORY_CANDIDATE_SOURCE_TYPES),
        "supported_provenance_states": sorted(SUPPORTED_PROVENANCE_STATES),
        "supported_quarantine_decisions": sorted(SUPPORTED_QUARANTINE_DECISIONS)
    }

def build_memory_provenance_quarantine_preflight_schema_summary() -> Dict[str, Any]:
    return memory_provenance_quarantine_preflight_schema_summary()

def build_memory_provenance_quarantine_preflight(
    memory_candidate: Optional[Dict[str, Any]] = None,
    provenance_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if memory_candidate is None:
        memory_candidate = {}
    if provenance_context is None:
        provenance_context = {}
    if metadata is None:
        metadata = {}

    failed = False
    decision = "quarantine_not_applicable"
    blocked_reasons = []
    uncertainty_flags = []
    provenance_flags = []
    boundary_flags = []
    warnings = []
    candidate_only_fields = []
    privacy_flags = []

    # Baseline validations
    if not memory_candidate:
        failed = True
        blocked_reasons.append("empty_memory_candidate")
        decision = "quarantine_not_applicable"

    source_type = metadata.get("memory_candidate_source_type")
    if not source_type or source_type not in SUPPORTED_MEMORY_CANDIDATE_SOURCE_TYPES:
        failed = True
        blocked_reasons.append(f"unknown_memory_candidate_source_type_{source_type}")
        decision = "quarantine_not_applicable"

    provenance_state = metadata.get("provenance_state")
    if not provenance_state or provenance_state not in SUPPORTED_PROVENANCE_STATES:
        failed = True
        blocked_reasons.append(f"unknown_provenance_state_{provenance_state}")
        decision = "quarantine_not_applicable"

    if not failed:
        if provenance_state == "provenance_missing":
            decision = "quarantine_blocked_missing_provenance"
            blocked_reasons.append("missing_provenance_blocks_future_memory_write")
        elif provenance_state == "provenance_partial":
            decision = "quarantine_review_required"
            warnings.append("partial_provenance_requires_quarantine_review")
        elif provenance_state == "provenance_complete_unverified":
            decision = "quarantine_review_required"
        elif provenance_state == "provenance_conflict_detected":
            decision = "quarantine_review_required"
            uncertainty_flags.append("provenance_conflict")
            provenance_flags.append("provenance_conflict_detected")

        # Check metadata blockers
        if metadata.get("origin_conflict"):
            decision = "quarantine_blocked_origin_conflict"
            blocked_reasons.append("origin_conflict_blocks_future_review")
        if metadata.get("fact_status_conflict"):
            decision = "quarantine_blocked_fact_status_conflict"
            blocked_reasons.append("fact_status_conflict_blocks_future_review")
        if metadata.get("identity_assertion"):
            decision = "quarantine_blocked_identity_assertion"
            blocked_reasons.append("identity_assertion_blocks_future_review")
        if metadata.get("privacy_risk"):
            decision = "quarantine_blocked_privacy_risk"
            blocked_reasons.append("privacy_risk_blocks_future_review")
            privacy_flags.append("privacy_risk_detected")
        if metadata.get("memory_truth_assertion"):
            decision = "quarantine_blocked_memory_truth_assertion"
            blocked_reasons.append("memory_truth_assertion_blocks_future_review")
        if metadata.get("self_model_update_request"):
            decision = "quarantine_blocked_self_model_update"
            blocked_reasons.append("self_model_update_request_blocks_future_review")
        if metadata.get("relationship_update_request"):
            decision = "quarantine_blocked_relationship_update"
            blocked_reasons.append("relationship_update_request_blocks_future_review")
        if metadata.get("external_fact_assertion"):
            decision = "quarantine_blocked_external_fact_assertion"
            blocked_reasons.append("external_fact_assertion_blocks_future_review")

        # Candidate type specific rules
        if source_type in [
            "sensory_observation_candidate",
            "multimodal_event_candidate",
            "memory_replay_observation_candidate",
            "operator_supplied_memory_candidate",
            "narrative_summary_candidate"
        ]:
            candidate_only_fields.append(source_type)

        if source_type == "self_model_candidate":
            warnings.append("self_model_candidate_cannot_update_self_model")
        elif source_type == "relationship_memory_candidate":
            warnings.append("relationship_memory_candidate_cannot_assert_relationship_state")
        elif source_type == "simulation_candidate":
            warnings.append("simulation_candidate_cannot_become_current_fact")
        elif source_type == "dream_candidate":
            warnings.append("dream_candidate_cannot_become_external_fact")

    return {
        "memory_provenance_preflight_passed": not failed,
        "memory_provenance_preflight_status": "valid" if not failed else "failed",
        "quarantine_decision": decision,
        "memory_candidate_id": str(metadata.get("memory_candidate_id", "unknown_id")),
        "memory_candidate_source_type": source_type if not failed else "unknown",
        "provenance_state": provenance_state if not failed else "unknown",
        "provenance_context": provenance_context,
        "metadata": metadata,
        "origin_summary": {"origin": metadata.get("origin", "unknown") if not failed else "unknown"},
        "fact_status_summary": {"fact_status": metadata.get("fact_status", "unknown") if not failed else "unknown"},

        "quarantine_required": True,
        "quarantine_review_allowed": decision in ["quarantine_review_required", "quarantine_future_review_allowed"],
        "quarantine_promotion_allowed": False,

        "memory_write_performed": False,
        "memory_write_allowed": False,
        "long_term_memory_write_allowed": False,
        "memory_truth_asserted": False,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "identity_asserted": False,
        "user_emotion_asserted": False,
        "user_intent_asserted": False,
        "relationship_state_asserted": False,

        "self_model_update_allowed": False,
        "relationship_update_allowed": False,
        "affect_transition_allowed": False,
        "hormone_transition_allowed": False,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,

        "vector_read_performed": False,
        "vector_load_performed": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
        "artifact_created_or_staged": False,

        "preflight_only": True,
        "candidate_only_fields": candidate_only_fields,
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": privacy_flags,
        "provenance_flags": provenance_flags,
        "boundary_flags": boundary_flags,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,

        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "origin_fact_status_required": True
    }

def validate_memory_provenance_quarantine_preflight(preflight: Dict[str, Any]) -> Dict[str, Any]:
    must_be_false = [
        "quarantine_promotion_allowed",
        "memory_write_performed",
        "memory_write_allowed",
        "long_term_memory_write_allowed",
        "memory_truth_asserted",
        "current_external_fact_asserted",
        "real_world_event_asserted",
        "identity_asserted",
        "user_emotion_asserted",
        "user_intent_asserted",
        "relationship_state_asserted",
        "self_model_update_allowed",
        "relationship_update_allowed",
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

    for key in must_be_false:
        if preflight.get(key) is not False:
            return {"validation_passed": False, "reason": f"{key}_must_be_false"}

    must_be_true = [
        "quarantine_required",
        "preflight_only",
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "origin_fact_status_required"
    ]

    for key in must_be_true:
        if preflight.get(key) is not True:
            return {"validation_passed": False, "reason": f"{key}_must_be_true"}

    return {"validation_passed": True, "reason": "ok"}

def build_memory_provenance_to_origin_fact_status_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "origin_fact_status_plan",
        "preflight": preflight,
        "valid": validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"]
    }

def build_memory_provenance_to_quarantine_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    valid = validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"]
    if not preflight.get("quarantine_review_allowed", False):
        valid = False
    return {
        "plan_type": "quarantine_plan",
        "preflight": preflight,
        "valid": valid
    }

def build_memory_provenance_to_memory_write_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "memory_write_plan",
        "preflight": preflight,
        "valid": False  # Memory write is never allowed from preflight
    }

def build_memory_provenance_to_appraisal_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "appraisal_plan",
        "preflight": preflight,
        "valid": validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"]
    }

def build_memory_provenance_to_agp_input_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "agp_input_plan",
        "preflight": preflight,
        "valid": validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"]
    }
