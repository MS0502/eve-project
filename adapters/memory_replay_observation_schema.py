"""Read-only memory replay observation schema for Round1121-1140."""

from typing import Any, Dict, List, Optional

VERSION = "v3_round1121_1140_memory_replay_observation_schema"

SUPPORTED_REPLAY_SOURCE_TYPES = [
    "episodic_replay_candidate",
    "semantic_replay_candidate",
    "narrative_replay_candidate",
    "self_model_replay_candidate",
    "relationship_replay_candidate",
    "dream_replay_candidate",
    "simulation_replay_candidate",
    "operator_supplied_replay_candidate"
]

SUPPORTED_REPLAY_CONFIDENCE_STATES = [
    "replay_unverified",
    "replay_low_confidence",
    "replay_medium_confidence",
    "replay_high_confidence_but_not_fact",
    "replay_conflict_detected",
    "replay_origin_unknown"
]

SUPPORTED_REPLAY_BOUNDARY_CLASSES = [
    "reconstructed_memory",
    "symbolic_memory_trace",
    "narrative_memory_trace",
    "simulated_memory_like_content",
    "dream_memory_like_content",
    "mixed_replay_boundary"
]

def memory_replay_observation_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "feature_track": "read_only_memory_replay_observation_schema",
        "supported_replay_source_types": sorted(SUPPORTED_REPLAY_SOURCE_TYPES),
        "supported_replay_confidence_states": sorted(SUPPORTED_REPLAY_CONFIDENCE_STATES),
        "supported_replay_boundary_classes": sorted(SUPPORTED_REPLAY_BOUNDARY_CLASSES)
    }

def build_memory_replay_observation_schema_summary() -> Dict[str, Any]:
    return memory_replay_observation_schema_summary()

def build_memory_replay_observation(
    memory_fragment: Optional[Dict[str, Any]] = None,
    replay_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if memory_fragment is None:
        memory_fragment = {}
    if replay_context is None:
        replay_context = {}
    if metadata is None:
        metadata = {}

    failed = False
    blocked_reasons = []
    uncertainty_flags = []
    boundary_flags = []
    warnings = []
    candidate_only_fields = []

    if not memory_fragment:
        failed = True
        blocked_reasons.append("empty_memory_fragment")

    replay_source_type = metadata.get("replay_source_type")
    if not replay_source_type or replay_source_type not in SUPPORTED_REPLAY_SOURCE_TYPES:
        failed = True
        blocked_reasons.append(f"unknown_replay_source_type_{replay_source_type}")

    replay_confidence_state = metadata.get("replay_confidence_state")
    if not replay_confidence_state or replay_confidence_state not in SUPPORTED_REPLAY_CONFIDENCE_STATES:
        failed = True
        blocked_reasons.append(f"unknown_replay_confidence_state_{replay_confidence_state}")

    replay_boundary_classification = metadata.get("replay_boundary_classification")
    if not replay_boundary_classification or replay_boundary_classification not in SUPPORTED_REPLAY_BOUNDARY_CLASSES:
        failed = True
        blocked_reasons.append(f"unknown_replay_boundary_classification_{replay_boundary_classification}")

    if replay_confidence_state == "replay_conflict_detected":
        uncertainty_flags.append("replay_conflict_detected")

    if replay_boundary_classification == "mixed_replay_boundary":
        boundary_flags.append("mixed_replay_boundary")

    if replay_source_type in [
        "semantic_replay_candidate",
        "narrative_replay_candidate",
        "operator_supplied_replay_candidate"
    ]:
        candidate_only_fields.append(replay_source_type)

    if replay_source_type == "self_model_replay_candidate":
        warnings.append("self_model_replay_cannot_update_self_model")
    elif replay_source_type == "relationship_replay_candidate":
        warnings.append("relationship_replay_cannot_assert_relationship_state")
    elif replay_source_type == "dream_replay_candidate":
        warnings.append("dream_replay_remains_dream_memory_like")
    elif replay_source_type == "simulation_replay_candidate":
        warnings.append("simulation_replay_remains_simulated_future")

    if replay_boundary_classification == "reconstructed_memory" and not failed:
        origin = "memory_replay"
        fact_status = "reconstructed_memory"
    elif replay_boundary_classification == "dream_memory_like_content" and not failed:
        origin = "dream_dmn"
        fact_status = "symbolic_visualization"
    elif replay_boundary_classification == "simulated_memory_like_content" and not failed:
        origin = "simulation"
        fact_status = "simulated_future"
    else:
        origin = "memory_replay"
        fact_status = replay_boundary_classification

    return {
        "memory_replay_observation_passed": not failed,
        "memory_replay_observation_status": "valid" if not failed else "failed",
        "replay_observation_id": str(metadata.get("replay_observation_id", "unknown_id")),
        "replay_source_type": replay_source_type if not failed else None,
        "replay_confidence_state": replay_confidence_state if not failed else None,
        "replay_boundary_classification": replay_boundary_classification if not failed else None,
        "memory_fragment": memory_fragment,
        "replay_context": replay_context,
        "metadata": metadata,
        "origin_summary": {"origin": origin},
        "fact_status_summary": {"fact_status": fact_status},
        "reconstructed_memory": True,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "memory_truth_asserted": False,
        "memory_write_performed": False,
        "long_term_memory_write_allowed": False,
        "quarantine_promotion_allowed": False,
        "self_model_update_allowed": False,
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
        "replay_only": True,
        "candidate_only_fields": candidate_only_fields,
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": [],
        "boundary_flags": boundary_flags,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }

def validate_memory_replay_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    if not observation.get("memory_replay_observation_passed"):
        return {"validation_passed": False, "reason": "observation_failed"}

    must_be_false = [
        "current_external_fact_asserted",
        "real_world_event_asserted",
        "memory_truth_asserted",
        "memory_write_performed",
        "long_term_memory_write_allowed",
        "quarantine_promotion_allowed",
        "self_model_update_allowed",
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

    for key in must_be_false:
        if observation.get(key) is not False:
            return {"validation_passed": False, "reason": f"{key}_must_be_false"}

    must_be_true = [
        "reconstructed_memory",
        "replay_only",
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "quarantine_required",
        "origin_fact_status_required"
    ]

    for key in must_be_true:
        if observation.get(key) is not True:
            return {"validation_passed": False, "reason": f"{key}_must_be_true"}

    return {"validation_passed": True, "reason": "ok"}

def build_memory_replay_to_origin_fact_status_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "origin_fact_status_plan",
        "observation": observation,
        "valid": validate_memory_replay_observation(observation)["validation_passed"]
    }

def build_memory_replay_to_multimodal_event_candidate_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "multimodal_event_candidate_plan",
        "observation": observation,
        "valid": validate_memory_replay_observation(observation)["validation_passed"]
    }

def build_memory_replay_to_cross_modal_binding_preflight_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    valid = validate_memory_replay_observation(observation)["validation_passed"]
    if observation.get("replay_confidence_state") == "replay_low_confidence":
        valid = False
    return {
        "plan_type": "cross_modal_binding_preflight_plan",
        "observation": observation,
        "valid": valid
    }

def build_memory_replay_to_memory_candidate_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "memory_candidate_plan",
        "observation": observation,
        "valid": validate_memory_replay_observation(observation)["validation_passed"]
    }

def build_memory_replay_to_appraisal_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "appraisal_plan",
        "observation": observation,
        "valid": validate_memory_replay_observation(observation)["validation_passed"]
    }

def build_memory_replay_to_agp_input_plan(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "agp_input_plan",
        "observation": observation,
        "valid": validate_memory_replay_observation(observation)["validation_passed"]
    }
