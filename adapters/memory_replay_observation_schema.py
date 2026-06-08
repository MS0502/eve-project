import uuid
import time

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

def build_memory_replay_observation_schema_summary():
    return {
        "schema_name": "memory_replay_observation_schema",
        "supported_replay_source_types": SUPPORTED_REPLAY_SOURCE_TYPES,
        "supported_replay_confidence_states": SUPPORTED_REPLAY_CONFIDENCE_STATES,
        "supported_replay_boundary_classes": SUPPORTED_REPLAY_BOUNDARY_CLASSES,
        "read_only": True,
        "no_mutation": True
    }

def memory_replay_observation_schema_summary():
    return build_memory_replay_observation_schema_summary()

def build_memory_replay_observation(memory_fragment=None, replay_context=None, metadata=None):
    if memory_fragment is None or not memory_fragment:
        return _fail_closed("Empty memory_fragment")

    if replay_context is None:
        replay_context = {}

    replay_source_type = replay_context.get("replay_source_type")
    replay_confidence_state = replay_context.get("replay_confidence_state")
    replay_boundary_classification = replay_context.get("replay_boundary_classification")

    if replay_source_type not in SUPPORTED_REPLAY_SOURCE_TYPES:
        return _fail_closed(f"Unknown replay source type: {replay_source_type}")

    if replay_confidence_state not in SUPPORTED_REPLAY_CONFIDENCE_STATES:
        return _fail_closed(f"Unknown replay confidence state: {replay_confidence_state}")

    if replay_boundary_classification not in SUPPORTED_REPLAY_BOUNDARY_CLASSES:
        return _fail_closed(f"Unknown replay boundary class: {replay_boundary_classification}")

    uncertainty_flags = []
    boundary_flags = []
    blocked_reasons = []
    warnings = []
    candidate_only_fields = ["memory_fragment", "replay_context", "metadata"]

    if replay_confidence_state == "replay_conflict_detected":
        uncertainty_flags.append("conflict_detected")
        boundary_flags.append("conflict_boundary_crossed")

    if replay_boundary_classification == "mixed_replay_boundary":
        boundary_flags.append("mixed_boundary")

    if replay_confidence_state == "replay_low_confidence":
        blocked_reasons.append("low_confidence_blocks_binding")

    return {
        "memory_replay_observation_passed": True,
        "memory_replay_observation_status": "built",
        "replay_observation_id": f"mrep_{uuid.uuid4().hex[:8]}",
        "replay_source_type": replay_source_type,
        "replay_confidence_state": replay_confidence_state,
        "replay_boundary_classification": replay_boundary_classification,
        "memory_fragment": memory_fragment,
        "replay_context": replay_context,
        "metadata": metadata or {},
        "origin_summary": {},
        "fact_status_summary": {},
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

def _fail_closed(reason):
    return {
        "memory_replay_observation_passed": False,
        "memory_replay_observation_status": f"failed: {reason}",
        "replay_observation_id": None,
        "replay_source_type": None,
        "replay_confidence_state": None,
        "replay_boundary_classification": None,
        "memory_fragment": {},
        "replay_context": {},
        "metadata": {},
        "origin_summary": {},
        "fact_status_summary": {},
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
        "candidate_only_fields": [],
        "uncertainty_flags": [],
        "privacy_flags": [],
        "boundary_flags": [],
        "blocked_reasons": [reason],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }

def validate_memory_replay_observation(observation):
    if not observation:
        return False
    return observation.get("memory_replay_observation_passed", False)

def build_memory_replay_to_origin_fact_status_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "origin_fact_status",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }

def build_memory_replay_to_multimodal_event_candidate_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "multimodal_event_candidate",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }

def build_memory_replay_to_cross_modal_binding_preflight_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "cross_modal_binding_preflight",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }

def build_memory_replay_to_memory_candidate_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "memory_candidate",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }

def build_memory_replay_to_appraisal_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "appraisal",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }

def build_memory_replay_to_agp_input_plan(observation):
    if not validate_memory_replay_observation(observation):
        return None
    return {
        "plan_type": "agp_input",
        "status": "blocked" if "low_confidence_blocks_binding" in observation.get("blocked_reasons", []) else "ready"
    }
