import uuid

SUPPORTED_VIRTUAL_SOURCE_TYPES = {
    "internal_virtual_scene_candidate",
    "internal_virtual_object_candidate",
    "internal_virtual_agent_candidate",
    "internal_virtual_location_candidate",
    "internal_virtual_action_candidate",
    "internal_virtual_interaction_candidate",
    "symbolic_virtual_visualization_candidate",
    "dmn_virtual_scene_candidate",
    "simulation_virtual_scene_candidate",
    "dream_virtual_scene_candidate"
}

SUPPORTED_VIRTUAL_BOUNDARY_CLASSES = {
    "internal_virtual_observation",
    "symbolic_virtual_observation",
    "dmn_virtual_observation",
    "simulated_virtual_observation",
    "dream_virtual_observation",
    "mixed_virtual_external_boundary"
}

SUPPORTED_VIRTUAL_CONFIDENCE_STATES = {
    "virtual_unverified",
    "virtual_low_confidence",
    "virtual_medium_confidence",
    "virtual_high_confidence_but_not_external_fact",
    "virtual_conflict_detected",
    "virtual_origin_unknown"
}


def _fail_closed(reason):
    return {
        "virtual_world_observation_passed": False,
        "virtual_world_observation_status": f"REJECTED: {reason}",
        "virtual_observation_id": str(uuid.uuid4()),
        "virtual_source_type": "unknown",
        "virtual_boundary_classification": "unknown",
        "virtual_confidence_state": "unknown",
        "virtual_scene": {},
        "observer_context": {},
        "metadata": {},
        "origin_summary": {},
        "fact_status_summary": {},
        "internal_virtual_observation": True,
        "external_reality_asserted": False,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "virtual_state_mutation_performed": False,
        "virtual_world_write_allowed": False,
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
        "observation_only": True,
        "candidate_only_fields": [],
        "uncertainty_flags": [reason],
        "privacy_flags": [],
        "virtual_boundary_flags": [],
        "blocked_reasons": [reason],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }


def build_virtual_world_observation(virtual_scene=None, observer_context=None, metadata=None):
    if not virtual_scene:
        return _fail_closed("empty virtual_scene")

    source_type = virtual_scene.get("virtual_source_type")
    if source_type not in SUPPORTED_VIRTUAL_SOURCE_TYPES:
        return _fail_closed(f"unknown virtual source type: {source_type}")

    boundary_class = virtual_scene.get("virtual_boundary_classification")
    if boundary_class not in SUPPORTED_VIRTUAL_BOUNDARY_CLASSES:
        return _fail_closed(f"unknown virtual boundary class: {boundary_class}")

    confidence_state = virtual_scene.get("virtual_confidence_state")
    if confidence_state not in SUPPORTED_VIRTUAL_CONFIDENCE_STATES:
        return _fail_closed(f"unknown confidence state: {confidence_state}")

    candidate_only_fields = []
    uncertainty_flags = []
    virtual_boundary_flags = []
    blocked_reasons = []

    if boundary_class == "mixed_virtual_external_boundary":
        virtual_boundary_flags.append("mixed_virtual_external_boundary_detected")
        blocked_reasons.append("mixed boundary blocks external assertion")

    if source_type == "symbolic_virtual_visualization_candidate" or boundary_class == "symbolic_virtual_observation":
        candidate_only_fields.append("symbolic_virtual_observation")

    if source_type == "dmn_virtual_scene_candidate" or boundary_class == "dmn_virtual_observation":
        candidate_only_fields.append("dmn_virtual_observation")

    if source_type == "simulation_virtual_scene_candidate":
        blocked_reasons.append("simulation virtual scene cannot become current fact")

    if source_type == "dream_virtual_scene_candidate":
        blocked_reasons.append("dream virtual scene cannot become external fact")

    if source_type == "internal_virtual_agent_candidate":
        blocked_reasons.append("virtual agent candidate cannot assert person identity")
        candidate_only_fields.append("identity")

    if source_type == "internal_virtual_interaction_candidate":
        blocked_reasons.append("virtual interaction candidate cannot assert relationship state")
        candidate_only_fields.append("relationship_state")

    if source_type == "internal_virtual_action_candidate":
        blocked_reasons.append("virtual action candidate cannot mutate world state")

    if confidence_state == "virtual_high_confidence_but_not_external_fact":
        blocked_reasons.append("high confidence virtual observation still cannot become external fact")

    if confidence_state == "virtual_conflict_detected":
        uncertainty_flags.append("virtual_conflict")
        virtual_boundary_flags.append("virtual_conflict")

    if confidence_state == "virtual_origin_unknown":
        uncertainty_flags.append("virtual_origin_unknown")

    return {
        "virtual_world_observation_passed": True,
        "virtual_world_observation_status": "VALIDATED",
        "virtual_observation_id": str(uuid.uuid4()),
        "virtual_source_type": source_type,
        "virtual_boundary_classification": boundary_class,
        "virtual_confidence_state": confidence_state,
        "virtual_scene": virtual_scene,
        "observer_context": observer_context or {},
        "metadata": metadata or {},
        "origin_summary": {},
        "fact_status_summary": {},
        "internal_virtual_observation": True,
        "external_reality_asserted": False,
        "current_external_fact_asserted": False,
        "real_world_event_asserted": False,
        "virtual_state_mutation_performed": False,
        "virtual_world_write_allowed": False,
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
        "observation_only": True,
        "candidate_only_fields": candidate_only_fields,
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": [],
        "virtual_boundary_flags": virtual_boundary_flags,
        "blocked_reasons": blocked_reasons,
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True
    }


def validate_virtual_world_observation(observation):
    if not observation.get("virtual_world_observation_passed"):
        return False
    if observation.get("external_reality_asserted"):
        return False
    if observation.get("virtual_state_mutation_performed"):
        return False
    return True


def build_virtual_world_to_origin_fact_status_plan(observation):
    return {
        "source": "virtual_world",
        "target": "origin_fact_status",
        "status": "pending_gate"
    }


def build_virtual_world_to_multimodal_event_candidate_plan(observation):
    return {
        "source": "virtual_world",
        "target": "multimodal_event_candidate",
        "status": "pending_gate"
    }


def build_virtual_world_to_cross_modal_binding_preflight_plan(observation):
    return {
        "source": "virtual_world",
        "target": "cross_modal_binding_preflight",
        "status": "pending_gate"
    }


def build_virtual_world_to_memory_candidate_plan(observation):
    return {
        "source": "virtual_world",
        "target": "memory_candidate",
        "status": "pending_gate"
    }


def build_virtual_world_to_appraisal_plan(observation):
    return {
        "source": "virtual_world",
        "target": "appraisal",
        "status": "pending_gate"
    }


def build_virtual_world_to_agp_input_plan(observation):
    return {
        "source": "virtual_world",
        "target": "agp_input",
        "status": "pending_gate"
    }


def build_virtual_world_observation_contract_summary():
    return {
        "supported_sources": list(SUPPORTED_VIRTUAL_SOURCE_TYPES),
        "supported_boundaries": list(SUPPORTED_VIRTUAL_BOUNDARY_CLASSES),
        "supported_confidences": list(SUPPORTED_VIRTUAL_CONFIDENCE_STATES),
        "read_only": True
    }


def virtual_world_observation_contract_summary():
    return build_virtual_world_observation_contract_summary()
