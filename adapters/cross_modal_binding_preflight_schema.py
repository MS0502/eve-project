"""Read-only cross-modal binding preflight schema for Round1101-1120."""

import json
from typing import Any, Dict, List, Optional

VERSION = "v3_round1101_1120_cross_modal_binding_preflight_schema"

SUPPORTED_BINDING_HYPOTHESIS_TYPES = {
    "temporal_alignment_hypothesis",
    "spatial_alignment_hypothesis",
    "semantic_alignment_hypothesis",
    "source_continuity_hypothesis",
    "transcript_reference_hypothesis",
    "audio_visual_context_hypothesis",
    "internal_symbolic_alignment_hypothesis",
    "memory_replay_alignment_hypothesis"
}

SUPPORTED_PREFLIGHT_DECISIONS = {
    "eligible_for_future_binding",
    "blocked_mixed_boundary",
    "blocked_identity_resolution_required",
    "blocked_low_confidence",
    "blocked_privacy_risk",
    "blocked_origin_conflict",
    "blocked_fact_status_conflict",
    "blocked_memory_replay_as_external",
    "blocked_virtual_as_external",
    "blocked_imagination_as_fact",
    "blocked_simulation_as_current_fact",
    "blocked_unknown_candidate"
}

def cross_modal_binding_preflight_schema_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "feature_track": "read_only_cross_modal_binding_preflight_schema",
        "supported_binding_hypothesis_types": sorted(list(SUPPORTED_BINDING_HYPOTHESIS_TYPES)),
        "supported_preflight_decisions": sorted(list(SUPPORTED_PREFLIGHT_DECISIONS))
    }

def build_cross_modal_binding_preflight_schema_summary() -> Dict[str, Any]:
    return cross_modal_binding_preflight_schema_summary()

def build_cross_modal_binding_preflight(
    candidate: Optional[Dict[str, Any]] = None,
    binding_hypotheses: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if candidate is None:
        candidate = {}
    if binding_hypotheses is None:
        binding_hypotheses = []
    if metadata is None:
        metadata = {}

    failed = False
    decision = "eligible_for_future_binding"
    blocked_reasons = []

    # Check for empty or unknown candidate
    if not candidate:
        failed = True
        decision = "blocked_unknown_candidate"
        blocked_reasons.append("empty_candidate")
    elif not candidate.get("multimodal_event_candidate_passed"):
        failed = True
        decision = "blocked_unknown_candidate"
        blocked_reasons.append("invalid_candidate")

    for hypothesis in binding_hypotheses:
        if hypothesis not in SUPPORTED_BINDING_HYPOTHESIS_TYPES:
            failed = True
            decision = "blocked_unknown_candidate"
            blocked_reasons.append(f"unknown_hypothesis_{hypothesis}")

    if not failed:
        candidate_type = candidate.get("candidate_type")
        uncertainty_flags = candidate.get("uncertainty_flags", [])

        # Identity resolution blocked
        if metadata.get("requires_identity_resolution"):
            failed = True
            decision = "blocked_identity_resolution_required"
            blocked_reasons.append("identity_resolution_required")

        # Face/Voice matching blocked
        if metadata.get("attempt_face_voice_match"):
            failed = True
            decision = "blocked_identity_resolution_required"
            blocked_reasons.append("face_voice_match_blocked")

        # Mixed boundary blocked
        if candidate_type == "mixed_boundary_candidate" and not metadata.get("explicitly_future_separated"):
            failed = True
            decision = "blocked_mixed_boundary"
            blocked_reasons.append("mixed_boundary")

        if "mixed_boundary_uncertainty" in uncertainty_flags and not metadata.get("explicitly_future_separated"):
            failed = True
            decision = "blocked_mixed_boundary"
            blocked_reasons.append("mixed_boundary")

        # Origin conflict blocked
        if metadata.get("origin_conflict"):
            failed = True
            decision = "blocked_origin_conflict"
            blocked_reasons.append("origin_conflict")

        # Fact status conflict blocked
        if metadata.get("fact_status_conflict"):
            failed = True
            decision = "blocked_fact_status_conflict"
            blocked_reasons.append("fact_status_conflict")

        # Low confidence blocked
        if metadata.get("low_confidence"):
            failed = True
            decision = "blocked_low_confidence"
            blocked_reasons.append("low_confidence")

        # Privacy risk blocked
        if metadata.get("privacy_risk"):
            failed = True
            decision = "blocked_privacy_risk"
            blocked_reasons.append("privacy_risk")

        # Memory replay as external blocked
        if candidate_type == "memory_replay_event_candidate" and metadata.get("intended_as_external"):
            failed = True
            decision = "blocked_memory_replay_as_external"
            blocked_reasons.append("memory_replay_as_external")

        # Virtual as external blocked
        if candidate_type == "internal_virtual_event_candidate" and metadata.get("intended_as_external"):
            failed = True
            decision = "blocked_virtual_as_external"
            blocked_reasons.append("virtual_as_external")

        # Imagination as fact blocked
        if candidate_type == "imagination_event_candidate" and metadata.get("intended_as_memory"):
            failed = True
            decision = "blocked_imagination_as_fact"
            blocked_reasons.append("imagination_as_fact")

        # Simulation as current fact blocked
        if candidate_type == "simulation_event_candidate" and metadata.get("intended_as_current"):
            failed = True
            decision = "blocked_simulation_as_current_fact"
            blocked_reasons.append("simulation_as_current_fact")

    if candidate.get("candidate_type") == "dmn_symbolic_event_candidate":
        # DMN Symbolic candidate remains internal/symbolic
        pass

    return {
        "preflight_passed": not failed,
        "preflight_decision": decision,
        "blocked_reasons": blocked_reasons,
        "future_binding_attempt_allowed": not failed,
        "binding_performed": False,
        "cross_modal_binding_performed": False,
        "identity_resolution_performed": False,
        "identity_asserted": False,
        "speaker_identity_asserted": False,
        "person_identity_asserted": False,
        "face_voice_match_asserted": False,
        "user_emotion_asserted": False,
        "user_intent_asserted": False,
        "relationship_state_asserted": False,
        "memory_fact_asserted": False,
        "external_fact_asserted": False,
        "real_world_event_asserted": False,
        "virtual_fact_asserted_as_external": False,
        "imagination_fact_asserted": False,
        "simulation_fact_asserted": False,
        "dmn_fact_asserted": False,
        "speaker_recognition_performed": False,
        "face_recognition_performed": False,
        "ocr_performed": False,
        "speech_to_text_performed": False,
        "model_loaded": False,
        "camera_activated": False,
        "microphone_activated": False,
        "raw_data_persisted": False,
        "memory_write_performed": False,
        "long_term_memory_write_allowed": False,
        "self_model_update_allowed": False,
        "affect_transition_allowed": False,
        "hormone_transition_allowed": False,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "vector_read_performed": False,
        "vector_load_performed": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
        "artifact_created_or_staged": False,
        "global_synchrony_produced": False
    }

def validate_cross_modal_binding_preflight(preflight: Dict[str, Any]) -> Dict[str, Any]:
    must_be_false = [
        "binding_performed",
        "cross_modal_binding_performed",
        "identity_resolution_performed",
        "identity_asserted",
        "speaker_identity_asserted",
        "person_identity_asserted",
        "face_voice_match_asserted",
        "user_emotion_asserted",
        "user_intent_asserted",
        "relationship_state_asserted",
        "memory_fact_asserted",
        "external_fact_asserted",
        "real_world_event_asserted",
        "virtual_fact_asserted_as_external",
        "imagination_fact_asserted",
        "simulation_fact_asserted",
        "dmn_fact_asserted",
        "speaker_recognition_performed",
        "face_recognition_performed",
        "ocr_performed",
        "speech_to_text_performed",
        "model_loaded",
        "camera_activated",
        "microphone_activated",
        "raw_data_persisted",
        "memory_write_performed",
        "long_term_memory_write_allowed",
        "self_model_update_allowed",
        "affect_transition_allowed",
        "hormone_transition_allowed",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
        "vector_read_performed",
        "vector_load_performed",
        "runtime_mutation_performed",
        "persistence_write_performed",
        "artifact_created_or_staged",
        "global_synchrony_produced"
    ]

    for key in must_be_false:
        if preflight.get(key, True) is not False:
            return {"validation_passed": False, "reason": f"{key}_not_false"}

    if preflight.get("future_binding_attempt_allowed", False) and preflight.get("preflight_decision") != "eligible_for_future_binding":
        return {"validation_passed": False, "reason": "allowed_but_not_eligible"}

    return {"validation_passed": True, "reason": "ok"}

def build_cross_modal_binding_to_origin_fact_status_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "origin_fact_status_plan",
        "preflight": preflight,
        "valid": validate_cross_modal_binding_preflight(preflight)["validation_passed"]
    }

def build_cross_modal_binding_to_memory_candidate_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "memory_candidate_plan",
        "preflight": preflight,
        "valid": validate_cross_modal_binding_preflight(preflight)["validation_passed"]
    }

def build_cross_modal_binding_to_appraisal_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "appraisal_plan",
        "preflight": preflight,
        "valid": validate_cross_modal_binding_preflight(preflight)["validation_passed"]
    }

def build_cross_modal_binding_to_agp_input_plan(preflight: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "agp_input_plan",
        "preflight": preflight,
        "valid": validate_cross_modal_binding_preflight(preflight)["validation_passed"]
    }
