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

def build_cross_modal_binding_preflight(candidate: Optional[Dict[str, Any]] = None, binding_hypotheses: Optional[List[Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if binding_hypotheses is None:
        binding_hypotheses = []
    if metadata is None:
        metadata = {}

    failed = False
    status = "pending"
    decision = "eligible_for_future_binding"
    blocked_reasons = []
    warnings = []
    uncertainty_flags = []
    privacy_flags = []
    boundary_flags = []

    candidate_id = "unknown"
    candidate_type = "unknown"
    origin_summary = {}
    fact_status_summary = {}
    boundary_classification = "unknown"
    mixed_boundary_detected = False

    if not candidate:
        failed = True
        status = "failed"
        decision = "blocked_unknown_candidate"
        blocked_reasons.append("empty_candidate")
    elif not isinstance(candidate, dict) or "candidate_type" not in candidate:
        failed = True
        status = "failed"
        decision = "blocked_unknown_candidate"
        blocked_reasons.append("unknown_candidate")
    else:
        candidate_id = candidate.get("candidate_id", "unknown")
        candidate_type = candidate.get("candidate_type", "unknown")

        # Determine mixed boundary explicitly
        if candidate_type == "mixed_boundary_candidate" or "mixed_boundary" in str(candidate):
            mixed_boundary_detected = True
            boundary_classification = "mixed_boundary"

        # Read specific metadata to block accordingly
        if metadata.get("low_confidence", False):
            failed = True
            status = "blocked"
            decision = "blocked_low_confidence"
            blocked_reasons.append("low_confidence")

        elif metadata.get("privacy_risk", False):
            failed = True
            status = "blocked"
            decision = "blocked_privacy_risk"
            blocked_reasons.append("privacy_risk")
            privacy_flags.append("privacy_risk_detected")

        elif metadata.get("origin_conflict", False):
            failed = True
            status = "blocked"
            decision = "blocked_origin_conflict"
            blocked_reasons.append("origin_conflict")

        elif metadata.get("fact_status_conflict", False):
            failed = True
            status = "blocked"
            decision = "blocked_fact_status_conflict"
            blocked_reasons.append("fact_status_conflict")

        elif metadata.get("identity_resolution_required", False):
            failed = True
            status = "blocked"
            decision = "blocked_identity_resolution_required"
            blocked_reasons.append("identity_resolution_required")

        elif metadata.get("face_voice_match_attempt", False):
            failed = True
            status = "blocked"
            decision = "blocked_identity_resolution_required"
            blocked_reasons.append("face_voice_match_blocked")

        elif candidate_type == "memory_replay_event_candidate" and metadata.get("intended_as_external", False):
            failed = True
            status = "blocked"
            decision = "blocked_memory_replay_as_external"
            blocked_reasons.append("memory_replay_cannot_be_external")

        elif candidate_type == "internal_virtual_event_candidate" and metadata.get("intended_as_external", False):
            failed = True
            status = "blocked"
            decision = "blocked_virtual_as_external"
            blocked_reasons.append("virtual_cannot_be_external")

        elif candidate_type == "imagination_event_candidate" and metadata.get("intended_as_fact", False):
            failed = True
            status = "blocked"
            decision = "blocked_imagination_as_fact"
            blocked_reasons.append("imagination_cannot_be_fact")

        elif candidate_type == "simulation_event_candidate" and metadata.get("intended_as_current_fact", False):
            failed = True
            status = "blocked"
            decision = "blocked_simulation_as_current_fact"
            blocked_reasons.append("simulation_cannot_be_current_fact")

        elif candidate_type == "dmn_symbolic_event_candidate" and metadata.get("intended_as_external", False):
            failed = True
            status = "blocked"
            decision = "blocked_virtual_as_external" # Mapping dmn as virtual blocked
            blocked_reasons.append("dmn_cannot_be_external")

        elif mixed_boundary_detected and not metadata.get("explicitly_future_separated", False):
            failed = True
            status = "blocked"
            decision = "blocked_mixed_boundary"
            blocked_reasons.append("mixed_boundary_not_separated")

        else:
            status = "passed"
            decision = "eligible_for_future_binding"
            boundary_classification = "clear"

    for hyp in binding_hypotheses:
        htype = hyp.get("hypothesis_type")
        if htype not in SUPPORTED_BINDING_HYPOTHESIS_TYPES:
            failed = True
            status = "failed"
            decision = "blocked_unknown_candidate" # General fallback for unknown
            blocked_reasons.append(f"unknown_hypothesis_type_{htype}")

    future_binding_attempt_allowed = False
    if decision == "eligible_for_future_binding":
        future_binding_attempt_allowed = True

    return {
        "cross_modal_binding_preflight_passed": not failed,
        "cross_modal_binding_preflight_status": status,
        "preflight_decision": decision,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "binding_hypotheses": binding_hypotheses,
        "origin_summary": origin_summary,
        "fact_status_summary": fact_status_summary,
        "boundary_classification": boundary_classification,
        "mixed_boundary_detected": mixed_boundary_detected,
        "future_binding_attempt_allowed": future_binding_attempt_allowed,

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

        "candidate_only_fields": ["preflight_decision", "future_binding_attempt_allowed"],
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": privacy_flags,
        "boundary_flags": boundary_flags,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,

        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True,
        "preflight_only": True
    }

def validate_cross_modal_binding_preflight(preflight: Dict[str, Any]) -> Dict[str, Any]:
    if preflight.get("cross_modal_binding_preflight_status") == "failed":
        return {"validation_passed": False, "reason": "preflight_failed"}

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
        "artifact_created_or_staged"
    ]

    for key in must_be_false:
        if preflight.get(key, True) is not False:
            return {"validation_passed": False, "reason": f"{key}_not_false"}

    must_be_true = [
        "appraisal_required",
        "agp_input_required",
        "memory_gate_required",
        "quarantine_required",
        "origin_fact_status_required",
        "preflight_only"
    ]

    for key in must_be_true:
        if preflight.get(key, False) is not True:
            return {"validation_passed": False, "reason": f"{key}_not_true"}

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
