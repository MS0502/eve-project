"""Round1081-1100 read-only multimodal event candidate contract for EVE v3.1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.sensory_observation_contract import (
    validate_sensory_observation,
)
from adapters.visual_observation_schema import (
    SUPPORTED_VISUAL_SOURCE_TYPES as EXTERNAL_VISUAL_SOURCES,
)
from adapters.virtual_visual_observation_schema import (
    SUPPORTED_VIRTUAL_VISUAL_SOURCES as VIRTUAL_VISUAL_SOURCES,
)
from adapters.auditory_observation_schema import (
    SUPPORTED_AUDITORY_SOURCE_TYPES as AUDITORY_SOURCES,
)
from adapters.observation_origin_fact_status_policy import (
    build_origin_fact_status_record,
)

ROUND1081_1100_VERSION = "v3_round1081_1100_multimodal_event_candidate_contract"
FEATURE_TRACK = "read_only_multimodal_event_candidate_contract"
NEXT_IMPLEMENTATION_RECOMMENDATION = "read_only_cross_modal_binding_and_identity_resolution_schema"

SUPPORTED_CANDIDATE_MODALITIES: tuple[str, ...] = (
    "sensory",
    "visual_external",
    "visual_internal_virtual",
    "auditory",
    "tool_state",
    "internal_state",
)

SUPPORTED_CANDIDATE_LINK_TYPES: tuple[str, ...] = (
    "temporal_cooccurrence_candidate",
    "spatial_context_candidate",
    "semantic_relatedness_candidate",
    "source_continuity_candidate",
    "attention_focus_candidate",
    "transcript_visual_reference_candidate",
    "audio_visual_context_candidate",
    "internal_symbolic_reference_candidate",
)

SUPPORTED_EVENT_CANDIDATE_TYPES: tuple[str, ...] = (
    "external_event_candidate",
    "internal_virtual_event_candidate",
    "memory_replay_event_candidate",
    "imagination_event_candidate",
    "simulation_event_candidate",
    "dmn_symbolic_event_candidate",
    "mixed_boundary_candidate",
)

def build_multimodal_event_candidate_contract_summary() -> dict[str, Any]:
    """Returns the current contract design summary for multimodal event candidate generation."""
    return {
        "version": ROUND1081_1100_VERSION,
        "track": FEATURE_TRACK,
        "supported_modalities": list(SUPPORTED_CANDIDATE_MODALITIES),
        "supported_candidate_links": list(SUPPORTED_CANDIDATE_LINK_TYPES),
        "supported_event_candidate_types": list(SUPPORTED_EVENT_CANDIDATE_TYPES),
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
        "korean_fixtures_preserved": ["민석"],
        "invariants": {
            "cross_modal_binding_performed": False,
            "identity_resolution_performed": False,
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
        }
    }

def multimodal_event_candidate_contract_summary() -> dict[str, Any]:
    return build_multimodal_event_candidate_contract_summary()


def _determine_event_candidate_type(observations: list[Mapping[str, Any]]) -> tuple[str, bool, list[str]]:
    has_external = False
    has_virtual = False
    has_memory = False
    has_imagination = False
    has_simulation = False
    has_dmn = False

    uncertainty_flags = []

    for obs in observations:
        mod = obs.get("modality", "")
        # Very simplified heuristic for boundary classification:
        if mod in ("visual_external", "auditory", "sensory", "tool_state"):
            has_external = True
        elif mod == "visual_internal_virtual":
            has_virtual = True
        elif mod == "memory_replay":
            has_memory = True
        elif mod == "imagination":
            has_imagination = True
        elif mod == "simulation":
            has_simulation = True
        elif mod == "dmn_inner_voice" or mod == "internal_state":
            has_dmn = True

    types_count = sum([has_external, has_virtual, has_memory, has_imagination, has_simulation, has_dmn])

    # Virtual visual + DMN is NOT mixed boundary per requirements, it remains internal symbolic
    if has_virtual and has_dmn and not any([has_external, has_memory, has_imagination, has_simulation]):
        mixed_boundary_detected = False
    else:
        mixed_boundary_detected = types_count > 1

    if mixed_boundary_detected:
        uncertainty_flags.append("mixed_boundary_detected")
        return "mixed_boundary_candidate", True, uncertainty_flags

    if has_external:
        return "external_event_candidate", False, uncertainty_flags
    elif has_virtual and not has_dmn:
        return "internal_virtual_event_candidate", False, uncertainty_flags
    elif has_memory:
        return "memory_replay_event_candidate", False, uncertainty_flags
    elif has_imagination:
        return "imagination_event_candidate", False, uncertainty_flags
    elif has_simulation:
        return "simulation_event_candidate", False, uncertainty_flags
    elif has_dmn:
        return "dmn_symbolic_event_candidate", False, uncertainty_flags
    elif has_virtual and has_dmn:
         return "dmn_symbolic_event_candidate", False, uncertainty_flags

    # Default fallback
    return "internal_virtual_event_candidate", False, uncertainty_flags


def build_multimodal_event_candidate(
    observations: list[Mapping[str, Any]] | None = None,
    candidate_links: list[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Builds a read-only multimodal event candidate from a list of observations."""
    observations = list(observations) if observations else []
    candidate_links = list(candidate_links) if candidate_links else []
    metadata = dict(metadata) if metadata else {}

    base = {
        "multimodal_event_candidate_passed": False,
        "multimodal_event_candidate_status": "failed_closed",
        "candidate_id": metadata.get("candidate_id", "unknown_multimodal_candidate"),
        "candidate_type": "unknown",
        "observations": [],
        "candidate_links": [],
        "origin_summary": {},
        "fact_status_summary": {},
        "boundary_classification": "unknown",
        "mixed_boundary_detected": False,
        "external_fact_asserted": False,
        "real_world_event_asserted": False,
        "identity_asserted": False,
        "speaker_identity_asserted": False,
        "person_identity_asserted": False,
        "user_emotion_asserted": False,
        "user_intent_asserted": False,
        "relationship_state_asserted": False,
        "memory_fact_asserted": False,
        "virtual_fact_asserted_as_external": False,
        "imagination_fact_asserted": False,
        "simulation_fact_asserted": False,
        "dmn_fact_asserted": False,
        "cross_modal_binding_performed": False,
        "identity_resolution_performed": False,
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
        "candidate_only_fields": [],
        "uncertainty_flags": [],
        "privacy_flags": [],
        "boundary_flags": [],
        "blocked_reasons": [],
        "warnings": [],
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "quarantine_required": True,
        "origin_fact_status_required": True,
        "event_candidate_only": True,
    }

    if not observations:
        base["blocked_reasons"].append("empty_observations_fail_closed")
        return base

    for obs in observations:
        mod = obs.get("modality", "")
        if mod not in SUPPORTED_CANDIDATE_MODALITIES and mod not in ["memory_replay", "imagination", "simulation", "dmn_inner_voice"]:
            base["blocked_reasons"].append(f"unknown_modality_fails_closed_{mod}")
            return base

    for link in candidate_links:
        l_type = link.get("link_type", "")
        if l_type not in SUPPORTED_CANDIDATE_LINK_TYPES:
            base["blocked_reasons"].append(f"unknown_candidate_link_type_fails_closed_{l_type}")
            return base

    ctype, mixed, unc_flags = _determine_event_candidate_type(observations)
    if ctype not in SUPPORTED_EVENT_CANDIDATE_TYPES:
        base["blocked_reasons"].append(f"unknown_event_candidate_type_fails_closed_{ctype}")
        return base

    base["candidate_type"] = ctype
    base["mixed_boundary_detected"] = mixed
    if mixed:
        base["uncertainty_flags"].extend(unc_flags)
        base["boundary_flags"].append("mixed_boundary_requires_isolation")

    base["observations"] = [deepcopy(obs) for obs in observations]
    base["candidate_links"] = [deepcopy(link) for link in candidate_links]

    base["multimodal_event_candidate_passed"] = True
    base["multimodal_event_candidate_status"] = "read_only_candidate_built"
    base["boundary_classification"] = ctype

    return base

def validate_multimodal_event_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validates the multimodal event candidate for safety invariants."""
    res = dict(candidate)

    res["cross_modal_binding_performed"] = False
    res["identity_resolution_performed"] = False
    res["speaker_recognition_performed"] = False
    res["face_recognition_performed"] = False
    res["ocr_performed"] = False
    res["speech_to_text_performed"] = False
    res["model_loaded"] = False
    res["camera_activated"] = False
    res["microphone_activated"] = False
    res["raw_data_persisted"] = False
    res["memory_write_performed"] = False
    res["long_term_memory_write_allowed"] = False
    res["self_model_update_allowed"] = False
    res["affect_transition_allowed"] = False
    res["hormone_transition_allowed"] = False
    res["agp_bypass_allowed"] = False
    res["fallback_bypass_allowed"] = False
    res["vector_read_performed"] = False
    res["vector_load_performed"] = False
    res["runtime_mutation_performed"] = False
    res["persistence_write_performed"] = False
    res["artifact_created_or_staged"] = False

    res["identity_asserted"] = False
    res["speaker_identity_asserted"] = False
    res["person_identity_asserted"] = False
    res["user_emotion_asserted"] = False
    res["user_intent_asserted"] = False
    res["relationship_state_asserted"] = False
    res["memory_fact_asserted"] = False
    res["virtual_fact_asserted_as_external"] = False
    res["imagination_fact_asserted"] = False
    res["simulation_fact_asserted"] = False
    res["dmn_fact_asserted"] = False
    res["external_fact_asserted"] = False
    res["real_world_event_asserted"] = False

    return res

def build_multimodal_event_to_origin_fact_status_plan(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "origin_fact_status_required": True,
        "mutation_performed": False,
        "plan_status": "origin_fact_status_pending",
    }

def build_multimodal_event_to_memory_candidate_plan(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "memory_gate_required": True,
        "memory_write_performed": False,
        "plan_status": "memory_gate_pending",
    }

def build_multimodal_event_to_appraisal_plan(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "appraisal_required": True,
        "mutation_performed": False,
        "plan_status": "appraisal_pending",
    }

def build_multimodal_event_to_agp_input_plan(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "agp_input_required": True,
        "agp_bypass_allowed": False,
        "plan_status": "agp_input_pending",
    }
