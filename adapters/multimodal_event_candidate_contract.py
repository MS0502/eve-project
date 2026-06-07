"""Read-only multimodal event candidate contract for Round1081-1100."""

import json
from typing import Any, Dict, List, Optional

VERSION = "v3_round1081_1100_multimodal_event_candidate_contract"

SUPPORTED_MODALITIES = {
    "sensory",
    "visual_external",
    "visual_internal_virtual",
    "auditory",
    "tool_state",
    "internal_state"
}

SUPPORTED_LINK_TYPES = {
    "temporal_cooccurrence_candidate",
    "spatial_context_candidate",
    "semantic_relatedness_candidate",
    "source_continuity_candidate",
    "attention_focus_candidate",
    "transcript_visual_reference_candidate",
    "audio_visual_context_candidate",
    "internal_symbolic_reference_candidate"
}

SUPPORTED_CANDIDATE_TYPES = {
    "external_event_candidate",
    "internal_virtual_event_candidate",
    "memory_replay_event_candidate",
    "imagination_event_candidate",
    "simulation_event_candidate",
    "dmn_symbolic_event_candidate",
    "mixed_boundary_candidate"
}

def multimodal_event_candidate_contract_summary() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "feature_track": "read_only_multimodal_event_candidate_contract",
        "supported_modalities": sorted(list(SUPPORTED_MODALITIES)),
        "supported_link_types": sorted(list(SUPPORTED_LINK_TYPES)),
        "supported_candidate_types": sorted(list(SUPPORTED_CANDIDATE_TYPES))
    }

def build_multimodal_event_candidate_contract_summary() -> Dict[str, Any]:
    return multimodal_event_candidate_contract_summary()

def build_multimodal_event_candidate(observations: Optional[List[Dict[str, Any]]] = None, candidate_links: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if observations is None:
        observations = []
    if candidate_links is None:
        candidate_links = []
    if metadata is None:
        metadata = {}

    failed = False
    blocked_reasons = []

    if not observations:
        failed = True
        blocked_reasons.append("empty_observations")

    modalities = set()
    sources = set()
    for obs in observations:
        modality = obs.get("modality")
        if modality not in SUPPORTED_MODALITIES:
            failed = True
            blocked_reasons.append(f"unknown_modality_{modality}")
        else:
            modalities.add(modality)

        source = obs.get("source")
        if source:
            sources.add(source)

    for link in candidate_links:
        if link not in SUPPORTED_LINK_TYPES:
            failed = True
            blocked_reasons.append(f"unknown_link_type_{link}")

    # Determine default candidate type if not given in metadata.
    # We apply specific logic rules to candidate types.
    candidate_type = metadata.get("candidate_type")

    uncertainty_flags = []
    boundary_flags = []

    if not failed:
        if candidate_type and candidate_type not in SUPPORTED_CANDIDATE_TYPES:
            failed = True
            blocked_reasons.append(f"unknown_candidate_type_{candidate_type}")

        has_virtual = "visual_internal_virtual" in modalities
        has_external_aud = "auditory" in modalities and any(s != "dmn_inner_voice_candidate" for s in sources)
        has_dmn = "dmn_inner_voice_candidate" in sources
        has_external_vis = "visual_external" in modalities

        if candidate_type is None:
            if has_virtual and has_external_aud:
                candidate_type = "mixed_boundary_candidate"
            elif has_virtual and has_dmn:
                candidate_type = "dmn_symbolic_event_candidate"
            elif has_external_vis and has_external_aud:
                candidate_type = "external_event_candidate"
            elif has_virtual:
                candidate_type = "internal_virtual_event_candidate"
            elif has_external_vis or has_external_aud:
                candidate_type = "external_event_candidate"
            else:
                candidate_type = "external_event_candidate"

        # Explicit overrides/enforcements
        if has_virtual and has_external_aud and not metadata.get("explicitly_separated", False):
            candidate_type = "mixed_boundary_candidate"

        if has_virtual and has_dmn:
            candidate_type = "dmn_symbolic_event_candidate"

        if candidate_type == "external_event_candidate" and "memory_replay" in metadata:
            candidate_type = "memory_replay_event_candidate"

        if candidate_type == "memory_replay_event_candidate" and "external" in str(metadata):
             # Handled correctly by remaining memory_replay
             pass

    if candidate_type == "mixed_boundary_candidate":
        uncertainty_flags.append("mixed_boundary_uncertainty")
        boundary_flags.append("cross_domain_boundary_detected")

    if candidate_type == "memory_replay_event_candidate" and metadata.get("intended_as_external", False):
        # Even if intended as external, memory replay cannot become external fact, it remains replay
        pass

    if candidate_type == "imagination_event_candidate" and metadata.get("intended_as_memory", False):
        pass

    if candidate_type == "simulation_event_candidate" and metadata.get("intended_as_current", False):
        pass


    return {
        "multimodal_event_candidate_passed": not failed,
        "blocked_reasons": blocked_reasons,
        "candidate_type": candidate_type if not failed else None,
        "observations": observations,
        "candidate_links": candidate_links,
        "metadata": metadata,
        "uncertainty_flags": uncertainty_flags,
        "boundary_flags": boundary_flags,
        "identity_asserted": False,
        "speaker_identity_asserted": False,
        "person_identity_asserted": False,
        "user_emotion_asserted": False,
        "user_intent_asserted": False,
        "relationship_state_asserted": False,
        "memory_fact_asserted": False,
        "cross_modal_binding_performed": False,
        "identity_resolution_performed": False,

        # OCR/STT/recognition/model/device flags must always be false.
        "ocr_performed": False,
        "stt_performed": False,
        "recognition_performed": False,
        "model_loaded": False,
        "device_activated": False,

        # memory/self/affect/hormone mutation flags must always be false.
        "memory_written": False,
        "self_model_updated": False,
        "affect_transitioned": False,
        "hormone_transitioned": False,

        # persistence/runtime/vector/artifact flags must always be false.
        "persistence_written": False,
        "runtime_mutated": False,
        "vector_written": False,
        "artifact_written": False,

        # AGP/fallback bypass flags must always be false.
        "agp_bypassed": False,
        "fallback_bypassed": False,
        "all_axis_activation_produced": False,
        "global_synchrony_produced": False,
        "is_candidate_only": True
    }

def validate_multimodal_event_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not candidate.get("multimodal_event_candidate_passed", False):
        return {"validation_passed": False, "reason": "candidate_failed_creation"}

    must_be_false = [
        "identity_asserted",
        "speaker_identity_asserted",
        "person_identity_asserted",
        "user_emotion_asserted",
        "user_intent_asserted",
        "relationship_state_asserted",
        "memory_fact_asserted",
        "cross_modal_binding_performed",
        "identity_resolution_performed",
        "ocr_performed",
        "stt_performed",
        "recognition_performed",
        "model_loaded",
        "device_activated",
        "memory_written",
        "self_model_updated",
        "affect_transitioned",
        "hormone_transitioned",
        "persistence_written",
        "runtime_mutated",
        "vector_written",
        "artifact_written",
        "agp_bypassed",
        "fallback_bypassed",
        "all_axis_activation_produced",
        "global_synchrony_produced"
    ]

    for key in must_be_false:
        if candidate.get(key, True) is not False:
            return {"validation_passed": False, "reason": f"{key}_not_false"}

    if not candidate.get("is_candidate_only", False):
         return {"validation_passed": False, "reason": "not_candidate_only"}

    return {"validation_passed": True, "reason": "ok"}

def build_multimodal_event_to_origin_fact_status_plan(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "origin_fact_status_plan",
        "candidate": candidate,
        "valid": validate_multimodal_event_candidate(candidate)["validation_passed"]
    }

def build_multimodal_event_to_memory_candidate_plan(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "memory_candidate_plan",
        "candidate": candidate,
        "valid": validate_multimodal_event_candidate(candidate)["validation_passed"]
    }

def build_multimodal_event_to_appraisal_plan(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "appraisal_plan",
        "candidate": candidate,
        "valid": validate_multimodal_event_candidate(candidate)["validation_passed"]
    }

def build_multimodal_event_to_agp_input_plan(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan_type": "agp_input_plan",
        "candidate": candidate,
        "valid": validate_multimodal_event_candidate(candidate)["validation_passed"]
    }
