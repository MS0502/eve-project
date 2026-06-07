"""Round1061-1080 read-only auditory observation schema for EVE v3.1."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.sensory_observation_contract import (
    build_sensory_memory_candidate_plan,
    build_sensory_observation,
    build_sensory_observation_contract_summary,
    build_sensory_observation_to_agp_input_plan,
    validate_sensory_observation,
    AUDITORY_SOURCE_TYPES as SENSORY_AUDITORY_SOURCE_TYPES,
)
from adapters.observation_origin_fact_status_policy import (
    build_origin_fact_status_record,
)

ROUND1061_1080_VERSION = "v3_round1061_1080_auditory_observation_schema"
FEATURE_TRACK = "read_only_auditory_observation_schema"
NEXT_IMPLEMENTATION_RECOMMENDATION = "read_only_multimodal_sensory_fusion_contract"

SUPPORTED_AUDITORY_SOURCE_TYPES: tuple[str, ...] = (
    "voice_message_candidate",
    "microphone_frame_candidate",
    "speech_transcript_candidate",
    "environmental_sound_candidate",
    "notification_sound_candidate",
    "tool_audio_event_candidate",
    "dmn_inner_voice_candidate",
)

SUPPORTED_OBSERVED_FEATURE_GROUPS: tuple[str, ...] = (
    "acoustic_features",
    "speech_presence_features",
    "transcript_metadata_features",
    "prosody_features",
    "environmental_sound_features",
    "notification_sound_features",
    "audio_context_features",
    "safety_relevance_features",
)

SUPPORTED_INFERRED_FEATURE_GROUPS: tuple[str, ...] = (
    "speaker_identity_candidate",
    "speaker_role_candidate",
    "speech_content_candidate",
    "prosody_state_candidate",
    "emotion_candidate",
    "intent_candidate",
    "urgency_candidate",
    "environmental_event_candidate",
    "auditory_salience_candidate",
)

AUDITORY_FALSE_INVARIANT_FLAGS = {
    "raw_data_persisted": False,
    "raw_audio_persisted": False,
    "raw_video_persisted": False,
    "microphone_activated": False,
    "speech_to_text_performed": False,
    "speaker_recognition_performed": False,
    "audio_model_loaded": False,
    "voice_biometric_asserted": False,
    "speaker_identity_asserted": False,
    "user_emotion_asserted": False,
    "user_intent_asserted": False,
    "relationship_state_asserted": False,
    "memory_fact_asserted": False,
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

AUDITORY_TRUE_INVARIANT_FLAGS = {
    "appraisal_required": True,
    "agp_input_required": True,
    "memory_gate_required": True,
    "sensory_contract_required": True,
    "origin_fact_status_required": True,
}

def _copy_mapping(m: Mapping[str, Any] | None) -> dict[str, Any]:
    if not m:
        return {}
    return deepcopy(dict(m))

def build_auditory_observation(
    source: str,
    observed_features: Mapping[str, Any] | None = None,
    inferred_features: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only auditory observation candidate from structured data only."""

    observed = _copy_mapping(observed_features)
    inferred = _copy_mapping(inferred_features)
    meta = _copy_mapping(metadata)

    blocked_reasons: list[str] = []
    warnings: list[str] = []
    privacy_flags: list[str] = []
    uncertainty_flags: list[str] = []
    candidate_only_fields: list[str] = []

    passed = True
    status = "auditory_observation_read_only_candidate"

    if source not in SUPPORTED_AUDITORY_SOURCE_TYPES:
        passed = False
        status = "auditory_observation_failed_closed"
        blocked_reasons.append(f"unknown_auditory_source_{source}")

    if meta.get("raw_audio_data_present"):
        blocked_reasons.append("raw_audio_data_must_not_enter_auditory_observation_schema")
        passed = False
        status = "auditory_observation_failed_closed"

    origin = "external_audio"
    fact_status = "observed_external"

    if source == "dmn_inner_voice_candidate":
        origin = "dream_dmn"
        fact_status = "symbolic_visualization"

    if source in ("voice_message_candidate", "microphone_frame_candidate", "speech_transcript_candidate"):
        privacy_flags.append("voice_or_speech_data_requires_privacy_handling")

    if meta.get("private_audio"):
        privacy_flags.append("private_audio_flag_set")

    overall_confidence = meta.get("confidence", {}).get("overall", "unknown")
    if overall_confidence in ("low", "unknown", "very_low"):
        uncertainty_flags.append(f"low_confidence_auditory_inference_{overall_confidence}")

    for field in SUPPORTED_INFERRED_FEATURE_GROUPS:
        if field in inferred:
            candidate_only_fields.append(field)

    if "speech_content" in observed:
        warnings.append("speech_content_should_be_candidate_unless_provided_as_metadata")
    if "speech_content_candidate" in inferred and not meta.get("transcript_metadata_provided"):
        candidate_only_fields.append("speech_content")

    quarantine_required = True
    if meta.get("malicious_content_flag"):
        warnings.append("malicious_auditory_content_flag_detected")

    event_candidate_allowed = passed
    thought_candidate_allowed = passed
    memory_candidate_allowed = passed

    obs = {
        "auditory_observation_passed": passed,
        "auditory_observation_status": status,
        "modality": "auditory",
        "source": source,
        "origin": origin,
        "fact_status": fact_status,
        "raw_audio_data_present": bool(meta.get("raw_audio_data_present")),
        **AUDITORY_FALSE_INVARIANT_FLAGS,
        **AUDITORY_TRUE_INVARIANT_FLAGS,
        "observed_features": observed,
        "inferred_features": inferred,
        "confidence": meta.get("confidence", {"overall": "unknown"}),
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": privacy_flags,
        "candidate_only_fields": candidate_only_fields,
        "boundary_flags": ["reality_boundary_preserved", "internal_external_boundary_preserved"],
        "event_candidate_allowed": event_candidate_allowed,
        "thought_candidate_allowed": thought_candidate_allowed,
        "memory_candidate_allowed": memory_candidate_allowed,
        "quarantine_required": quarantine_required,
        "blocked_reasons": blocked_reasons,
        "warnings": warnings,
    }

    return obs


def validate_auditory_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate auditory observation schema strictly."""

    obs = _copy_mapping(observation)
    blocked: list[str] = []

    required_fields = tuple(build_auditory_observation("voice_message_candidate").keys())
    missing = [field for field in required_fields if field not in obs]
    if missing:
        blocked.append("missing_required_auditory_observation_fields")

    if obs.get("modality") != "auditory":
        blocked.append("auditory_observation_modality_must_be_auditory")

    if obs.get("source") not in SUPPORTED_AUDITORY_SOURCE_TYPES:
        blocked.append("unknown_auditory_source_fails_closed")

    if obs.get("raw_audio_data_present") is True and not obs.get("blocked_reasons"):
        blocked.append("raw_audio_data_must_not_enter_auditory_observation_schema")

    for key, expected in AUDITORY_FALSE_INVARIANT_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"invariant_violation_{key}_must_be_false")

    for key, expected in AUDITORY_TRUE_INVARIANT_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"invariant_violation_{key}_must_be_true")

    if obs.get("source") == "dmn_inner_voice_candidate" and obs.get("fact_status") == "observed_external":
        blocked.append("dmn_inner_voice_candidate_cannot_use_observed_external_fact_status")

    passed = not bool(blocked) and obs.get("auditory_observation_passed", False)

    return {
        "validation_passed": passed,
        "validation_status": "auditory_observation_valid" if passed else "auditory_observation_invalid",
        "missing_fields": missing,
        "blocked_reasons": blocked,
        "read_only_invariants_preserved": passed,
    }


def build_auditory_observation_to_sensory_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only compatibility plan for the sensory observation contract."""

    validation = validate_auditory_observation(observation)

    sensory_source = str(observation.get("source", "unknown_auditory_source"))

    sensory_observation = build_sensory_observation(
        "auditory",
        sensory_source,
        observed_features=observation.get("observed_features") if isinstance(observation.get("observed_features"), Mapping) else {},
        inferred_features=observation.get("inferred_features") if isinstance(observation.get("inferred_features"), Mapping) else {},
        metadata={
            "confidence": observation.get("confidence") if isinstance(observation.get("confidence"), Mapping) else {},
        }
    )

    if not validation["validation_passed"]:
        sensory_observation["observation_passed"] = False
        sensory_observation["observation_status"] = "sensory_observation_blocked_by_auditory_validation"
        if isinstance(sensory_observation.get("blocked_reasons"), list):
            sensory_observation["blocked_reasons"].extend(validation["blocked_reasons"])

    return sensory_observation


def build_auditory_observation_to_origin_fact_status_record(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only compatibility record for the origin/fact-status policy."""

    validation = validate_auditory_observation(observation)

    record = build_origin_fact_status_record(
        origin=str(observation.get("origin", "external_audio")),
        fact_status=str(observation.get("fact_status", "observed_external")),
        source=str(observation.get("source", "unknown_auditory_source")),
    )

    if not validation["validation_passed"]:
        record["origin_fact_status_passed"] = False
        record["blocked_reasons"].extend(validation["blocked_reasons"])

    if observation.get("source") == "dmn_inner_voice_candidate" and record.get("fact_status") == "observed_external":
        record["origin_fact_status_passed"] = False
        record["blocked_reasons"].append("dmn_inner_voice_candidate_must_not_use_observed_external")

    return record


def build_auditory_observation_to_event_candidate_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan auditory-to-event conversion as candidates only, never facts."""

    validation = validate_auditory_observation(observation)
    agp_plan = build_sensory_observation_to_agp_input_plan(
        build_sensory_observation("auditory", str(observation.get("source", "unknown_auditory_source")))
    )

    return {
        "plan_version": ROUND1061_1080_VERSION,
        "plan_status": "auditory_event_candidate_plan_read_only" if validation["validation_passed"] else "auditory_event_candidate_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "event_candidate_allowed": validation["validation_passed"],
        "event_fact_asserted": False,
        "emotion_fact_asserted": False,
        "intent_fact_asserted": False,
        "relationship_fact_asserted": False,
        "agp_plan": agp_plan,
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_auditory_observation_memory_candidate_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan auditory memory/self-model gating without permitting writes."""

    validation = validate_auditory_observation(observation)
    sensory_plan = build_sensory_memory_candidate_plan(
        build_sensory_observation("auditory", str(observation.get("source", "unknown_auditory_source")))
    )

    candidate_requested = bool(
        observation.get("memory_candidate_allowed") or
        observation.get("thought_candidate_allowed") or
        observation.get("quarantine_required")
    )

    return {
        "plan_version": ROUND1061_1080_VERSION,
        "plan_status": "auditory_memory_candidate_plan_read_only" if validation["validation_passed"] else "auditory_memory_candidate_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "memory_candidate_allowed": validation["validation_passed"] and bool(observation.get("memory_candidate_allowed")),
        "memory_write_performed": False,
        "self_model_update_allowed": False,
        "quarantine_required": candidate_requested or bool(observation.get("quarantine_required")),
        "appraisal_required": True,
        "sensory_plan": sensory_plan,
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_auditory_observation_schema_summary() -> dict[str, Any]:
    """Summary of the auditory observation schema."""

    schema_keys = tuple(build_auditory_observation("voice_message_candidate").keys())
    sensory_summary = build_sensory_observation_contract_summary()

    return {
        "schema_version": ROUND1061_1080_VERSION,
        "feature_track": FEATURE_TRACK,
        "schema_status": "read_only_auditory_schema_design_test_surface_only",
        "derived_from_sensory_contract_version": sensory_summary["contract_version"],
        "supported_auditory_sources": SUPPORTED_AUDITORY_SOURCE_TYPES,
        "observed_feature_groups": SUPPORTED_OBSERVED_FEATURE_GROUPS,
        "inferred_candidate_groups": SUPPORTED_INFERRED_FEATURE_GROUPS,
        "auditory_observation_schema_required_fields": schema_keys,
        "read_only_false_invariants": list(AUDITORY_FALSE_INVARIANT_FLAGS.keys()),
        "read_only_true_invariants": list(AUDITORY_TRUE_INVARIANT_FLAGS.keys()),
        "anti_global_synchrony_blocked": True,
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def auditory_observation_schema_summary() -> dict[str, Any]:
    """Backward-compatible alias for operator/report callers."""
    return build_auditory_observation_schema_summary()
