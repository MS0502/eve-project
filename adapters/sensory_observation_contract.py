"""Round981-1000 read-only sensory observation contract for EVE v3.

The contract is a pure-data boundary for future multimodal EVE inputs.  It
normalizes already-described observations only; it does not connect devices,
run OCR/STT/model inference, load vectors, persist raw sensory data, write
memory, update self-model state, mutate affect/hormones, or bypass AGP/fallback
verification.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

ROUND981_1000_VERSION = "v3_round981_1000_sensory_observation_contract"
FEATURE_TRACK = "read_only_sensory_observation_contract_for_multimodal_eve"
OBSERVATION_ID_POLICY = "deterministic_caller_supplied_or_contentless_sequence_id_no_raw_data_hashing"

SUPPORTED_MODALITIES: tuple[str, ...] = (
    "text",
    "visual",
    "auditory",
    "internal_state",
    "tool_state",
    "multimodal",
)

TEXT_SOURCE_TYPES: tuple[str, ...] = (
    "user_text_candidate",
    "operator_text_candidate",
    "system_text_candidate",
)

VISUAL_SOURCE_TYPES: tuple[str, ...] = (
    "user_uploaded_image",
    "camera_frame_candidate",
    "screenshot_candidate",
    "screen_ui_candidate",
)

AUDITORY_SOURCE_TYPES: tuple[str, ...] = (
    "voice_message_candidate",
    "microphone_frame_candidate",
    "speech_transcript_candidate",
    "environmental_sound_candidate",
)

INTERNAL_STATE_SOURCE_TYPES: tuple[str, ...] = (
    "battery",
    "temperature",
    "ram_pressure",
    "storage_pressure",
    "network_state",
    "process_health",
    "error_state",
    "latency_state",
)

TOOL_STATE_SOURCE_TYPES: tuple[str, ...] = (
    "file_change",
    "notification",
    "calendar_event",
    "telegram_event",
    "github_event",
    "test_result",
    "search_result",
    "app_state",
)

MULTIMODAL_SOURCE_TYPES: tuple[str, ...] = (
    "multimodal_candidate",
    "paired_text_visual_candidate",
    "paired_text_audio_candidate",
    "paired_visual_audio_candidate",
)

SOURCE_TYPES_BY_MODALITY: dict[str, tuple[str, ...]] = {
    "text": TEXT_SOURCE_TYPES,
    "visual": VISUAL_SOURCE_TYPES,
    "auditory": AUDITORY_SOURCE_TYPES,
    "internal_state": INTERNAL_STATE_SOURCE_TYPES,
    "tool_state": TOOL_STATE_SOURCE_TYPES,
    "multimodal": MULTIMODAL_SOURCE_TYPES,
}

FALSE_INVARIANT_FLAGS: dict[str, bool] = {
    "raw_data_persisted": False,
    "raw_image_persisted": False,
    "raw_audio_persisted": False,
    "raw_video_persisted": False,
    "raw_screen_recording_persisted": False,
    "ocr_performed": False,
    "speech_to_text_performed": False,
    "camera_activated": False,
    "microphone_activated": False,
    "model_loaded": False,
    "vector_read_performed": False,
    "vector_load_performed": False,
    "memory_write_performed": False,
    "long_term_memory_write_allowed": False,
    "self_model_update_allowed": False,
    "affect_transition_allowed": False,
    "hormone_transition_allowed": False,
    "agp_bypass_allowed": False,
    "fallback_bypass_allowed": False,
    "runtime_mutation_performed": False,
    "persistence_write_performed": False,
    "artifact_created_or_staged": False,
}

TRUE_GATE_FLAGS: dict[str, bool] = {
    "appraisal_required": True,
    "agp_input_required": True,
    "memory_gate_required": True,
}

RAW_DATA_KEYS: tuple[str, ...] = (
    "raw_data",
    "raw_payload",
    "raw_image",
    "raw_image_bytes",
    "raw_audio",
    "raw_audio_bytes",
    "raw_video",
    "raw_video_bytes",
    "raw_screen_recording",
    "image_bytes",
    "audio_bytes",
    "video_bytes",
)

EMOTION_FACT_KEYS: tuple[str, ...] = (
    "emotion",
    "user_emotion",
    "person_emotion",
    "speaker_emotion",
    "emotion_fact",
    "user_emotion_fact",
)

VISUAL_INFERENCE_KEYS: tuple[str, ...] = (
    "expression",
    "facial_expression",
    "posture",
    "gaze",
    "person_identity",
    "face_identity",
)

AUDITORY_INFERENCE_KEYS: tuple[str, ...] = (
    "prosody",
    "tone",
    "speaker_identity",
    "voice_identity",
    "speaker_mood",
)

PERSON_PRIVACY_KEYS: tuple[str, ...] = (
    "person",
    "people",
    "face",
    "identity",
    "person_identity",
    "speaker_identity",
    "voice_identity",
    "user_identity",
)

TOOL_STATE_METADATA_ONLY_KEYS: tuple[str, ...] = (
    "path_metadata",
    "event_type",
    "timestamp_policy",
    "status",
    "exit_code",
    "count",
    "summary",
    "source_label",
)

TOOL_STATE_FORBIDDEN_CONTENT_KEYS: tuple[str, ...] = (
    "file_content",
    "raw_content",
    "content",
    "body",
    "message_body",
    "document_text",
    "secret",
    "token",
)

SOCIAL_SELF_IDENTITY_AXES: tuple[str, ...] = (
    "social",
    "social_relationship",
    "self",
    "self_model",
    "identity",
    "identity_integrity",
)

AFFECT_AXIS_GROUPS: tuple[str, ...] = (
    "survival_stability",
    "risk_defense",
    "social_relationship",
    "learning_exploration",
    "self_identity_agency",
    "expression_action",
)

NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "Add a read-only operator-authorized sensory adapter preflight that accepts this contract output and still performs no OCR, STT, device activation, model load, memory write, or runtime mutation."
)


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _copy_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return deepcopy(list(value))
    return [deepcopy(value)]


def _contains_any_key(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in mapping for key in keys)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _privacy_flags_for(modality: str, source: str, observed: Mapping[str, Any], inferred: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    combined = {**observed, **inferred}
    if modality in {"visual", "multimodal"} and (
        source in {"user_uploaded_image", "camera_frame_candidate", "screenshot_candidate"}
        or _contains_any_key(combined, PERSON_PRIVACY_KEYS)
    ):
        flags.extend(["visual_person_privacy_review_required", "face_or_person_identity_candidate_only"])
    if modality in {"auditory", "multimodal"} and (
        source in {"voice_message_candidate", "microphone_frame_candidate", "speech_transcript_candidate"}
        or _contains_any_key(combined, PERSON_PRIVACY_KEYS)
    ):
        flags.extend(["auditory_voice_privacy_review_required", "voice_or_speaker_identity_candidate_only"])
    return list(dict.fromkeys(flags))


def _confidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    provided = metadata.get("confidence")
    if isinstance(provided, Mapping):
        return deepcopy(dict(provided))
    return {
        "policy": "caller_supplied_confidence_metadata_only_no_model_inference",
        "overall": metadata.get("confidence_overall", "unknown"),
    }


def _base_observation(modality: str, source: str, observed: dict[str, Any], inferred: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    raw_data_present = bool(metadata.get("raw_data_present")) or _contains_any_key(metadata, RAW_DATA_KEYS)
    uncertainty_flags = _copy_list(metadata.get("uncertainty_flags"))
    privacy_flags = _copy_list(metadata.get("privacy_flags"))
    for flag in _privacy_flags_for(modality, source, observed, inferred):
        _append_unique(privacy_flags, flag)

    observation = {
        "contract_version": ROUND981_1000_VERSION,
        "feature_track": FEATURE_TRACK,
        "observation_passed": True,
        "observation_status": "sensory_observation_read_only_candidate",
        "modality": modality,
        "source": source,
        "observation_id_policy": OBSERVATION_ID_POLICY,
        "raw_data_present": raw_data_present,
        **FALSE_INVARIANT_FLAGS,
        "observed_features": observed,
        "inferred_features": inferred,
        "confidence": _confidence(metadata),
        "uncertainty_flags": uncertainty_flags,
        "privacy_flags": privacy_flags,
        **TRUE_GATE_FLAGS,
        "thought_candidate_allowed": False,
        "memory_candidate_allowed": False,
        "quarantine_required": bool(privacy_flags) or bool(metadata.get("malicious_or_hostile_content_candidate")),
        "blocked_reasons": [],
        "warnings": [],
        "cross_modal_binding_performed": False,
        "cross_modal_binding_deferred": modality == "multimodal",
        "global_synchrony_blocked": True,
        "all_axis_affect_hormone_activation_allowed": False,
        "hardware_operational_only": modality == "internal_state",
        "hardware_non_panic": modality == "internal_state",
        "tool_state_path_metadata_only": modality == "tool_state",
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
    }
    return observation


def _apply_validation_rules(observation: dict[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    modality = observation["modality"]
    source = observation["source"]
    observed = observation["observed_features"]
    inferred = observation["inferred_features"]
    blocked = observation["blocked_reasons"]
    warnings = observation["warnings"]

    if modality not in SUPPORTED_MODALITIES:
        blocked.append("unknown_modality_fails_closed")
    elif source not in SOURCE_TYPES_BY_MODALITY[modality]:
        blocked.append("unknown_source_fails_closed")

    if observation["raw_data_present"]:
        blocked.append("raw_sensory_data_must_not_enter_observation_contract")

    if _contains_any_key(observed, RAW_DATA_KEYS) or _contains_any_key(inferred, RAW_DATA_KEYS):
        blocked.append("raw_sensory_payload_fields_are_forbidden")

    if modality in {"visual", "auditory", "multimodal"} and _contains_any_key(observed, EMOTION_FACT_KEYS):
        blocked.append("visual_or_auditory_user_emotion_must_not_be_asserted_as_fact")

    if modality in {"visual", "multimodal"} and _contains_any_key(inferred, VISUAL_INFERENCE_KEYS + EMOTION_FACT_KEYS):
        warnings.append("visual_expression_posture_face_identity_are_inference_candidates_only")

    if modality in {"auditory", "multimodal"} and _contains_any_key(inferred, AUDITORY_INFERENCE_KEYS + EMOTION_FACT_KEYS):
        warnings.append("auditory_prosody_voice_speaker_identity_are_inference_candidates_only")

    if modality == "internal_state":
        axes = set(_copy_list(metadata.get("target_axes"))) | set(_copy_list(metadata.get("affect_axis_groups")))
        if any(axis in SOCIAL_SELF_IDENTITY_AXES for axis in axes):
            blocked.append("internal_hardware_observations_must_not_target_social_self_identity_axes")
        observation["hardware_operational_only"] = True
        observation["hardware_non_panic"] = True
        warnings.append("internal_state_observation_is_operational_only_non_panic")

    if modality == "tool_state":
        non_metadata_keys = [key for key in observed if key not in TOOL_STATE_METADATA_ONLY_KEYS]
        if non_metadata_keys or _contains_any_key(observed, TOOL_STATE_FORBIDDEN_CONTENT_KEYS):
            blocked.append("tool_state_observation_must_remain_path_metadata_only")
        observation["tool_state_path_metadata_only"] = not bool(non_metadata_keys)

    if metadata.get("memory_update_candidate") or metadata.get("self_model_update_candidate"):
        observation["memory_candidate_allowed"] = True
        observation["quarantine_required"] = True
        warnings.append("memory_or_self_model_candidate_requires_appraisal_quarantine_before_any_update")

    if metadata.get("thought_candidate") is True:
        observation["thought_candidate_allowed"] = True
        warnings.append("thought_candidate_requires_appraisal_and_agp_input_before_use")

    if metadata.get("malicious_or_hostile_content_candidate") is True:
        observation["quarantine_required"] = True
        warnings.append("malicious_or_hostile_sensory_content_preserves_appraisal_quarantine_gates")

    requested_axes = set(_copy_list(metadata.get("requested_affect_axis_groups")))
    if requested_axes >= set(AFFECT_AXIS_GROUPS) or metadata.get("global_synchrony_requested") is True:
        blocked.append("one_observation_cannot_produce_all_axis_affect_hormone_activation")
    observation["global_synchrony_blocked"] = True
    observation["all_axis_affect_hormone_activation_allowed"] = False

    for key, expected in FALSE_INVARIANT_FLAGS.items():
        if observation.get(key) is not expected:
            blocked.append(f"{key}_must_remain_false")

    if blocked:
        observation["observation_passed"] = False
        observation["observation_status"] = "sensory_observation_failed_closed"
    return observation


def build_sensory_observation(
    modality: str,
    source: str,
    observed_features: Mapping[str, Any] | None = None,
    inferred_features: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only normalized sensory observation candidate.

    The function never inspects raw files, never performs OCR/STT/model
    inference, never loads vectors, and never mutates runtime state.  Unknown
    modality/source combinations and unsafe payload shapes fail closed.
    """

    observed = _copy_mapping(observed_features)
    inferred = _copy_mapping(inferred_features)
    meta = _copy_mapping(metadata)
    observation = _base_observation(modality, source, observed, inferred, meta)
    return _apply_validation_rules(observation, meta)


def validate_sensory_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that an observation preserves the read-only contract."""

    obs = _copy_mapping(observation)
    blocked: list[str] = []
    required_fields = tuple(build_sensory_observation("text", "user_text_candidate").keys())
    missing = [field for field in required_fields if field not in obs]
    if missing:
        blocked.append("missing_required_observation_fields")
    if obs.get("modality") not in SUPPORTED_MODALITIES:
        blocked.append("unknown_modality_fails_closed")
    elif obs.get("source") not in SOURCE_TYPES_BY_MODALITY[obs.get("modality")]:
        blocked.append("unknown_source_fails_closed")
    for key, expected in FALSE_INVARIANT_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"{key}_must_remain_false")
    for key, expected in TRUE_GATE_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"{key}_must_remain_true")
    if obs.get("raw_data_present") is True:
        blocked.append("raw_sensory_data_must_not_enter_observation_contract")
    if obs.get("agp_bypass_allowed") is True or obs.get("fallback_bypass_allowed") is True:
        blocked.append("agp_and_fallback_bypass_must_remain_forbidden")
    if obs.get("memory_write_performed") is True or obs.get("self_model_update_allowed") is True:
        blocked.append("memory_and_self_model_mutation_must_remain_forbidden")
    return {
        "validation_passed": not blocked,
        "validation_status": "sensory_observation_contract_valid" if not blocked else "sensory_observation_contract_failed_closed",
        "missing_fields": missing,
        "blocked_reasons": blocked,
        "read_only_invariants_preserved": not blocked,
    }


def build_sensory_observation_to_agp_input_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan how a normalized observation may later become AGP input data."""

    validation = validate_sensory_observation(observation)
    return {
        "plan_version": ROUND981_1000_VERSION,
        "plan_status": "agp_input_plan_read_only" if validation["validation_passed"] else "agp_input_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "source_observation_status": observation.get("observation_status"),
        "agp_input_required": True,
        "appraisal_required": True,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "raw_text_or_media_direct_to_agp_allowed": False,
        "category_anchor_must_come_from_eve_internal_activation": True,
        "emotion_fact_from_sensory_inference_allowed": False,
        "cross_modal_binding_performed": False,
        "cross_modal_binding_deferred": observation.get("modality") == "multimodal",
        "runtime_mutation_performed": False,
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_sensory_memory_candidate_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan memory/self-model gating for a normalized observation candidate."""

    validation = validate_sensory_observation(observation)
    candidate_requested = bool(observation.get("memory_candidate_allowed") or observation.get("thought_candidate_allowed"))
    return {
        "plan_version": ROUND981_1000_VERSION,
        "plan_status": "memory_candidate_plan_read_only" if validation["validation_passed"] else "memory_candidate_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "memory_gate_required": True,
        "quarantine_required": True if candidate_requested or observation.get("quarantine_required") else False,
        "appraisal_required_before_memory": True,
        "appraisal_required_before_self_model": True,
        "raw_sensory_data_memory_write_allowed": False,
        "long_term_memory_write_allowed": False,
        "self_model_update_allowed": False,
        "memory_write_performed": False,
        "persistence_write_performed": False,
        "runtime_mutation_performed": False,
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_sensory_observation_contract_summary() -> dict[str, Any]:
    schema_keys = tuple(build_sensory_observation("text", "user_text_candidate").keys())
    return {
        "contract_version": ROUND981_1000_VERSION,
        "feature_track": FEATURE_TRACK,
        "contract_status": "read_only_design_test_surface_only",
        "supported_modalities": SUPPORTED_MODALITIES,
        "supported_sources_by_modality": SOURCE_TYPES_BY_MODALITY,
        "observation_schema_required_fields": schema_keys,
        "false_invariant_flags": FALSE_INVARIANT_FLAGS,
        "true_gate_flags": TRUE_GATE_FLAGS,
        "raw_sensory_non_persistence_policy": {
            "raw_image_persisted": False,
            "raw_audio_persisted": False,
            "raw_video_persisted": False,
            "raw_screen_recording_persisted": False,
            "raw_data_may_enter_memory_or_thought_directly": False,
        },
        "no_device_or_model_policy": {
            "camera_activated": False,
            "microphone_activated": False,
            "ocr_performed": False,
            "speech_to_text_performed": False,
            "model_loaded": False,
        },
        "gate_compatibility_policy": {
            "appraisal_required": True,
            "agp_input_required": True,
            "memory_gate_required": True,
            "agp_bypass_allowed": False,
            "fallback_bypass_allowed": False,
            "affect_transition_allowed": False,
            "hormone_transition_allowed": False,
        },
        "confidence_uncertainty_policy": "confidence is caller-supplied metadata only; uncertainty flags are carried for appraisal and never become facts",
        "privacy_policy": "visual/audio people, face, voice, or identity signals become privacy-flagged candidates only",
        "hardware_policy": "internal_state observations are operational-only, non-panic, and cannot target social/self/identity axes",
        "tool_state_policy": "tool_state observations are path-metadata-only unless a later explicit operator-authorized round changes the contract",
        "cross_modal_binding_policy": "planned only; binding is deferred and not executed in round981-1000",
        "anti_global_synchrony_policy": "one sensory observation cannot produce all-axis affect/hormone activation; global synchrony remains blocked",
        "next_implementation_recommendations": (NEXT_IMPLEMENTATION_RECOMMENDATION,),
    }


def sensory_observation_contract_summary() -> dict[str, Any]:
    """Backward-compatible alias for operator/report callers."""

    return build_sensory_observation_contract_summary()
