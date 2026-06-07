"""Round1001-1020 read-only visual observation schema for EVE v3.

This module specializes the Round981-1000 sensory observation contract for
already-described visual candidates.  It is schema/design/test-surface only: it
never connects a camera, processes image bytes, performs OCR, loads vision
models, performs face recognition, persists raw visual data, writes memory,
updates self-model state, mutates affect/hormones, reads vectors, or bypasses
AGP/fallback gates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from adapters.sensory_observation_contract import (
    AFFECT_AXIS_GROUPS,
    VISUAL_SOURCE_TYPES as SENSORY_VISUAL_SOURCE_TYPES,
    build_sensory_memory_candidate_plan,
    build_sensory_observation,
    build_sensory_observation_contract_summary,
    build_sensory_observation_to_agp_input_plan,
    validate_sensory_observation,
)

ROUND1001_1020_VERSION = "v3_round1001_1020_visual_observation_schema"
FEATURE_TRACK = "read_only_visual_observation_schema_from_sensory_contract"

SUPPORTED_VISUAL_SOURCE_TYPES: tuple[str, ...] = SENSORY_VISUAL_SOURCE_TYPES

SUPPORTED_OBSERVED_FEATURE_GROUPS: tuple[str, ...] = (
    "scene_features",
    "object_features",
    "person_presence_features",
    "screen_ui_features",
    "lighting_features",
    "motion_features",
    "safety_relevance_features",
)

SUPPORTED_INFERRED_FEATURE_GROUPS: tuple[str, ...] = (
    "person_identity_candidate",
    "expression_candidate",
    "posture_candidate",
    "gaze_candidate",
    "activity_candidate",
    "place_candidate",
    "screen_state_candidate",
    "visual_salience_candidate",
)

CANDIDATE_ONLY_FIELDS: tuple[str, ...] = SUPPORTED_INFERRED_FEATURE_GROUPS

RAW_VISUAL_DATA_KEYS: tuple[str, ...] = (
    "raw_data",
    "raw_payload",
    "raw_image",
    "raw_image_bytes",
    "image_bytes",
    "raw_video",
    "raw_video_bytes",
    "video_bytes",
    "raw_screen_recording",
    "screen_recording_bytes",
)

ASSERTION_FORBIDDEN_KEYS: tuple[str, ...] = (
    "person_identity_asserted",
    "user_emotion_asserted",
    "user_intent_asserted",
    "relationship_state_asserted",
    "memory_fact_asserted",
    "identity_fact",
    "user_emotion",
    "emotion_fact",
    "intent_fact",
    "user_intent",
    "relationship_state",
    "memory_fact",
)

VISUAL_FALSE_INVARIANT_FLAGS: dict[str, bool] = {
    "raw_data_persisted": False,
    "raw_image_persisted": False,
    "raw_video_persisted": False,
    "raw_screen_recording_persisted": False,
    "camera_activated": False,
    "ocr_performed": False,
    "vision_model_loaded": False,
    "face_recognition_performed": False,
    "person_identity_asserted": False,
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

VISUAL_TRUE_GATE_FLAGS: dict[str, bool] = {
    "appraisal_required": True,
    "agp_input_required": True,
    "memory_gate_required": True,
    "sensory_contract_required": True,
}

NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "Add an operator-authorized visual adapter preflight that accepts path metadata only and still performs no camera activation, OCR, vision model load, face recognition, persistence, memory write, or runtime mutation."
)

LOW_CONFIDENCE_THRESHOLD = 0.50


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _copy_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return deepcopy(list(value))
    return [deepcopy(value)]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _contains_any_key(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in mapping for key in keys)


def _confidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    provided = metadata.get("confidence")
    if isinstance(provided, Mapping):
        return deepcopy(dict(provided))
    return {
        "policy": "caller_supplied_visual_confidence_metadata_only_no_vision_model_inference",
        "overall": metadata.get("confidence_overall", "unknown"),
    }


def _confidence_values(confidence: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for value in confidence.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _privacy_flags_for(source: str, observed: Mapping[str, Any], inferred: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    flags = _copy_list(metadata.get("privacy_flags"))
    combined = {**observed, **inferred}
    if source in {"user_uploaded_image", "camera_frame_candidate"} or "person_presence_features" in observed or "person_identity_candidate" in inferred:
        _append_unique(flags, "visual_person_or_face_privacy_review_required")
        _append_unique(flags, "person_identity_candidate_only")
    if source in {"screenshot_candidate", "screen_ui_candidate"} or "screen_ui_features" in observed or "screen_state_candidate" in inferred:
        _append_unique(flags, "screen_or_ui_path_metadata_privacy_review_required")
        _append_unique(flags, "screen_content_ocr_forbidden_in_round1001_1020")
    if metadata.get("private_scene_candidate") is True or combined.get("private_scene_candidate") is True:
        _append_unique(flags, "private_scene_visual_privacy_review_required")
    return flags


def _uncertainty_flags_for(confidence: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    flags = _copy_list(metadata.get("uncertainty_flags"))
    if any(value < LOW_CONFIDENCE_THRESHOLD for value in _confidence_values(confidence)):
        _append_unique(flags, "low_confidence_visual_inference_candidate")
    return flags


def _base_visual_observation(source: str, observed: dict[str, Any], inferred: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    confidence = _confidence(metadata)
    raw_visual_data_present = bool(metadata.get("raw_visual_data_present") or metadata.get("raw_data_present")) or _contains_any_key(metadata, RAW_VISUAL_DATA_KEYS)
    observation = {
        "schema_version": ROUND1001_1020_VERSION,
        "feature_track": FEATURE_TRACK,
        "visual_observation_passed": True,
        "visual_observation_status": "visual_observation_read_only_candidate",
        "modality": "visual",
        "source": source,
        "raw_visual_data_present": raw_visual_data_present,
        **VISUAL_FALSE_INVARIANT_FLAGS,
        "observed_features": observed,
        "inferred_features": inferred,
        "confidence": confidence,
        "uncertainty_flags": _uncertainty_flags_for(confidence, metadata),
        "privacy_flags": _privacy_flags_for(source, observed, inferred, metadata),
        "candidate_only_fields": list(CANDIDATE_ONLY_FIELDS),
        **VISUAL_TRUE_GATE_FLAGS,
        "event_candidate_allowed": bool(metadata.get("event_candidate") or metadata.get("thought_candidate")),
        "thought_candidate_allowed": bool(metadata.get("thought_candidate")),
        "memory_candidate_allowed": bool(metadata.get("memory_update_candidate") or metadata.get("self_model_update_candidate")),
        "quarantine_required": bool(metadata.get("malicious_or_hostile_content_candidate")) or bool(metadata.get("memory_update_candidate")) or bool(metadata.get("self_model_update_candidate")),
        "blocked_reasons": [],
        "warnings": [],
        "screen_path_metadata_only_required": source in {"screenshot_candidate", "screen_ui_candidate"},
        "raw_visual_direct_to_agp_allowed": False,
        "emotion_fact_from_visual_allowed": False,
        "intent_fact_from_visual_allowed": False,
        "relationship_fact_from_visual_allowed": False,
        "memory_fact_from_visual_allowed": False,
        "global_synchrony_blocked": True,
        "all_axis_affect_hormone_activation_allowed": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
    }
    if observation["privacy_flags"]:
        observation["quarantine_required"] = True
    return observation


def _apply_visual_validation_rules(observation: dict[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    source = observation["source"]
    observed = observation["observed_features"]
    inferred = observation["inferred_features"]
    blocked = observation["blocked_reasons"]
    warnings = observation["warnings"]

    if source not in SUPPORTED_VISUAL_SOURCE_TYPES:
        blocked.append("unknown_visual_source_fails_closed")

    if observation["raw_visual_data_present"]:
        blocked.append("raw_visual_data_must_not_enter_visual_observation_schema")
    if _contains_any_key(observed, RAW_VISUAL_DATA_KEYS) or _contains_any_key(inferred, RAW_VISUAL_DATA_KEYS):
        blocked.append("raw_visual_payload_fields_are_forbidden")

    if _contains_any_key(observed, ASSERTION_FORBIDDEN_KEYS) or _contains_any_key(inferred, ASSERTION_FORBIDDEN_KEYS):
        blocked.append("visual_observation_must_not_assert_identity_emotion_intent_relationship_or_memory_fact")

    if set(inferred).intersection(CANDIDATE_ONLY_FIELDS):
        warnings.append("visual_inferred_features_are_candidate_only_not_facts")

    if "person_identity_candidate" in inferred:
        warnings.append("person_identity_candidate_requires_appraisal_and_cannot_be_asserted")
    for key in ("expression_candidate", "posture_candidate", "gaze_candidate", "activity_candidate"):
        if key in inferred:
            warnings.append(f"{key}_requires_appraisal_and_cannot_assert_user_emotion_or_intent")

    if source in {"screenshot_candidate", "screen_ui_candidate"}:
        warnings.append("screenshot_or_screen_ui_source_is_path_metadata_only_without_operator_authorized_content_processing")

    if metadata.get("malicious_or_hostile_content_candidate") is True:
        observation["quarantine_required"] = True
        warnings.append("malicious_or_hostile_visual_content_preserves_appraisal_quarantine_gates")

    if metadata.get("memory_update_candidate") or metadata.get("self_model_update_candidate"):
        observation["memory_candidate_allowed"] = True
        observation["quarantine_required"] = True
        warnings.append("visual_memory_or_self_model_candidate_requires_appraisal_quarantine_before_any_update")

    requested_axes = set(_copy_list(metadata.get("requested_affect_axis_groups")))
    if requested_axes >= set(AFFECT_AXIS_GROUPS) or metadata.get("global_synchrony_requested") is True:
        blocked.append("one_visual_observation_cannot_produce_all_axis_affect_hormone_activation")
    observation["global_synchrony_blocked"] = True
    observation["all_axis_affect_hormone_activation_allowed"] = False

    for key, expected in VISUAL_FALSE_INVARIANT_FLAGS.items():
        if observation.get(key) is not expected:
            blocked.append(f"{key}_must_remain_false")
    for key, expected in VISUAL_TRUE_GATE_FLAGS.items():
        if observation.get(key) is not expected:
            blocked.append(f"{key}_must_remain_true")

    if blocked:
        observation["visual_observation_passed"] = False
        observation["visual_observation_status"] = "visual_observation_failed_closed"
    return observation


def build_visual_observation(
    source: str,
    observed_features: Mapping[str, Any] | None = None,
    inferred_features: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only visual observation candidate from structured data only."""

    observed = _copy_mapping(observed_features)
    inferred = _copy_mapping(inferred_features)
    meta = _copy_mapping(metadata)
    observation = _base_visual_observation(source, observed, inferred, meta)
    return _apply_visual_validation_rules(observation, meta)


def validate_visual_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a visual observation preserves all read-only invariants."""

    obs = _copy_mapping(observation)
    blocked: list[str] = []
    required_fields = tuple(build_visual_observation("user_uploaded_image").keys())
    missing = [field for field in required_fields if field not in obs]
    if missing:
        blocked.append("missing_required_visual_observation_fields")
    if obs.get("modality") != "visual":
        blocked.append("visual_observation_modality_must_be_visual")
    if obs.get("source") not in SUPPORTED_VISUAL_SOURCE_TYPES:
        blocked.append("unknown_visual_source_fails_closed")
    if obs.get("raw_visual_data_present") is True:
        blocked.append("raw_visual_data_must_not_enter_visual_observation_schema")
    for key, expected in VISUAL_FALSE_INVARIANT_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"{key}_must_remain_false")
    for key, expected in VISUAL_TRUE_GATE_FLAGS.items():
        if obs.get(key) is not expected:
            blocked.append(f"{key}_must_remain_true")
    if obs.get("agp_bypass_allowed") is True or obs.get("fallback_bypass_allowed") is True:
        blocked.append("agp_and_fallback_bypass_must_remain_forbidden")
    if obs.get("memory_write_performed") is True or obs.get("self_model_update_allowed") is True:
        blocked.append("memory_and_self_model_mutation_must_remain_forbidden")
    return {
        "validation_passed": not blocked,
        "validation_status": "visual_observation_schema_valid" if not blocked else "visual_observation_schema_failed_closed",
        "missing_fields": missing,
        "blocked_reasons": blocked,
        "read_only_invariants_preserved": not blocked,
    }


def build_visual_observation_to_sensory_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only compatibility plan for the sensory observation contract."""

    validation = validate_visual_observation(observation)
    sensory_observation = build_sensory_observation(
        "visual",
        str(observation.get("source", "unknown_visual_source")),
        observed_features=observation.get("observed_features") if isinstance(observation.get("observed_features"), Mapping) else {},
        inferred_features=observation.get("inferred_features") if isinstance(observation.get("inferred_features"), Mapping) else {},
        metadata={
            "confidence": observation.get("confidence") if isinstance(observation.get("confidence"), Mapping) else {},
            "privacy_flags": observation.get("privacy_flags", []),
            "uncertainty_flags": observation.get("uncertainty_flags", []),
            "thought_candidate": observation.get("thought_candidate_allowed") is True,
            "memory_update_candidate": observation.get("memory_candidate_allowed") is True,
            "malicious_or_hostile_content_candidate": False,
        },
    )
    sensory_validation = validate_sensory_observation(sensory_observation)
    return {
        "plan_version": ROUND1001_1020_VERSION,
        "plan_status": "visual_to_sensory_contract_plan_read_only" if validation["validation_passed"] and sensory_validation["validation_passed"] else "visual_to_sensory_contract_plan_blocked",
        "plan_passed": validation["validation_passed"] and sensory_validation["validation_passed"],
        "sensory_contract_required": True,
        "sensory_observation_candidate": sensory_observation,
        "sensory_validation": sensory_validation,
        "appraisal_required": True,
        "agp_input_required": True,
        "memory_gate_required": True,
        "raw_visual_data_forwarded": False,
        "ocr_performed": False,
        "vision_model_loaded": False,
        "face_recognition_performed": False,
        "runtime_mutation_performed": False,
        "persistence_write_performed": False,
        "blocked_reasons": validation["blocked_reasons"] + sensory_validation["blocked_reasons"],
    }


def build_visual_observation_to_event_candidate_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan visual-to-event conversion as candidates only, never facts."""

    validation = validate_visual_observation(observation)
    agp_plan = build_sensory_observation_to_agp_input_plan(build_sensory_observation("visual", str(observation.get("source", "unknown_visual_source"))))
    return {
        "plan_version": ROUND1001_1020_VERSION,
        "plan_status": "visual_event_candidate_plan_read_only" if validation["validation_passed"] else "visual_event_candidate_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "event_candidate_allowed": validation["validation_passed"],
        "event_fact_asserted": False,
        "thought_candidate_allowed": bool(observation.get("thought_candidate_allowed")) and validation["validation_passed"],
        "appraisal_required": True,
        "agp_input_required": True,
        "agp_bypass_allowed": False,
        "fallback_bypass_allowed": False,
        "raw_visual_direct_to_event_allowed": False,
        "user_emotion_fact_allowed": False,
        "user_intent_fact_allowed": False,
        "relationship_state_fact_allowed": False,
        "memory_fact_allowed": False,
        "candidate_only_fields": list(CANDIDATE_ONLY_FIELDS),
        "sensory_agp_plan_gate_summary": {
            "agp_input_required": agp_plan["agp_input_required"],
            "category_anchor_must_come_from_eve_internal_activation": agp_plan["category_anchor_must_come_from_eve_internal_activation"],
        },
        "runtime_mutation_performed": False,
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_visual_observation_memory_candidate_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Plan visual memory/self-model gating without permitting writes."""

    validation = validate_visual_observation(observation)
    sensory_plan = build_sensory_memory_candidate_plan(build_sensory_observation("visual", str(observation.get("source", "unknown_visual_source"))))
    candidate_requested = bool(observation.get("memory_candidate_allowed") or observation.get("thought_candidate_allowed") or observation.get("quarantine_required"))
    return {
        "plan_version": ROUND1001_1020_VERSION,
        "plan_status": "visual_memory_candidate_plan_read_only" if validation["validation_passed"] else "visual_memory_candidate_plan_blocked",
        "plan_passed": validation["validation_passed"],
        "memory_candidate_allowed": validation["validation_passed"] and bool(observation.get("memory_candidate_allowed")),
        "memory_gate_required": True,
        "quarantine_required": True if candidate_requested else False,
        "appraisal_required_before_memory": True,
        "appraisal_required_before_self_model": True,
        "raw_visual_data_memory_write_allowed": False,
        "long_term_memory_write_allowed": False,
        "self_model_update_allowed": False,
        "memory_write_performed": False,
        "persistence_write_performed": False,
        "runtime_mutation_performed": False,
        "sensory_memory_plan_gate_summary": {
            "memory_gate_required": sensory_plan["memory_gate_required"],
            "appraisal_required_before_memory": sensory_plan["appraisal_required_before_memory"],
            "long_term_memory_write_allowed": sensory_plan["long_term_memory_write_allowed"],
        },
        "blocked_reasons": validation["blocked_reasons"],
    }


def build_visual_observation_schema_summary() -> dict[str, Any]:
    schema_keys = tuple(build_visual_observation("user_uploaded_image").keys())
    sensory_summary = build_sensory_observation_contract_summary()
    return {
        "schema_version": ROUND1001_1020_VERSION,
        "feature_track": FEATURE_TRACK,
        "schema_status": "read_only_visual_schema_design_test_surface_only",
        "derived_from_sensory_contract_version": sensory_summary["contract_version"],
        "supported_visual_sources": SUPPORTED_VISUAL_SOURCE_TYPES,
        "observed_feature_groups": SUPPORTED_OBSERVED_FEATURE_GROUPS,
        "inferred_candidate_groups": SUPPORTED_INFERRED_FEATURE_GROUPS,
        "visual_observation_schema_required_fields": schema_keys,
        "candidate_only_inference_policy": "identity, expression, posture, gaze, activity, place, screen state, and visual salience are candidates only and never facts",
        "non_assertion_policy": {
            "person_identity_asserted": False,
            "user_emotion_asserted": False,
            "user_intent_asserted": False,
            "relationship_state_asserted": False,
            "memory_fact_asserted": False,
        },
        "raw_visual_non_persistence_policy": {
            "raw_data_persisted": False,
            "raw_image_persisted": False,
            "raw_video_persisted": False,
            "raw_screen_recording_persisted": False,
        },
        "no_device_or_model_policy": {
            "camera_activated": False,
            "ocr_performed": False,
            "vision_model_loaded": False,
            "face_recognition_performed": False,
        },
        "gate_compatibility_policy": {
            "appraisal_required": True,
            "agp_input_required": True,
            "memory_gate_required": True,
            "sensory_contract_required": True,
            "agp_bypass_allowed": False,
            "fallback_bypass_allowed": False,
            "affect_transition_allowed": False,
            "hormone_transition_allowed": False,
        },
        "confidence_uncertainty_policy": f"caller-supplied confidence below {LOW_CONFIDENCE_THRESHOLD:.2f} adds uncertainty flags and never becomes fact evidence",
        "privacy_policy": "person/face, screenshot, screen UI, and private-scene candidates are privacy-flagged and quarantined for appraisal",
        "screen_source_policy": "screenshot_candidate and screen_ui_candidate are path-metadata-only in round1001-1020 unless a later explicit operator-authorized round changes the schema",
        "affect_hormone_ladder_compatibility": "visual observation may not directly update affect/hormones and cannot request all-axis/global synchrony activation",
        "persistence_runtime_vector_artifact_policy": {
            "vector_read_performed": False,
            "vector_load_performed": False,
            "persistence_write_performed": False,
            "runtime_mutation_performed": False,
            "artifact_created_or_staged": False,
        },
        "next_implementation_recommendations": (NEXT_IMPLEMENTATION_RECOMMENDATION,),
    }


def visual_observation_schema_summary() -> dict[str, Any]:
    """Backward-compatible alias for operator/report callers."""

    return build_visual_observation_schema_summary()
