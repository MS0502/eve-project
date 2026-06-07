from __future__ import annotations

from pathlib import Path

import pytest

from adapters.sensory_observation_contract import AFFECT_AXIS_GROUPS
from adapters.visual_observation_schema import (
    CANDIDATE_ONLY_FIELDS,
    SUPPORTED_INFERRED_FEATURE_GROUPS,
    SUPPORTED_OBSERVED_FEATURE_GROUPS,
    SUPPORTED_VISUAL_SOURCE_TYPES,
    VISUAL_FALSE_INVARIANT_FLAGS,
    build_visual_observation,
    build_visual_observation_memory_candidate_plan,
    build_visual_observation_schema_summary,
    build_visual_observation_to_event_candidate_plan,
    build_visual_observation_to_sensory_observation,
    validate_visual_observation,
)
from scripts.operator_report_round1001_1020_visual_observation_schema import build_operator_report


@pytest.mark.parametrize("source", SUPPORTED_VISUAL_SOURCE_TYPES)
def test_supported_sources_build_read_only_visual_observation(source):
    observation = build_visual_observation(source, observed_features={"scene_features": {"summary": "candidate"}})
    assert observation["visual_observation_passed"] is True
    assert observation["modality"] == "visual"
    assert observation["source"] == source
    assert validate_visual_observation(observation)["validation_passed"] is True
    for key, expected in VISUAL_FALSE_INVARIANT_FLAGS.items():
        assert observation[key] is expected
    assert observation["appraisal_required"] is True
    assert observation["agp_input_required"] is True
    assert observation["memory_gate_required"] is True
    assert observation["sensory_contract_required"] is True


def test_user_uploaded_image_builds_read_only_visual_observation():
    observation = build_visual_observation("user_uploaded_image", observed_features={"object_features": {"count": 1}})
    assert observation["visual_observation_status"] == "visual_observation_read_only_candidate"
    assert observation["raw_visual_data_present"] is False
    assert observation["raw_data_persisted"] is False


def test_camera_frame_candidate_does_not_activate_camera():
    observation = build_visual_observation("camera_frame_candidate")
    assert observation["visual_observation_passed"] is True
    assert observation["camera_activated"] is False
    assert observation["vision_model_loaded"] is False


@pytest.mark.parametrize("source", ["screenshot_candidate", "screen_ui_candidate"])
def test_screenshot_and_screen_ui_candidates_are_without_ocr_and_privacy_flagged(source):
    observation = build_visual_observation(source, observed_features={"screen_ui_features": {"path_metadata": "only"}})
    assert observation["visual_observation_passed"] is True
    assert observation["ocr_performed"] is False
    assert "screen_or_ui_path_metadata_privacy_review_required" in observation["privacy_flags"]
    assert "screenshot_or_screen_ui_source_is_path_metadata_only_without_operator_authorized_content_processing" in observation["warnings"]


def test_unknown_visual_source_fails_closed():
    observation = build_visual_observation("unknown_visual_source")
    assert observation["visual_observation_passed"] is False
    assert "unknown_visual_source_fails_closed" in observation["blocked_reasons"]
    assert validate_visual_observation(observation)["validation_passed"] is False


def test_visual_observation_converts_to_sensory_observation_contract_plan():
    observation = build_visual_observation("user_uploaded_image", observed_features={"person_presence_features": {"present": True}})
    plan = build_visual_observation_to_sensory_observation(observation)
    assert plan["plan_passed"] is True
    assert plan["sensory_contract_required"] is True
    assert plan["sensory_observation_candidate"]["modality"] == "visual"
    assert plan["sensory_validation"]["validation_passed"] is True
    assert plan["raw_visual_data_forwarded"] is False


def test_visual_to_event_plan_produces_candidates_only():
    observation = build_visual_observation("user_uploaded_image", metadata={"event_candidate": True})
    plan = build_visual_observation_to_event_candidate_plan(observation)
    assert plan["plan_passed"] is True
    assert plan["event_candidate_allowed"] is True
    assert plan["event_fact_asserted"] is False
    assert plan["user_emotion_fact_allowed"] is False
    assert plan["user_intent_fact_allowed"] is False
    assert plan["relationship_state_fact_allowed"] is False
    assert plan["memory_fact_allowed"] is False


def test_visual_to_memory_plan_preserves_memory_gate_and_quarantine():
    observation = build_visual_observation("user_uploaded_image", metadata={"memory_update_candidate": True, "self_model_update_candidate": True})
    plan = build_visual_observation_memory_candidate_plan(observation)
    assert plan["memory_gate_required"] is True
    assert plan["quarantine_required"] is True
    assert plan["appraisal_required_before_memory"] is True
    assert plan["appraisal_required_before_self_model"] is True
    assert plan["long_term_memory_write_allowed"] is False
    assert plan["self_model_update_allowed"] is False


@pytest.mark.parametrize("candidate_key", ["person_identity_candidate", "expression_candidate", "posture_candidate", "gaze_candidate", "activity_candidate"])
def test_visual_inference_groups_are_candidate_only(candidate_key):
    observation = build_visual_observation("user_uploaded_image", inferred_features={candidate_key: "candidate_only"})
    assert observation["visual_observation_passed"] is True
    assert candidate_key in observation["candidate_only_fields"]
    assert "visual_inferred_features_are_candidate_only_not_facts" in observation["warnings"]
    assert observation["person_identity_asserted"] is False
    assert observation["user_emotion_asserted"] is False
    assert observation["user_intent_asserted"] is False


@pytest.mark.parametrize(
    ("features", "blocked_reason"),
    [
        ({"user_emotion": "happy"}, "visual_observation_must_not_assert_identity_emotion_intent_relationship_or_memory_fact"),
        ({"user_intent": "help"}, "visual_observation_must_not_assert_identity_emotion_intent_relationship_or_memory_fact"),
        ({"relationship_state": "friend"}, "visual_observation_must_not_assert_identity_emotion_intent_relationship_or_memory_fact"),
        ({"memory_fact": "met"}, "visual_observation_must_not_assert_identity_emotion_intent_relationship_or_memory_fact"),
    ],
)
def test_visual_observation_cannot_assert_emotion_intent_relationship_or_memory_fact(features, blocked_reason):
    observation = build_visual_observation("user_uploaded_image", observed_features=features)
    assert observation["visual_observation_passed"] is False
    assert blocked_reason in observation["blocked_reasons"]


def test_low_confidence_creates_uncertainty_flags():
    observation = build_visual_observation("user_uploaded_image", inferred_features={"expression_candidate": "candidate"}, metadata={"confidence": {"expression_candidate": 0.24}})
    assert "low_confidence_visual_inference_candidate" in observation["uncertainty_flags"]


def test_face_person_observations_set_privacy_flags():
    observation = build_visual_observation("user_uploaded_image", observed_features={"person_presence_features": {"present": True}}, inferred_features={"person_identity_candidate": "candidate"})
    assert "visual_person_or_face_privacy_review_required" in observation["privacy_flags"]
    assert "person_identity_candidate_only" in observation["privacy_flags"]
    assert observation["quarantine_required"] is True


@pytest.mark.parametrize(
    "flag",
    [
        "raw_data_persisted",
        "raw_image_persisted",
        "raw_video_persisted",
        "raw_screen_recording_persisted",
        "camera_activated",
        "ocr_performed",
        "vision_model_loaded",
        "face_recognition_performed",
        "memory_write_performed",
        "long_term_memory_write_allowed",
        "self_model_update_allowed",
        "affect_transition_allowed",
        "hormone_transition_allowed",
        "agp_bypass_allowed",
        "fallback_bypass_allowed",
        "vector_read_performed",
        "vector_load_performed",
        "persistence_write_performed",
        "runtime_mutation_performed",
        "artifact_created_or_staged",
        "runtime_mapping_enabled_default",
        "enforcement_enabled_default",
        "vector_contents_read",
        "vectors_loaded",
    ],
)
def test_visual_observation_forbidden_flags_always_false(flag):
    observation = build_visual_observation("camera_frame_candidate", observed_features={"person_presence_features": {"present": True}})
    assert observation[flag] is False


@pytest.mark.parametrize(
    "metadata_or_features",
    [
        {"metadata": {"raw_visual_data_present": True}},
        {"metadata": {"raw_image_bytes": b"blocked"}},
        {"observed_features": {"raw_video": "blocked"}},
        {"inferred_features": {"raw_screen_recording": "blocked"}},
    ],
)
def test_raw_visual_payloads_fail_closed_and_are_never_persisted(metadata_or_features):
    observation = build_visual_observation("user_uploaded_image", **metadata_or_features)
    assert observation["visual_observation_passed"] is False
    assert any(reason in observation["blocked_reasons"] for reason in ("raw_visual_data_must_not_enter_visual_observation_schema", "raw_visual_payload_fields_are_forbidden"))
    assert observation["raw_image_persisted"] is False
    assert observation["raw_video_persisted"] is False
    assert observation["raw_screen_recording_persisted"] is False


def test_malicious_visual_content_preserves_appraisal_quarantine_gate():
    observation = build_visual_observation("user_uploaded_image", metadata={"malicious_or_hostile_content_candidate": True})
    assert observation["visual_observation_passed"] is True
    assert observation["quarantine_required"] is True
    assert observation["appraisal_required"] is True
    assert observation["memory_gate_required"] is True
    assert "malicious_or_hostile_visual_content_preserves_appraisal_quarantine_gates" in observation["warnings"]


def test_memory_self_candidates_preserve_appraisal_before_memory_self_model():
    observation = build_visual_observation("user_uploaded_image", metadata={"memory_update_candidate": True, "self_model_update_candidate": True})
    plan = build_visual_observation_memory_candidate_plan(observation)
    assert observation["memory_candidate_allowed"] is True
    assert observation["memory_write_performed"] is False
    assert plan["appraisal_required_before_memory"] is True
    assert plan["appraisal_required_before_self_model"] is True
    assert plan["self_model_update_allowed"] is False


def test_one_visual_observation_cannot_produce_all_axis_affect_hormone_activation_and_global_synchrony_blocked():
    observation = build_visual_observation(
        "user_uploaded_image",
        metadata={"requested_affect_axis_groups": AFFECT_AXIS_GROUPS, "global_synchrony_requested": True},
    )
    assert observation["visual_observation_passed"] is False
    assert observation["global_synchrony_blocked"] is True
    assert observation["all_axis_affect_hormone_activation_allowed"] is False
    assert "one_visual_observation_cannot_produce_all_axis_affect_hormone_activation" in observation["blocked_reasons"]


def test_summary_lists_schema_sources_feature_groups_and_one_recommendation():
    summary = build_visual_observation_schema_summary()
    assert summary["feature_track"] == "read_only_visual_observation_schema_from_sensory_contract"
    assert tuple(summary["supported_visual_sources"]) == SUPPORTED_VISUAL_SOURCE_TYPES
    assert tuple(summary["observed_feature_groups"]) == SUPPORTED_OBSERVED_FEATURE_GROUPS
    assert tuple(summary["inferred_candidate_groups"]) == SUPPORTED_INFERRED_FEATURE_GROUPS
    assert tuple(summary["inferred_candidate_groups"]) == CANDIDATE_ONLY_FIELDS
    assert len(summary["next_implementation_recommendations"]) == 1


def test_operator_report_proves_no_live_mutation_persistence_vectors_artifacts_and_affect_ladder_compatibility():
    report = build_operator_report()
    assert report["success"] is True
    assert report["proofs"]["no_memory_write_proof"]["passed"] is True
    assert report["proofs"]["no_self_model_update_proof"]["passed"] is True
    assert report["proofs"]["no_affect_hormone_transition_proof"]["passed"] is True
    assert report["proofs"]["no_persistence_runtime_vector_artifact_proof"]["passed"] is True
    assert report["proofs"]["anti_global_synchrony_proof"]["passed"] is True
    assert report["proofs"]["compatibility_proof_with_affect_ladder_modules"]["passed"] is True
    assert report["checks"]["exactly_one_next_implementation_recommendation"] is True


def test_no_persistence_enablement_runtime_mapping_enforcement_vector_content_read_or_artifact_creation():
    observation = build_visual_observation("screen_ui_candidate", observed_features={"screen_ui_features": {"path_metadata": "only"}})
    assert observation["persistence_write_performed"] is False
    assert observation["runtime_mapping_enabled_default"] is False
    assert observation["enforcement_enabled_default"] is False
    assert observation["vector_contents_read"] is False
    assert observation["vectors_loaded"] is False
    assert observation["artifact_created_or_staged"] is False


def test_korean_fixtures_and_minseok_token_remain_preserved():
    fixture_text = Path("tests/fixtures/korean_conversation_fixtures.py").read_text(encoding="utf-8")
    main_text = Path("main.py").read_text(encoding="utf-8")
    assert "KOREAN_CONVERSATION_FIXTURES" in fixture_text
    assert "민석" in main_text
    observation = build_visual_observation("user_uploaded_image", observed_features={"scene_features": {"caption_candidate": "민석"}})
    assert observation["observed_features"]["scene_features"]["caption_candidate"] == "민석"
    assert observation["visual_observation_passed"] is True
