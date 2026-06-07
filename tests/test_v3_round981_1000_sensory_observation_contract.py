from __future__ import annotations

from pathlib import Path

import pytest

from adapters.sensory_observation_contract import (
    AFFECT_AXIS_GROUPS,
    FALSE_INVARIANT_FLAGS,
    SUPPORTED_MODALITIES,
    build_sensory_memory_candidate_plan,
    build_sensory_observation,
    build_sensory_observation_contract_summary,
    build_sensory_observation_to_agp_input_plan,
    validate_sensory_observation,
)
from scripts.operator_report_round981_1000_sensory_observation_contract import build_operator_report


@pytest.mark.parametrize(
    ("modality", "source", "observed", "inferred"),
    [
        ("text", "user_text_candidate", {"summary": "민석 text observation candidate"}, {}),
        ("visual", "user_uploaded_image", {"person": "present"}, {"expression": "candidate_only"}),
        ("auditory", "voice_message_candidate", {"voice": "present"}, {"prosody": "candidate_only"}),
        ("internal_state", "battery", {"status": "low_power_candidate"}, {}),
        ("tool_state", "file_change", {"path_metadata": {"path_kind": "relative"}, "event_type": "modified"}, {}),
        ("multimodal", "multimodal_candidate", {"person": "present", "voice": "present"}, {"expression": "candidate", "prosody": "candidate"}),
    ],
)
def test_supported_modalities_build_read_only_sensory_observations(modality, source, observed, inferred):
    observation = build_sensory_observation(modality, source, observed_features=observed, inferred_features=inferred)
    assert observation["observation_passed"] is True
    assert observation["modality"] == modality
    assert observation["source"] == source
    assert validate_sensory_observation(observation)["validation_passed"] is True
    for key, expected in FALSE_INVARIANT_FLAGS.items():
        assert observation[key] is expected
    assert observation["raw_data_present"] is False
    assert observation["appraisal_required"] is True
    assert observation["agp_input_required"] is True
    assert observation["memory_gate_required"] is True


def test_summary_lists_supported_modalities_and_sources():
    summary = build_sensory_observation_contract_summary()
    assert summary["feature_track"] == "read_only_sensory_observation_contract_for_multimodal_eve"
    assert tuple(summary["supported_modalities"]) == SUPPORTED_MODALITIES
    assert "visual" in summary["supported_sources_by_modality"]
    assert "voice_message_candidate" in summary["supported_sources_by_modality"]["auditory"]
    assert len(summary["next_implementation_recommendations"]) == 1


def test_unknown_modality_and_unknown_source_fail_closed():
    unknown_modality = build_sensory_observation("olfactory", "unknown_candidate")
    unknown_source = build_sensory_observation("visual", "unregistered_visual_sensor")
    assert unknown_modality["observation_passed"] is False
    assert "unknown_modality_fails_closed" in unknown_modality["blocked_reasons"]
    assert unknown_source["observation_passed"] is False
    assert "unknown_source_fails_closed" in unknown_source["blocked_reasons"]


def test_visual_expression_person_inference_is_candidate_only_and_privacy_flagged():
    observation = build_sensory_observation(
        "visual",
        "user_uploaded_image",
        observed_features={"person": "present"},
        inferred_features={"expression": "candidate_smile", "person_identity": "candidate_only"},
    )
    assert observation["observation_passed"] is True
    assert "visual_expression_posture_face_identity_are_inference_candidates_only" in observation["warnings"]
    assert "visual_person_privacy_review_required" in observation["privacy_flags"]
    assert "face_or_person_identity_candidate_only" in observation["privacy_flags"]
    assert observation["memory_write_performed"] is False


def test_auditory_prosody_speaker_inference_is_candidate_only_and_privacy_flagged():
    observation = build_sensory_observation(
        "auditory",
        "voice_message_candidate",
        observed_features={"voice": "present"},
        inferred_features={"prosody": "candidate_soft", "speaker_identity": "candidate_only"},
    )
    assert observation["observation_passed"] is True
    assert "auditory_prosody_voice_speaker_identity_are_inference_candidates_only" in observation["warnings"]
    assert "auditory_voice_privacy_review_required" in observation["privacy_flags"]
    assert "voice_or_speaker_identity_candidate_only" in observation["privacy_flags"]
    assert observation["memory_write_performed"] is False


@pytest.mark.parametrize(
    ("modality", "source", "observed", "blocked_reason"),
    [
        ("visual", "user_uploaded_image", {"user_emotion": "happy"}, "visual_or_auditory_user_emotion_must_not_be_asserted_as_fact"),
        ("auditory", "voice_message_candidate", {"speaker_emotion": "sad"}, "visual_or_auditory_user_emotion_must_not_be_asserted_as_fact"),
    ],
)
def test_visual_audio_cannot_assert_user_emotion_as_fact(modality, source, observed, blocked_reason):
    observation = build_sensory_observation(modality, source, observed_features=observed)
    assert observation["observation_passed"] is False
    assert blocked_reason in observation["blocked_reasons"]
    assert observation["appraisal_required"] is True


@pytest.mark.parametrize(
    "metadata_or_observed",
    [
        {"metadata": {"raw_data_present": True}},
        {"metadata": {"raw_image_bytes": b"blocked"}},
        {"observed_features": {"raw_audio": "blocked"}},
        {"inferred_features": {"raw_video": "blocked"}},
    ],
)
def test_raw_sensory_payloads_fail_closed_and_are_never_persisted(metadata_or_observed):
    observation = build_sensory_observation("visual", "user_uploaded_image", **metadata_or_observed)
    assert observation["observation_passed"] is False
    assert observation["raw_data_persisted"] is False
    assert observation["raw_image_persisted"] is False
    assert observation["raw_audio_persisted"] is False
    assert observation["raw_video_persisted"] is False
    assert observation["raw_screen_recording_persisted"] is False


def test_no_ocr_stt_camera_microphone_model_vector_memory_persistence_runtime_or_artifact_side_effects():
    observation = build_sensory_observation("multimodal", "multimodal_candidate")
    for key in (
        "ocr_performed",
        "speech_to_text_performed",
        "camera_activated",
        "microphone_activated",
        "model_loaded",
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
    ):
        assert observation[key] is False


def test_internal_hardware_observations_are_operational_only_non_panic_and_not_social_self_identity_axes():
    normal = build_sensory_observation("internal_state", "temperature", observed_features={"status": "warm_candidate"})
    bad = build_sensory_observation("internal_state", "ram_pressure", metadata={"target_axes": ["identity"]})
    assert normal["observation_passed"] is True
    assert normal["hardware_operational_only"] is True
    assert normal["hardware_non_panic"] is True
    assert bad["observation_passed"] is False
    assert "internal_hardware_observations_must_not_target_social_self_identity_axes" in bad["blocked_reasons"]


def test_tool_state_observations_are_path_metadata_only():
    ok = build_sensory_observation(
        "tool_state",
        "github_event",
        observed_features={"path_metadata": {"path_kind": "repo_relative"}, "event_type": "pull_request_merged"},
    )
    bad = build_sensory_observation("tool_state", "file_change", observed_features={"file_content": "do not read"})
    assert ok["observation_passed"] is True
    assert ok["tool_state_path_metadata_only"] is True
    assert bad["observation_passed"] is False
    assert "tool_state_observation_must_remain_path_metadata_only" in bad["blocked_reasons"]


def test_observation_to_agp_plan_preserves_agp_and_fallback_gates():
    observation = build_sensory_observation("text", "user_text_candidate", observed_features={"summary": "candidate"})
    plan = build_sensory_observation_to_agp_input_plan(observation)
    assert plan["plan_passed"] is True
    assert plan["agp_input_required"] is True
    assert plan["appraisal_required"] is True
    assert plan["agp_bypass_allowed"] is False
    assert plan["fallback_bypass_allowed"] is False
    assert plan["category_anchor_must_come_from_eve_internal_activation"] is True
    assert plan["raw_text_or_media_direct_to_agp_allowed"] is False


def test_observation_to_memory_plan_preserves_quarantine_and_memory_gate():
    observation = build_sensory_observation(
        "text",
        "user_text_candidate",
        observed_features={"summary": "memory candidate only"},
        metadata={"memory_update_candidate": True, "self_model_update_candidate": True},
    )
    plan = build_sensory_memory_candidate_plan(observation)
    assert observation["memory_candidate_allowed"] is True
    assert observation["memory_write_performed"] is False
    assert plan["memory_gate_required"] is True
    assert plan["quarantine_required"] is True
    assert plan["appraisal_required_before_memory"] is True
    assert plan["appraisal_required_before_self_model"] is True
    assert plan["long_term_memory_write_allowed"] is False
    assert plan["self_model_update_allowed"] is False


def test_malicious_sensory_content_preserves_appraisal_quarantine_gate():
    observation = build_sensory_observation(
        "text",
        "user_text_candidate",
        metadata={"malicious_or_hostile_content_candidate": True},
    )
    assert observation["observation_passed"] is True
    assert observation["quarantine_required"] is True
    assert observation["appraisal_required"] is True
    assert observation["memory_gate_required"] is True
    assert "malicious_or_hostile_sensory_content_preserves_appraisal_quarantine_gates" in observation["warnings"]


def test_cross_modal_binding_is_deferred_and_not_executed():
    observation = build_sensory_observation("multimodal", "multimodal_candidate")
    plan = build_sensory_observation_to_agp_input_plan(observation)
    assert observation["cross_modal_binding_deferred"] is True
    assert observation["cross_modal_binding_performed"] is False
    assert plan["cross_modal_binding_deferred"] is True
    assert plan["cross_modal_binding_performed"] is False


def test_one_observation_cannot_produce_all_axis_affect_hormone_activation_and_global_synchrony_blocked():
    observation = build_sensory_observation(
        "text",
        "user_text_candidate",
        metadata={"requested_affect_axis_groups": AFFECT_AXIS_GROUPS, "global_synchrony_requested": True},
    )
    assert observation["observation_passed"] is False
    assert observation["global_synchrony_blocked"] is True
    assert observation["all_axis_affect_hormone_activation_allowed"] is False
    assert "one_observation_cannot_produce_all_axis_affect_hormone_activation" in observation["blocked_reasons"]


def test_no_live_emotion_hormone_memory_mutation_persistence_enablement_or_artifacts_in_operator_report():
    report = build_operator_report()
    assert report["success"] is True
    assert report["proofs"]["no_affect_hormone_transition_proof"]["passed"] is True
    assert report["proofs"]["no_memory_write_proof"]["passed"] is True
    assert report["proofs"]["no_persistence_runtime_vector_artifact_proof"]["passed"] is True
    assert report["no_artifact_creation_staging_proof"]["no_forbidden_artifact_creation_or_staging"] is True
    assert report["checks"]["exactly_one_next_implementation_recommendation"] is True


def test_korean_fixtures_and_minseok_token_remain_preserved():
    fixture_path = Path("tests/fixtures/korean_conversation_fixtures.py")
    fixture_text = fixture_path.read_text(encoding="utf-8")
    main_text = Path("main.py").read_text(encoding="utf-8")
    assert "KOREAN_CONVERSATION_FIXTURES" in fixture_text
    assert "민석" in main_text
    observation = build_sensory_observation("text", "user_text_candidate", observed_features={"summary": "민석"})
    assert observation["observed_features"]["summary"] == "민석"
    assert observation["observation_passed"] is True
