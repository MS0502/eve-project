"""Tests for Round1061-1080 auditory observation schema."""

from adapters.auditory_observation_schema import (
    build_auditory_observation,
    validate_auditory_observation,
    build_auditory_observation_to_sensory_observation,
    build_auditory_observation_to_origin_fact_status_record,
    build_auditory_observation_to_event_candidate_plan,
    build_auditory_observation_memory_candidate_plan,
)

def test_build_voice_message_candidate():
    obs = build_auditory_observation(
        "voice_message_candidate",
        observed_features={"acoustic_features": {"duration": 5.0}},
        metadata={"private_audio": True}
    )
    assert obs["auditory_observation_passed"] is True
    assert obs["modality"] == "auditory"
    assert obs["source"] == "voice_message_candidate"
    assert "voice_or_speech_data_requires_privacy_handling" in obs["privacy_flags"]
    assert "private_audio_flag_set" in obs["privacy_flags"]
    assert obs["microphone_activated"] is False
    assert obs["speech_to_text_performed"] is False

    validation = validate_auditory_observation(obs)
    assert validation["validation_passed"] is True

def test_build_microphone_frame_candidate():
    obs = build_auditory_observation(
        "microphone_frame_candidate",
        metadata={"confidence": {"overall": "low"}}
    )
    assert obs["auditory_observation_passed"] is True
    assert obs["source"] == "microphone_frame_candidate"
    assert obs["microphone_activated"] is False
    assert "low_confidence_auditory_inference_low" in obs["uncertainty_flags"]

    validation = validate_auditory_observation(obs)
    assert validation["validation_passed"] is True

def test_unknown_source_fails_closed():
    obs = build_auditory_observation("unknown_source")
    assert obs["auditory_observation_passed"] is False
    assert obs["auditory_observation_status"] == "auditory_observation_failed_closed"
    assert "unknown_auditory_source_unknown_source" in obs["blocked_reasons"]

    validation = validate_auditory_observation(obs)
    assert validation["validation_passed"] is False

def test_raw_audio_fails_closed():
    obs = build_auditory_observation(
        "voice_message_candidate",
        metadata={"raw_audio_data_present": True}
    )
    assert obs["auditory_observation_passed"] is False
    assert "raw_audio_data_must_not_enter_auditory_observation_schema" in obs["blocked_reasons"]
    assert obs["raw_audio_data_present"] is True


def test_auditory_observation_to_sensory_contract():
    obs = build_auditory_observation("environmental_sound_candidate")
    sensory = build_auditory_observation_to_sensory_observation(obs)
    assert sensory["observation_passed"] is True
    assert sensory["modality"] == "auditory"
    assert sensory["source"] == "environmental_sound_candidate"
    assert sensory["microphone_activated"] is False

def test_auditory_observation_to_origin_fact_status():
    obs = build_auditory_observation("notification_sound_candidate")
    record = build_auditory_observation_to_origin_fact_status_record(obs)
    assert record["origin_fact_status_passed"] is True
    assert record["origin"] == "external_audio"
    assert record["fact_status"] == "observed_external"

def test_dmn_inner_voice_origin_fact_status():
    obs = build_auditory_observation("dmn_inner_voice_candidate")
    record = build_auditory_observation_to_origin_fact_status_record(obs)
    assert record["origin_fact_status_passed"] is True
    assert record["origin"] == "dream_dmn"
    assert record["fact_status"] == "symbolic_visualization"

def test_auditory_memory_candidate_plan_quarantine():
    obs = build_auditory_observation("voice_message_candidate")
    plan = build_auditory_observation_memory_candidate_plan(obs)
    assert plan["plan_passed"] is True
    assert plan["memory_write_performed"] is False
    assert plan["quarantine_required"] is True

def test_candidate_only_inferences():
    obs = build_auditory_observation(
        "voice_message_candidate",
        observed_features={"speech_content": "hello"},
        inferred_features={
            "speaker_identity_candidate": "minseok",
            "emotion_candidate": "happy",
            "intent_candidate": "greeting",
            "prosody_state_candidate": "excited"
        }
    )
    assert obs["auditory_observation_passed"] is True
    assert "speaker_identity_candidate" in obs["candidate_only_fields"]
    assert "emotion_candidate" in obs["candidate_only_fields"]
    assert "intent_candidate" in obs["candidate_only_fields"]
    assert "prosody_state_candidate" in obs["candidate_only_fields"]
    assert "speech_content_should_be_candidate_unless_provided_as_metadata" in obs["warnings"]

def test_speech_content_candidate_if_no_transcript_metadata():
    obs = build_auditory_observation(
        "speech_transcript_candidate",
        inferred_features={"speech_content_candidate": "hello"}
    )
    assert "speech_content" in obs["candidate_only_fields"]

def test_speech_content_not_candidate_if_transcript_metadata():
    obs = build_auditory_observation(
        "speech_transcript_candidate",
        inferred_features={"speech_content_candidate": "hello"},
        metadata={"transcript_metadata_provided": True}
    )
    assert "speech_content" not in obs["candidate_only_fields"]

def test_anti_global_synchrony_and_no_mutation():
    from adapters.auditory_observation_schema import build_auditory_observation_schema_summary
    summary = build_auditory_observation_schema_summary()
    assert summary["anti_global_synchrony_blocked"] is True

    obs = build_auditory_observation("environmental_sound_candidate")
    assert obs["memory_write_performed"] is False
    assert obs["self_model_update_allowed"] is False
    assert obs["affect_transition_allowed"] is False
    assert obs["hormone_transition_allowed"] is False
    assert obs["persistence_write_performed"] is False
    assert obs["vector_read_performed"] is False
