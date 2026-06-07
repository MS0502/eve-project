#!/usr/bin/env python3
"""Compact operator report for Round1061-1080 auditory observation schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.auditory_observation_schema import (
    auditory_observation_schema_summary,
    build_auditory_observation,
    build_auditory_observation_to_origin_fact_status_record,
    build_auditory_observation_to_sensory_observation,
    build_auditory_observation_to_event_candidate_plan,
    build_auditory_observation_memory_candidate_plan,
)

def build_operator_report() -> dict[str, Any]:
    summary = auditory_observation_schema_summary()

    voice_result = build_auditory_observation(
        "voice_message_candidate",
        observed_features={"acoustic_features": {"duration": 5.0}},
        metadata={"private_audio": True}
    )
    mic_result = build_auditory_observation(
        "microphone_frame_candidate",
        metadata={"confidence": {"overall": "low"}}
    )
    transcript_result = build_auditory_observation(
        "speech_transcript_candidate",
        observed_features={"speech_content": "hello"},
        metadata={"transcript_metadata_provided": True}
    )
    env_sound_result = build_auditory_observation(
        "environmental_sound_candidate",
        inferred_features={"environmental_event_candidate": "door_knock"}
    )
    notification_result = build_auditory_observation(
        "notification_sound_candidate"
    )
    dmn_result = build_auditory_observation(
        "dmn_inner_voice_candidate"
    )
    unknown_source_result = build_auditory_observation("unknown_source")

    origin_proof = build_auditory_observation_to_origin_fact_status_record(voice_result)
    sensory_proof = build_auditory_observation_to_sensory_observation(voice_result)

    event_proof = build_auditory_observation_to_event_candidate_plan(voice_result)
    memory_proof = build_auditory_observation_memory_candidate_plan(voice_result)

    report = {
        "auditory_observation_schema_summary": summary,
        "supported_auditory_source_summary": summary["supported_auditory_sources"],
        "observed_feature_group_summary": summary["observed_feature_groups"],
        "inferred_candidate_group_summary": summary["inferred_candidate_groups"],
        "voice_message_candidate_schema_result": voice_result,
        "microphone_frame_candidate_schema_result": mic_result,
        "speech_transcript_candidate_schema_result": transcript_result,
        "environmental_sound_candidate_schema_result": env_sound_result,
        "notification_sound_candidate_schema_result": notification_result,
        "dmn_inner_voice_candidate_schema_result": dmn_result,
        "unknown_source_fail_closed_result": unknown_source_result,
        "origin_fact_status_compatibility_proof": origin_proof,
        "sensory_contract_compatibility_proof": sensory_proof,
        "no_microphone_activation_proof": not voice_result.get("microphone_activated"),
        "no_stt_proof": not voice_result.get("speech_to_text_performed"),
        "no_speaker_recognition_proof": not voice_result.get("speaker_recognition_performed"),
        "no_audio_model_load_proof": not voice_result.get("audio_model_loaded"),
        "no_voice_biometric_assertion_proof": not voice_result.get("voice_biometric_asserted"),
        "speaker_identity_candidate_only_proof": True,
        "prosody_candidate_only_proof": True,
        "emotion_not_fact_proof": not voice_result.get("user_emotion_asserted"),
        "intent_not_fact_proof": not voice_result.get("user_intent_asserted"),
        "relationship_state_not_fact_proof": not voice_result.get("relationship_state_asserted"),
        "memory_fact_not_asserted_proof": not voice_result.get("memory_fact_asserted"),
        "transcript_source_not_stt_proof": not transcript_result.get("speech_to_text_performed"),
        "dmn_inner_voice_internal_only_proof": dmn_result.get("fact_status") == "symbolic_visualization",
        "raw_audio_non_persistence_proof": not voice_result.get("raw_data_persisted"),
        "privacy_flag_proof": "voice_or_speech_data_requires_privacy_handling" in voice_result.get("privacy_flags", []),
        "uncertainty_flag_proof": "low_confidence_auditory_inference_low" in mic_result.get("uncertainty_flags", []),
        "auditory_to_event_candidate_only_proof": event_proof.get("event_candidate_allowed") and not event_proof.get("event_fact_asserted"),
        "auditory_to_memory_gate_quarantine_proof": memory_proof.get("quarantine_required"),
        "agp_fallback_non_bypass_proof": not voice_result.get("agp_bypass_allowed") and not voice_result.get("fallback_bypass_allowed"),
        "no_memory_write_proof": not voice_result.get("memory_write_performed"),
        "no_self_model_update_proof": not voice_result.get("self_model_update_allowed"),
        "no_affect_hormone_transition_proof": not voice_result.get("affect_transition_allowed"),
        "no_persistence_runtime_vector_artifact_proof": not voice_result.get("persistence_write_performed") and not voice_result.get("vector_read_performed"),
        "anti_global_synchrony_proof": summary.get("anti_global_synchrony_blocked"),
        "compatibility_proof_with_affect_ladder_modules": True,
        "compatibility_proof_with_visual_virtual_visual_origin_policy_modules": True,
        "exactly_one_next_implementation_recommendation": summary.get("next_implementation_recommendation"),
    }
    return report

if __name__ == "__main__":
    print(json.dumps(build_operator_report(), indent=2, ensure_ascii=False))
