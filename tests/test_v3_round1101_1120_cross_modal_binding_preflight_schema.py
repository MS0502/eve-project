import pytest
from adapters.cross_modal_binding_preflight_schema import (
    build_cross_modal_binding_preflight,
    validate_cross_modal_binding_preflight,
    build_cross_modal_binding_preflight_schema_summary
)

def test_eligible_multimodal_event_candidate():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["preflight_passed"] is True
    assert preflight["preflight_decision"] == "eligible_for_future_binding"
    assert preflight["future_binding_attempt_allowed"] is True

    valid = validate_cross_modal_binding_preflight(preflight)
    assert valid["validation_passed"] is True

def test_eligible_result_does_not_perform_binding():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["binding_performed"] is False
    assert preflight["cross_modal_binding_performed"] is False
    assert preflight["future_binding_attempt_allowed"] is True

def test_external_visual_auditory_candidate_future_attempt_not_asserted_event():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate", "uncertainty_flags": []}
    preflight = build_cross_modal_binding_preflight(candidate, ["audio_visual_context_hypothesis"])

    assert preflight["preflight_passed"] is True
    assert preflight["preflight_decision"] == "eligible_for_future_binding"
    assert preflight["future_binding_attempt_allowed"] is True
    assert preflight["real_world_event_asserted"] is False
    assert preflight["external_fact_asserted"] is False

def test_mixed_boundary_candidate_is_blocked():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "mixed_boundary_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_mixed_boundary"

def test_virtual_visual_external_auditory_is_blocked_unless_future_separated():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "mixed_boundary_candidate", "uncertainty_flags": ["mixed_boundary_uncertainty"]}

    # Blocked by default
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])
    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_mixed_boundary"

    # Allowed if explicitly future separated
    preflight_separated = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"explicitly_future_separated": True})
    assert preflight_separated["preflight_passed"] is True
    assert preflight_separated["preflight_decision"] == "eligible_for_future_binding"

def test_dmn_symbolic_candidate_remains_internal_symbolic():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "dmn_symbolic_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["internal_symbolic_alignment_hypothesis"])

    assert preflight["preflight_passed"] is True
    assert preflight["preflight_decision"] == "eligible_for_future_binding"
    assert preflight["dmn_fact_asserted"] is False
    assert preflight["real_world_event_asserted"] is False

def test_memory_replay_cannot_become_external_fact():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "memory_replay_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["memory_replay_alignment_hypothesis"], {"intended_as_external": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_memory_replay_as_external"
    assert preflight["external_fact_asserted"] is False

def test_imagination_cannot_become_fact():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "imagination_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["semantic_alignment_hypothesis"], {"intended_as_memory": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_imagination_as_fact"
    assert preflight["imagination_fact_asserted"] is False

def test_simulation_cannot_become_current_fact():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "simulation_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["semantic_alignment_hypothesis"], {"intended_as_current": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_simulation_as_current_fact"
    assert preflight["simulation_fact_asserted"] is False

def test_unknown_candidate_fails_closed():
    # Invalid candidate (doesn't have passed flag)
    candidate = {"candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_unknown_candidate"

def test_empty_candidate_fails_closed():
    preflight = build_cross_modal_binding_preflight({}, ["temporal_alignment_hypothesis"])

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_unknown_candidate"

def test_unknown_hypothesis_type_fails_closed():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["unknown_hypothesis"])

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_unknown_candidate"

def test_low_confidence_blocks_binding_attempt():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"low_confidence": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_low_confidence"

def test_privacy_risk_blocks_binding_attempt():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"privacy_risk": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_privacy_risk"

def test_origin_conflict_blocks_binding_attempt():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"origin_conflict": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_origin_conflict"

def test_fact_status_conflict_blocks_binding_attempt():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"fact_status_conflict": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_fact_status_conflict"

def test_identity_resolution_required_blocks_binding_attempt():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"], {"requires_identity_resolution": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_identity_resolution_required"

def test_face_voice_matching_remains_blocked():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["audio_visual_context_hypothesis"], {"attempt_face_voice_match": True})

    assert preflight["preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_identity_resolution_required"

def test_all_assertion_flags_remain_false():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["identity_asserted"] is False
    assert preflight["speaker_identity_asserted"] is False
    assert preflight["person_identity_asserted"] is False
    assert preflight["face_voice_match_asserted"] is False
    assert preflight["user_emotion_asserted"] is False
    assert preflight["user_intent_asserted"] is False
    assert preflight["relationship_state_asserted"] is False
    assert preflight["memory_fact_asserted"] is False

def test_all_recognition_ocr_stt_model_device_flags_remain_false():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["speaker_recognition_performed"] is False
    assert preflight["face_recognition_performed"] is False
    assert preflight["ocr_performed"] is False
    assert preflight["speech_to_text_performed"] is False
    assert preflight["model_loaded"] is False
    assert preflight["camera_activated"] is False
    assert preflight["microphone_activated"] is False

def test_all_memory_self_affect_hormone_persistence_runtime_vector_artifact_mutation_flags_remain_false():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["memory_write_performed"] is False
    assert preflight["long_term_memory_write_allowed"] is False
    assert preflight["self_model_update_allowed"] is False
    assert preflight["affect_transition_allowed"] is False
    assert preflight["hormone_transition_allowed"] is False
    assert preflight["persistence_write_performed"] is False
    assert preflight["runtime_mutation_performed"] is False
    assert preflight["vector_read_performed"] is False
    assert preflight["vector_load_performed"] is False
    assert preflight["artifact_created_or_staged"] is False
    assert preflight["raw_data_persisted"] is False
    assert preflight["global_synchrony_produced"] is False

def test_agp_fallback_bypass_flags_remain_false():
    candidate = {"multimodal_event_candidate_passed": True, "candidate_type": "external_event_candidate"}
    preflight = build_cross_modal_binding_preflight(candidate, ["temporal_alignment_hypothesis"])

    assert preflight["agp_bypass_allowed"] is False
    assert preflight["fallback_bypass_allowed"] is False

def test_korean_fixtures_preserved():
    # Preserve literal "민석"
    korean_string = "민석"
    candidate = {
        "multimodal_event_candidate_passed": True,
        "candidate_type": "external_event_candidate",
        "metadata": {"subject": korean_string}
    }
    preflight = build_cross_modal_binding_preflight(candidate, ["semantic_alignment_hypothesis"])

    assert preflight["preflight_passed"] is True
    # Verify the logic didn't mangle inputs if we ever need to use them, though preflight just returns a standard structure
    # Testing that it doesn't fail given a string with Korean chars
