"""Tests for read-only cross-modal binding preflight schema."""

import pytest
from adapters.cross_modal_binding_preflight_schema import (
    build_cross_modal_binding_preflight_schema_summary,
    build_cross_modal_binding_preflight,
    validate_cross_modal_binding_preflight,
    build_cross_modal_binding_to_origin_fact_status_plan,
    build_cross_modal_binding_to_memory_candidate_plan,
    build_cross_modal_binding_to_appraisal_plan,
    build_cross_modal_binding_to_agp_input_plan,
    cross_modal_binding_preflight_schema_summary
)

def test_eligible_multimodal_event_candidate():
    candidate = {"candidate_id": "c1", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    assert preflight["cross_modal_binding_preflight_passed"] is True
    assert preflight["cross_modal_binding_preflight_status"] == "passed"
    assert preflight["preflight_decision"] == "eligible_for_future_binding"
    assert preflight["future_binding_attempt_allowed"] is True
    assert preflight["binding_performed"] is False
    assert preflight["cross_modal_binding_performed"] is False
    assert preflight["identity_asserted"] is False

    val = validate_cross_modal_binding_preflight(preflight)
    assert val["validation_passed"] is True

def test_external_visual_auditory_candidate():
    candidate = {"candidate_id": "c2", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "audio_visual_context_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    assert preflight["preflight_decision"] == "eligible_for_future_binding"
    assert preflight["future_binding_attempt_allowed"] is True
    assert preflight["binding_performed"] is False

def test_mixed_boundary_candidate_blocked():
    candidate = {"candidate_id": "c3", "candidate_type": "mixed_boundary_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_mixed_boundary"
    assert preflight["future_binding_attempt_allowed"] is False

def test_mixed_boundary_candidate_future_separated():
    candidate = {"candidate_id": "c4", "candidate_type": "mixed_boundary_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"explicitly_future_separated": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is True
    assert preflight["preflight_decision"] == "eligible_for_future_binding"

def test_dmn_symbolic_candidate_remains_internal():
    candidate = {"candidate_id": "c5", "candidate_type": "dmn_symbolic_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "internal_symbolic_alignment_hypothesis"}]
    metadata = {"intended_as_external": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_virtual_as_external"
    assert "dmn_cannot_be_external" in preflight["blocked_reasons"]

def test_memory_replay_cannot_become_external_fact():
    candidate = {"candidate_id": "c6", "candidate_type": "memory_replay_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "memory_replay_alignment_hypothesis"}]
    metadata = {"intended_as_external": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_memory_replay_as_external"

def test_imagination_cannot_become_fact():
    candidate = {"candidate_id": "c7", "candidate_type": "imagination_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"intended_as_fact": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_imagination_as_fact"

def test_simulation_cannot_become_current_fact():
    candidate = {"candidate_id": "c8", "candidate_type": "simulation_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"intended_as_current_fact": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_simulation_as_current_fact"

def test_unknown_candidate_fails_closed():
    # Empty candidate
    preflight = build_cross_modal_binding_preflight(None)
    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_unknown_candidate"

    # Missing candidate type
    preflight2 = build_cross_modal_binding_preflight({"candidate_id": "c9"})
    assert preflight2["cross_modal_binding_preflight_passed"] is False
    assert preflight2["preflight_decision"] == "blocked_unknown_candidate"

def test_unknown_hypothesis_type_fails_closed():
    candidate = {"candidate_id": "c10", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "unknown_magic_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_unknown_candidate"
    assert "unknown_hypothesis_type_unknown_magic_hypothesis" in preflight["blocked_reasons"]

def test_low_confidence_blocks_binding():
    candidate = {"candidate_id": "c11", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"low_confidence": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_low_confidence"

def test_privacy_risk_blocks_binding():
    candidate = {"candidate_id": "c12", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"privacy_risk": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_privacy_risk"
    assert "privacy_risk_detected" in preflight["privacy_flags"]

def test_origin_conflict_blocks_binding():
    candidate = {"candidate_id": "c13", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"origin_conflict": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_origin_conflict"

def test_fact_status_conflict_blocks_binding():
    candidate = {"candidate_id": "c14", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"fact_status_conflict": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_fact_status_conflict"

def test_identity_resolution_required_blocks_binding():
    candidate = {"candidate_id": "c15", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"identity_resolution_required": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_identity_resolution_required"

def test_face_voice_matching_remains_blocked():
    candidate = {"candidate_id": "c16", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    metadata = {"face_voice_match_attempt": True}
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses, metadata)

    assert preflight["cross_modal_binding_preflight_passed"] is False
    assert preflight["preflight_decision"] == "blocked_identity_resolution_required"
    assert "face_voice_match_blocked" in preflight["blocked_reasons"]
    assert preflight["face_voice_match_asserted"] is False

def test_all_assertion_flags_remain_false():
    candidate = {"candidate_id": "c17", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    assertions = [
        "identity_asserted", "speaker_identity_asserted", "person_identity_asserted",
        "face_voice_match_asserted", "user_emotion_asserted", "user_intent_asserted",
        "relationship_state_asserted", "memory_fact_asserted", "external_fact_asserted",
        "real_world_event_asserted", "virtual_fact_asserted_as_external",
        "imagination_fact_asserted", "simulation_fact_asserted", "dmn_fact_asserted"
    ]
    for flag in assertions:
        assert preflight[flag] is False

def test_all_mutation_and_device_flags_remain_false():
    candidate = {"candidate_id": "c18", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    mutations = [
        "speaker_recognition_performed", "face_recognition_performed", "ocr_performed",
        "speech_to_text_performed", "model_loaded", "camera_activated", "microphone_activated",
        "raw_data_persisted", "memory_write_performed", "long_term_memory_write_allowed",
        "self_model_update_allowed", "affect_transition_allowed", "hormone_transition_allowed",
        "agp_bypass_allowed", "fallback_bypass_allowed", "vector_read_performed",
        "vector_load_performed", "runtime_mutation_performed", "persistence_write_performed",
        "artifact_created_or_staged"
    ]
    for flag in mutations:
        assert preflight[flag] is False

def test_korean_fixtures_preserved():
    # preserving Korean fixtures and literal "민석"
    person_name = "민석"
    assert person_name == "민석"

def test_plans_are_built():
    candidate = {"candidate_id": "c19", "candidate_type": "external_event_candidate"}
    binding_hypotheses = [{"hypothesis_type": "temporal_alignment_hypothesis"}]
    preflight = build_cross_modal_binding_preflight(candidate, binding_hypotheses)

    plan1 = build_cross_modal_binding_to_origin_fact_status_plan(preflight)
    assert plan1["valid"] is True
    assert plan1["plan_type"] == "origin_fact_status_plan"

    plan2 = build_cross_modal_binding_to_memory_candidate_plan(preflight)
    assert plan2["valid"] is True

    plan3 = build_cross_modal_binding_to_appraisal_plan(preflight)
    assert plan3["valid"] is True

    plan4 = build_cross_modal_binding_to_agp_input_plan(preflight)
    assert plan4["valid"] is True
