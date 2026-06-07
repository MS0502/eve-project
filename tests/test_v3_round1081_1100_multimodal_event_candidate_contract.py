"""Tests for Round1081-1100 read-only multimodal event candidate contract."""

import pytest
from adapters.multimodal_event_candidate_contract import (
    build_multimodal_event_candidate,
    validate_multimodal_event_candidate,
    build_multimodal_event_candidate_contract_summary,
)

def test_external_visual_and_auditory_builds_read_only_external_event_candidate_only():
    candidate = build_multimodal_event_candidate([
        {"modality": "visual_external"},
        {"modality": "auditory"}
    ])
    assert candidate["candidate_type"] == "external_event_candidate"
    assert not candidate["real_world_event_asserted"]
    assert not candidate["external_fact_asserted"]

def test_virtual_visual_and_dmn_inner_voice_builds_internal_symbolic_candidate_only():
    candidate = build_multimodal_event_candidate([
        {"modality": "visual_internal_virtual"},
        {"modality": "dmn_inner_voice"}
    ])
    assert candidate["candidate_type"] == "dmn_symbolic_event_candidate"

def test_external_auditory_and_virtual_visual_becomes_mixed_boundary_candidate():
    candidate = build_multimodal_event_candidate([
        {"modality": "auditory"},
        {"modality": "visual_internal_virtual"}
    ])
    assert candidate["candidate_type"] == "mixed_boundary_candidate"
    assert candidate["mixed_boundary_detected"] is True
    assert "mixed_boundary_detected" in candidate["uncertainty_flags"]
    assert "mixed_boundary_requires_isolation" in candidate["boundary_flags"]

def test_memory_replay_cannot_become_external_fact():
    candidate = build_multimodal_event_candidate([
        {"modality": "memory_replay"}
    ])
    assert candidate["candidate_type"] == "memory_replay_event_candidate"
    assert not candidate["external_fact_asserted"]

def test_imagination_cannot_become_memory_fact():
    candidate = build_multimodal_event_candidate([
        {"modality": "imagination"}
    ])
    assert candidate["candidate_type"] == "imagination_event_candidate"
    assert not candidate["memory_fact_asserted"]

def test_simulation_cannot_become_current_fact():
    candidate = build_multimodal_event_candidate([
        {"modality": "simulation"}
    ])
    assert candidate["candidate_type"] == "simulation_event_candidate"
    assert not candidate["real_world_event_asserted"]
    assert not candidate["external_fact_asserted"]

def test_empty_observations_fail_closed():
    candidate = build_multimodal_event_candidate([])
    assert "empty_observations_fail_closed" in candidate["blocked_reasons"]
    assert candidate["multimodal_event_candidate_status"] == "failed_closed"

def test_unknown_modality_fails_closed():
    candidate = build_multimodal_event_candidate([{"modality": "unknown"}])
    assert any("unknown_modality" in r for r in candidate["blocked_reasons"])
    assert candidate["multimodal_event_candidate_status"] == "failed_closed"

def test_unknown_candidate_link_type_fails_closed():
    candidate = build_multimodal_event_candidate([{"modality": "auditory"}], [{"link_type": "invalid_link"}])
    assert any("unknown_candidate_link" in r for r in candidate["blocked_reasons"])
    assert candidate["multimodal_event_candidate_status"] == "failed_closed"


def test_all_side_effect_and_mutation_flags_remain_false():
    candidate = build_multimodal_event_candidate([
        {"modality": "auditory"}
    ])
    validated = validate_multimodal_event_candidate(candidate)

    assert not validated["identity_asserted"]
    assert not validated["speaker_identity_asserted"]
    assert not validated["person_identity_asserted"]
    assert not validated["user_emotion_asserted"]
    assert not validated["user_intent_asserted"]
    assert not validated["relationship_state_asserted"]
    assert not validated["memory_fact_asserted"]
    assert not validated["virtual_fact_asserted_as_external"]
    assert not validated["imagination_fact_asserted"]
    assert not validated["simulation_fact_asserted"]
    assert not validated["dmn_fact_asserted"]
    assert not validated["cross_modal_binding_performed"]
    assert not validated["identity_resolution_performed"]
    assert not validated["speaker_recognition_performed"]
    assert not validated["face_recognition_performed"]
    assert not validated["ocr_performed"]
    assert not validated["speech_to_text_performed"]
    assert not validated["model_loaded"]
    assert not validated["camera_activated"]
    assert not validated["microphone_activated"]
    assert not validated["raw_data_persisted"]
    assert not validated["memory_write_performed"]
    assert not validated["long_term_memory_write_allowed"]
    assert not validated["self_model_update_allowed"]
    assert not validated["affect_transition_allowed"]
    assert not validated["hormone_transition_allowed"]
    assert not validated["agp_bypass_allowed"]
    assert not validated["fallback_bypass_allowed"]
    assert not validated["vector_read_performed"]
    assert not validated["vector_load_performed"]
    assert not validated["runtime_mutation_performed"]
    assert not validated["persistence_write_performed"]
    assert not validated["artifact_created_or_staged"]

def test_korean_fixtures_are_preserved():
    summary = build_multimodal_event_candidate_contract_summary()
    assert "민석" in summary["korean_fixtures_preserved"]
