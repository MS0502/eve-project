import pytest
from adapters.multimodal_event_candidate_contract import (
    build_multimodal_event_candidate,
    validate_multimodal_event_candidate,
    build_multimodal_event_candidate_contract_summary,
    build_multimodal_event_to_origin_fact_status_plan,
    build_multimodal_event_to_memory_candidate_plan,
    build_multimodal_event_to_appraisal_plan,
    build_multimodal_event_to_agp_input_plan
)

def test_external_visual_and_auditory():
    obs = [
        {"modality": "visual_external", "source": "camera"},
        {"modality": "auditory", "source": "microphone"}
    ]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "external_event_candidate"

def test_virtual_visual_and_dmn_inner_voice():
    obs = [
        {"modality": "visual_internal_virtual", "source": "virtual_room_view"},
        {"modality": "auditory", "source": "dmn_inner_voice_candidate"}
    ]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "dmn_symbolic_event_candidate"

def test_external_auditory_and_virtual_visual_mixed_boundary():
    obs = [
        {"modality": "visual_internal_virtual", "source": "virtual_room_view"},
        {"modality": "auditory", "source": "microphone"}
    ]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "mixed_boundary_candidate"
    assert "mixed_boundary_uncertainty" in candidate["uncertainty_flags"]
    assert "cross_domain_boundary_detected" in candidate["boundary_flags"]

def test_memory_replay_cannot_become_external_fact():
    obs = [{"modality": "visual_external", "source": "camera"}]
    candidate = build_multimodal_event_candidate(obs, metadata={"memory_replay": True, "intended_as_external": True})
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "memory_replay_event_candidate"

def test_imagination_cannot_become_memory_fact():
    obs = [{"modality": "visual_internal_virtual", "source": "virtual_room_view"}]
    candidate = build_multimodal_event_candidate(obs, metadata={"candidate_type": "imagination_event_candidate", "intended_as_memory": True})
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "imagination_event_candidate"

def test_simulation_cannot_become_current_fact():
    obs = [{"modality": "visual_internal_virtual", "source": "virtual_room_view"}]
    candidate = build_multimodal_event_candidate(obs, metadata={"candidate_type": "simulation_event_candidate", "intended_as_current": True})
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["candidate_type"] == "simulation_event_candidate"

def test_empty_observations_fails_closed():
    candidate = build_multimodal_event_candidate([])
    assert candidate["multimodal_event_candidate_passed"] is False
    assert "empty_observations" in candidate["blocked_reasons"]

def test_unknown_modality_fails_closed():
    obs = [{"modality": "unknown_modality", "source": "camera"}]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["multimodal_event_candidate_passed"] is False
    assert "unknown_modality_unknown_modality" in candidate["blocked_reasons"]

def test_unknown_link_type_fails_closed():
    obs = [{"modality": "visual_external", "source": "camera"}]
    candidate = build_multimodal_event_candidate(obs, candidate_links=["unknown_link"])
    assert candidate["multimodal_event_candidate_passed"] is False
    assert "unknown_link_type_unknown_link" in candidate["blocked_reasons"]

def test_unknown_candidate_type_fails_closed():
    obs = [{"modality": "visual_external", "source": "camera"}]
    candidate = build_multimodal_event_candidate(obs, metadata={"candidate_type": "unknown_candidate"})
    assert candidate["multimodal_event_candidate_passed"] is False
    assert "unknown_candidate_type_unknown_candidate" in candidate["blocked_reasons"]

def test_all_flags_remain_false():
    obs = [{"modality": "visual_external", "source": "camera"}]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["identity_asserted"] is False
    assert candidate["speaker_identity_asserted"] is False
    assert candidate["person_identity_asserted"] is False
    assert candidate["user_emotion_asserted"] is False
    assert candidate["user_intent_asserted"] is False
    assert candidate["relationship_state_asserted"] is False
    assert candidate["memory_fact_asserted"] is False
    assert candidate["cross_modal_binding_performed"] is False
    assert candidate["identity_resolution_performed"] is False

    assert candidate["ocr_performed"] is False
    assert candidate["stt_performed"] is False
    assert candidate["recognition_performed"] is False
    assert candidate["model_loaded"] is False
    assert candidate["device_activated"] is False

    assert candidate["memory_written"] is False
    assert candidate["self_model_updated"] is False
    assert candidate["affect_transitioned"] is False
    assert candidate["hormone_transitioned"] is False

    assert candidate["persistence_written"] is False
    assert candidate["runtime_mutated"] is False
    assert candidate["vector_written"] is False
    assert candidate["artifact_written"] is False

    assert candidate["agp_bypassed"] is False
    assert candidate["fallback_bypassed"] is False

    assert validate_multimodal_event_candidate(candidate)["validation_passed"] is True

def test_korean_fixtures_and_minseok_preserved():
    obs = [{"modality": "visual_external", "source": "camera", "target_person": "민석"}]
    candidate = build_multimodal_event_candidate(obs)
    assert candidate["multimodal_event_candidate_passed"] is True
    assert candidate["observations"][0]["target_person"] == "민석"
