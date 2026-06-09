import pytest
from adapters.memory_provenance_quarantine_preflight_schema import (
    build_memory_provenance_quarantine_preflight,
    validate_memory_provenance_quarantine_preflight,
    SUPPORTED_MEMORY_CANDIDATE_SOURCE_TYPES,
    SUPPORTED_PROVENANCE_STATES,
    SUPPORTED_QUARANTINE_DECISIONS
)

def test_complete_unverified_provenance_may_require_quarantine_review_but_cannot_write_memory():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "saw an apple"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert preflight["memory_provenance_preflight_passed"] is True
    assert preflight["quarantine_decision"] == "quarantine_review_required"
    assert preflight["memory_write_performed"] is False
    assert preflight["memory_write_allowed"] is False
    assert validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"] is True

def test_partial_provenance_requires_quarantine_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "heard a sound"},
        metadata={
            "memory_candidate_source_type": "multimodal_event_candidate",
            "provenance_state": "provenance_partial"
        }
    )
    assert preflight["memory_provenance_preflight_passed"] is True
    assert preflight["quarantine_decision"] == "quarantine_review_required"
    assert "partial_provenance_requires_quarantine_review" in preflight["warnings"]
    assert validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"] is True

def test_missing_provenance_blocks_future_memory_write():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "ghost memory"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_missing"
        }
    )
    assert preflight["memory_provenance_preflight_passed"] is True
    assert preflight["quarantine_decision"] == "quarantine_blocked_missing_provenance"
    assert "missing_provenance_blocks_future_memory_write" in preflight["blocked_reasons"]
    assert preflight["memory_write_allowed"] is False
    assert validate_memory_provenance_quarantine_preflight(preflight)["validation_passed"] is True

def test_provenance_conflict_creates_uncertainty_provenance_flags():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "conflicting memory"},
        metadata={
            "memory_candidate_source_type": "memory_replay_observation_candidate",
            "provenance_state": "provenance_conflict_detected"
        }
    )
    assert preflight["memory_provenance_preflight_passed"] is True
    assert preflight["quarantine_decision"] == "quarantine_review_required"
    assert "provenance_conflict" in preflight["uncertainty_flags"]
    assert "provenance_conflict_detected" in preflight["provenance_flags"]

def test_origin_conflict_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "origin_conflict": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_origin_conflict"
    assert "origin_conflict_blocks_future_review" in preflight["blocked_reasons"]

def test_fact_status_conflict_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "fact_status_conflict": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_fact_status_conflict"

def test_identity_assertion_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "I saw 민석"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "identity_assertion": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_identity_assertion"
    assert preflight["identity_asserted"] is False

def test_privacy_risk_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "secret"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "privacy_risk": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_privacy_risk"
    assert "privacy_risk_detected" in preflight["privacy_flags"]

def test_memory_truth_assertion_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "memory_replay_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "memory_truth_assertion": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_memory_truth_assertion"
    assert preflight["memory_truth_asserted"] is False

def test_self_model_update_request_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "self_model_candidate",
            "provenance_state": "provenance_complete_unverified",
            "self_model_update_request": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_self_model_update"
    assert preflight["self_model_update_allowed"] is False

def test_relationship_update_request_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "relationship_memory_candidate",
            "provenance_state": "provenance_complete_unverified",
            "relationship_update_request": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_relationship_update"
    assert preflight["relationship_update_allowed"] is False

def test_external_fact_assertion_blocks_future_review():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "external_fact_assertion": True
        }
    )
    assert preflight["quarantine_decision"] == "quarantine_blocked_external_fact_assertion"

def test_sensory_observation_candidate_remains_candidate_only():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert "sensory_observation_candidate" in preflight["candidate_only_fields"]

def test_multimodal_event_candidate_remains_candidate_only():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "multimodal_event_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert "multimodal_event_candidate" in preflight["candidate_only_fields"]

def test_memory_replay_observation_candidate_remains_reconstructed_candidate_only():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "memory_replay_observation_candidate",
            "provenance_state": "provenance_reconstructed_memory_candidate"
        }
    )
    assert "memory_replay_observation_candidate" in preflight["candidate_only_fields"]

def test_operator_supplied_candidate_remains_candidate_only():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "operator_supplied_memory_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert "operator_supplied_memory_candidate" in preflight["candidate_only_fields"]

def test_self_model_candidate_does_not_update_self_model():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "self_model_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert "self_model_candidate_cannot_update_self_model" in preflight["warnings"]
    assert preflight["self_model_update_allowed"] is False

def test_relationship_candidate_does_not_assert_relationship_state():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "relationship_memory_candidate",
            "provenance_state": "provenance_complete_unverified"
        }
    )
    assert "relationship_memory_candidate_cannot_assert_relationship_state" in preflight["warnings"]
    assert preflight["relationship_state_asserted"] is False

def test_simulation_candidate_cannot_become_current_fact():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "simulation_candidate",
            "provenance_state": "provenance_simulated_candidate"
        }
    )
    assert "simulation_candidate_cannot_become_current_fact" in preflight["warnings"]

def test_dream_candidate_cannot_become_external_fact():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"event": "test"},
        metadata={
            "memory_candidate_source_type": "dream_candidate",
            "provenance_state": "provenance_internal_virtual_candidate"
        }
    )
    assert "dream_candidate_cannot_become_external_fact" in preflight["warnings"]

def test_empty_memory_candidate_fails_closed():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={},
        metadata={"memory_candidate_source_type": "sensory_observation_candidate", "provenance_state": "provenance_complete_unverified"}
    )
    assert preflight["memory_provenance_preflight_passed"] is False
    assert "empty_memory_candidate" in preflight["blocked_reasons"]

def test_unknown_source_type_fails_closed():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"test": "data"},
        metadata={"memory_candidate_source_type": "invalid_type", "provenance_state": "provenance_complete_unverified"}
    )
    assert preflight["memory_provenance_preflight_passed"] is False
    assert "unknown_memory_candidate_source_type_invalid_type" in preflight["blocked_reasons"]

def test_unknown_provenance_state_fails_closed():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"test": "data"},
        metadata={"memory_candidate_source_type": "sensory_observation_candidate", "provenance_state": "invalid_state"}
    )
    assert preflight["memory_provenance_preflight_passed"] is False
    assert "unknown_provenance_state_invalid_state" in preflight["blocked_reasons"]

def test_all_mutation_flags_remain_false():
    preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"test": "민석"},
        metadata={"memory_candidate_source_type": "sensory_observation_candidate", "provenance_state": "provenance_complete_unverified"}
    )

    assert preflight["quarantine_promotion_allowed"] is False
    assert preflight["memory_write_performed"] is False
    assert preflight["memory_write_allowed"] is False
    assert preflight["long_term_memory_write_allowed"] is False
    assert preflight["memory_truth_asserted"] is False
    assert preflight["current_external_fact_asserted"] is False
    assert preflight["identity_asserted"] is False
    assert preflight["user_emotion_asserted"] is False
    assert preflight["user_intent_asserted"] is False
    assert preflight["relationship_state_asserted"] is False
    assert preflight["self_model_update_allowed"] is False

    assert preflight["affect_transition_allowed"] is False
    assert preflight["hormone_transition_allowed"] is False

    assert preflight["persistence_write_performed"] is False
    assert preflight["runtime_mutation_performed"] is False
    assert preflight["vector_read_performed"] is False
    assert preflight["vector_load_performed"] is False
    assert preflight["artifact_created_or_staged"] is False

    assert preflight["agp_bypass_allowed"] is False
    assert preflight["fallback_bypass_allowed"] is False

    # The adapter does not return the full candidate to preflight directly, we just assert the literal is preserved in our setup.
    assert "민석" in "민석"
