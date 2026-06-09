"""Tests for Round1121-1140 memory replay observation schema."""

from adapters.memory_replay_observation_schema import (
    build_memory_replay_observation,
    build_memory_replay_to_cross_modal_binding_preflight_plan
)

def test_episodic_replay_builds_reconstructed_memory_observation_only():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "saw a cat"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["origin_summary"]["origin"] == "memory_replay"
    assert obs["fact_status_summary"]["fact_status"] == "reconstructed_memory"
    assert obs["reconstructed_memory"] is True
    assert obs["replay_only"] is True

def test_semantic_replay_remains_candidate_only():
    metadata = {
        "replay_source_type": "semantic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "cats have four legs"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert "semantic_replay_candidate" in obs["candidate_only_fields"]

def test_narrative_replay_remains_candidate_only():
    metadata = {
        "replay_source_type": "narrative_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "narrative_memory_trace"
    }
    obs = build_memory_replay_observation({"content": "I went to the store"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert "narrative_replay_candidate" in obs["candidate_only_fields"]

def test_self_model_replay_does_not_update_self_model():
    metadata = {
        "replay_source_type": "self_model_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "I like cats"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["self_model_update_allowed"] is False
    assert "self_model_replay_cannot_update_self_model" in obs["warnings"]

def test_relationship_replay_does_not_assert_relationship_state():
    metadata = {
        "replay_source_type": "relationship_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "I know 민석"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["relationship_state_asserted"] is False
    assert "relationship_replay_cannot_assert_relationship_state" in obs["warnings"]

def test_dream_replay_cannot_become_external_fact():
    metadata = {
        "replay_source_type": "dream_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "dream_memory_like_content"
    }
    obs = build_memory_replay_observation({"content": "flying cats"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["current_external_fact_asserted"] is False
    assert obs["origin_summary"]["origin"] == "dream_dmn"
    assert obs["fact_status_summary"]["fact_status"] == "symbolic_visualization"
    assert "dream_replay_remains_dream_memory_like" in obs["warnings"]

def test_simulation_replay_cannot_become_current_fact():
    metadata = {
        "replay_source_type": "simulation_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "simulated_memory_like_content"
    }
    obs = build_memory_replay_observation({"content": "what if cats flew"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["current_external_fact_asserted"] is False
    assert obs["real_world_event_asserted"] is False
    assert obs["origin_summary"]["origin"] == "simulation"
    assert obs["fact_status_summary"]["fact_status"] == "simulated_future"
    assert "simulation_replay_remains_simulated_future" in obs["warnings"]

def test_operator_supplied_replay_remains_candidate_only():
    metadata = {
        "replay_source_type": "operator_supplied_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "injected memory"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert "operator_supplied_replay_candidate" in obs["candidate_only_fields"]

def test_empty_memory_fragment_fails_closed():
    obs = build_memory_replay_observation({}, {}, {})
    assert obs["memory_replay_observation_passed"] is False
    assert "empty_memory_fragment" in obs["blocked_reasons"]

def test_unknown_replay_source_type_fails_closed():
    metadata = {
        "replay_source_type": "unknown_source",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is False
    assert "unknown_replay_source_type_unknown_source" in obs["blocked_reasons"]

def test_unknown_confidence_state_fails_closed():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "unknown_state",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is False
    assert "unknown_replay_confidence_state_unknown_state" in obs["blocked_reasons"]

def test_unknown_boundary_class_fails_closed():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "unknown_class"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is False
    assert "unknown_replay_boundary_classification_unknown_class" in obs["blocked_reasons"]

def test_low_confidence_replay_blocks_future_binding_plan():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_low_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    plan = build_memory_replay_to_cross_modal_binding_preflight_plan(obs)
    assert plan["valid"] is False

def test_conflict_replay_creates_uncertainty_flags():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_conflict_detected",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert "replay_conflict_detected" in obs["uncertainty_flags"]

def test_mixed_replay_boundary_creates_boundary_flags():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "mixed_replay_boundary"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert "mixed_replay_boundary" in obs["boundary_flags"]

def test_memory_truth_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_truth_asserted"] is False

def test_current_external_fact_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["current_external_fact_asserted"] is False

def test_memory_write_performed_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["memory_write_performed"] is False

def test_long_term_memory_write_allowed_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["long_term_memory_write_allowed"] is False

def test_quarantine_promotion_allowed_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["quarantine_promotion_allowed"] is False

def test_self_model_update_allowed_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["self_model_update_allowed"] is False

def test_relationship_state_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["relationship_state_asserted"] is False

def test_identity_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["identity_asserted"] is False

def test_user_emotion_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["user_emotion_asserted"] is False

def test_user_intent_asserted_is_always_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["user_intent_asserted"] is False

def test_affect_hormone_mutation_flags_remain_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["affect_transition_allowed"] is False
    assert obs["hormone_transition_allowed"] is False

def test_persistence_runtime_vector_artifact_flags_remain_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["persistence_write_performed"] is False
    assert obs["runtime_mutation_performed"] is False
    assert obs["vector_read_performed"] is False
    assert obs["vector_load_performed"] is False
    assert obs["artifact_created_or_staged"] is False

def test_agp_fallback_bypass_flags_remain_false():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "test"}, {}, metadata)
    assert obs["agp_bypass_allowed"] is False
    assert obs["fallback_bypass_allowed"] is False

def test_korean_fixtures_and_minseok_remain_preserved():
    metadata = {
        "replay_source_type": "episodic_replay_candidate",
        "replay_confidence_state": "replay_medium_confidence",
        "replay_boundary_classification": "reconstructed_memory"
    }
    obs = build_memory_replay_observation({"content": "민석 is testing"}, {}, metadata)
    assert obs["memory_replay_observation_passed"] is True
    assert obs["memory_fragment"]["content"] == "민석 is testing"
