import pytest
from adapters.memory_replay_observation_schema import (
    build_memory_replay_observation_schema_summary,
    build_memory_replay_observation,
    validate_memory_replay_observation,
    build_memory_replay_to_origin_fact_status_plan,
    build_memory_replay_to_multimodal_event_candidate_plan,
    build_memory_replay_to_cross_modal_binding_preflight_plan,
    build_memory_replay_to_memory_candidate_plan,
    build_memory_replay_to_appraisal_plan,
    build_memory_replay_to_agp_input_plan,
    memory_replay_observation_schema_summary
)

def test_schema_summary():
    summary = build_memory_replay_observation_schema_summary()
    assert summary["read_only"] is True
    assert summary["schema_name"] == "memory_replay_observation_schema"
    assert memory_replay_observation_schema_summary() == summary

def test_empty_memory_fragment_fails_closed():
    obs = build_memory_replay_observation(
        memory_fragment={},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_unverified",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is False
    assert "Empty memory_fragment" in obs["memory_replay_observation_status"]

def test_unknown_replay_source_type_fails_closed():
    obs = build_memory_replay_observation(
        memory_fragment={"k": "v"},
        replay_context={
            "replay_source_type": "invalid",
            "replay_confidence_state": "replay_unverified",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is False

def test_unknown_confidence_state_fails_closed():
    obs = build_memory_replay_observation(
        memory_fragment={"k": "v"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "invalid",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is False

def test_unknown_boundary_class_fails_closed():
    obs = build_memory_replay_observation(
        memory_fragment={"k": "v"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_unverified",
            "replay_boundary_classification": "invalid"
        }
    )
    assert obs["memory_replay_observation_passed"] is False

def test_episodic_replay_builds_reconstructed_memory():
    obs = build_memory_replay_observation(
        memory_fragment={"event": "민석 ate an apple"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert obs["reconstructed_memory"] is True
    assert obs["replay_source_type"] == "episodic_replay_candidate"

def test_semantic_replay_remains_candidate_only():
    obs = build_memory_replay_observation(
        memory_fragment={"fact": "sky is blue"},
        replay_context={
            "replay_source_type": "semantic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "memory_fragment" in obs["candidate_only_fields"]
    assert obs["replay_only"] is True

def test_narrative_replay_remains_candidate_only():
    obs = build_memory_replay_observation(
        memory_fragment={"story": "long time ago..."},
        replay_context={
            "replay_source_type": "narrative_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "narrative_memory_trace"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "memory_fragment" in obs["candidate_only_fields"]
    assert obs["replay_only"] is True

def test_self_model_replay_does_not_update_self_model():
    obs = build_memory_replay_observation(
        memory_fragment={"trait": "brave"},
        replay_context={
            "replay_source_type": "self_model_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert obs["self_model_update_allowed"] is False
    assert obs["replay_source_type"] == "self_model_replay_candidate"

def test_relationship_replay_does_not_assert_relationship_state():
    obs = build_memory_replay_observation(
        memory_fragment={"friend": "민석"},
        replay_context={
            "replay_source_type": "relationship_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert obs["relationship_state_asserted"] is False

def test_dream_replay_cannot_become_external_fact():
    obs = build_memory_replay_observation(
        memory_fragment={"scene": "flying"},
        replay_context={
            "replay_source_type": "dream_replay_candidate",
            "replay_confidence_state": "replay_unverified",
            "replay_boundary_classification": "dream_memory_like_content"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert obs["current_external_fact_asserted"] is False
    assert obs["real_world_event_asserted"] is False

def test_simulation_replay_cannot_become_current_fact():
    obs = build_memory_replay_observation(
        memory_fragment={"sim": "what if"},
        replay_context={
            "replay_source_type": "simulation_replay_candidate",
            "replay_confidence_state": "replay_unverified",
            "replay_boundary_classification": "simulated_memory_like_content"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert obs["current_external_fact_asserted"] is False
    assert obs["real_world_event_asserted"] is False

def test_operator_supplied_replay_remains_candidate_only():
    obs = build_memory_replay_observation(
        memory_fragment={"data": "test"},
        replay_context={
            "replay_source_type": "operator_supplied_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "memory_fragment" in obs["candidate_only_fields"]

def test_low_confidence_replay_blocks_future_binding_plan():
    obs = build_memory_replay_observation(
        memory_fragment={"mem": "blur"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_low_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "low_confidence_blocks_binding" in obs["blocked_reasons"]
    plan = build_memory_replay_to_cross_modal_binding_preflight_plan(obs)
    assert plan["status"] == "blocked"

def test_conflict_replay_creates_uncertainty_flags():
    obs = build_memory_replay_observation(
        memory_fragment={"mem": "clash"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_conflict_detected",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "conflict_detected" in obs["uncertainty_flags"]
    assert "conflict_boundary_crossed" in obs["boundary_flags"]

def test_mixed_replay_boundary_creates_boundary_flags():
    obs = build_memory_replay_observation(
        memory_fragment={"mem": "mix"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "mixed_replay_boundary"
        }
    )
    assert obs["memory_replay_observation_passed"] is True
    assert "mixed_boundary" in obs["boundary_flags"]

def test_all_invariants():
    obs = build_memory_replay_observation(
        memory_fragment={"mem": "test", "name": "민석"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert obs["memory_truth_asserted"] is False
    assert obs["current_external_fact_asserted"] is False
    assert obs["memory_write_performed"] is False
    assert obs["long_term_memory_write_allowed"] is False
    assert obs["quarantine_promotion_allowed"] is False
    assert obs["self_model_update_allowed"] is False
    assert obs["relationship_state_asserted"] is False
    assert obs["identity_asserted"] is False
    assert obs["user_emotion_asserted"] is False
    assert obs["user_intent_asserted"] is False
    assert obs["affect_transition_allowed"] is False
    assert obs["hormone_transition_allowed"] is False
    assert obs["persistence_write_performed"] is False
    assert obs["runtime_mutation_performed"] is False
    assert obs["vector_read_performed"] is False
    assert obs["vector_load_performed"] is False
    assert obs["artifact_created_or_staged"] is False
    assert obs["agp_bypass_allowed"] is False
    assert obs["fallback_bypass_allowed"] is False

def test_plans_valid():
    obs = build_memory_replay_observation(
        memory_fragment={"mem": "test"},
        replay_context={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    assert build_memory_replay_to_origin_fact_status_plan(obs)["status"] == "ready"
    assert build_memory_replay_to_multimodal_event_candidate_plan(obs)["status"] == "ready"
    assert build_memory_replay_to_cross_modal_binding_preflight_plan(obs)["status"] == "ready"
    assert build_memory_replay_to_memory_candidate_plan(obs)["status"] == "ready"
    assert build_memory_replay_to_appraisal_plan(obs)["status"] == "ready"
    assert build_memory_replay_to_agp_input_plan(obs)["status"] == "ready"

def test_invalid_obs_plans():
    assert build_memory_replay_to_origin_fact_status_plan(None) is None
