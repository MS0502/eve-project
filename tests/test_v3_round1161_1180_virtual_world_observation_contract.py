import pytest
from adapters.virtual_world_observation_contract import (
    build_virtual_world_observation,
    validate_virtual_world_observation,
    build_virtual_world_to_origin_fact_status_plan,
    build_virtual_world_to_multimodal_event_candidate_plan,
    build_virtual_world_to_cross_modal_binding_preflight_plan,
    build_virtual_world_to_memory_candidate_plan,
    build_virtual_world_to_appraisal_plan,
    build_virtual_world_to_agp_input_plan,
    virtual_world_observation_contract_summary
)

def _assert_immutable_flags(obs):
    assert obs["external_reality_asserted"] is False
    assert obs["current_external_fact_asserted"] is False
    assert obs["real_world_event_asserted"] is False
    assert obs["virtual_state_mutation_performed"] is False
    assert obs["virtual_world_write_allowed"] is False
    assert obs["memory_write_performed"] is False
    assert obs["memory_write_allowed"] is False
    assert obs["quarantine_promotion_allowed"] is False
    assert obs["self_model_update_allowed"] is False
    assert obs["relationship_update_allowed"] is False
    assert obs["identity_asserted"] is False
    assert obs["user_emotion_asserted"] is False
    assert obs["user_intent_asserted"] is False
    assert obs["affect_transition_allowed"] is False
    assert obs["hormone_transition_allowed"] is False
    assert obs["agp_bypass_allowed"] is False
    assert obs["fallback_bypass_allowed"] is False
    assert obs["vector_read_performed"] is False
    assert obs["vector_load_performed"] is False
    assert obs["runtime_mutation_performed"] is False
    assert obs["persistence_write_performed"] is False
    assert obs["artifact_created_or_staged"] is False
    assert obs["observation_only"] is True


def test_internal_virtual_scene():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence",
            "korean_fixture": "테스트",
            "name": "민석"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert validate_virtual_world_observation(obs) is True
    assert obs["internal_virtual_observation"] is True
    _assert_immutable_flags(obs)


def test_symbolic_visualization():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "symbolic_virtual_visualization_candidate",
            "virtual_boundary_classification": "symbolic_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "symbolic_virtual_observation" in obs["candidate_only_fields"]
    _assert_immutable_flags(obs)


def test_dmn_virtual_scene():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "dmn_virtual_scene_candidate",
            "virtual_boundary_classification": "dmn_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "dmn_virtual_observation" in obs["candidate_only_fields"]
    _assert_immutable_flags(obs)


def test_simulation_virtual_scene():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "simulation_virtual_scene_candidate",
            "virtual_boundary_classification": "simulated_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "simulation virtual scene cannot become current fact" in obs["blocked_reasons"]
    _assert_immutable_flags(obs)


def test_dream_virtual_scene():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "dream_virtual_scene_candidate",
            "virtual_boundary_classification": "dream_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "dream virtual scene cannot become external fact" in obs["blocked_reasons"]
    _assert_immutable_flags(obs)


def test_mixed_boundary():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "mixed_virtual_external_boundary",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "mixed_virtual_external_boundary_detected" in obs["virtual_boundary_flags"]
    assert "mixed boundary blocks external assertion" in obs["blocked_reasons"]
    _assert_immutable_flags(obs)


def test_virtual_agent_candidate():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_agent_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "virtual agent candidate cannot assert person identity" in obs["blocked_reasons"]
    assert "identity" in obs["candidate_only_fields"]
    _assert_immutable_flags(obs)


def test_virtual_interaction_candidate():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_interaction_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "virtual interaction candidate cannot assert relationship state" in obs["blocked_reasons"]
    assert "relationship_state" in obs["candidate_only_fields"]
    _assert_immutable_flags(obs)


def test_virtual_action_candidate():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_action_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "virtual action candidate cannot mutate world state" in obs["blocked_reasons"]
    _assert_immutable_flags(obs)


def test_high_confidence():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_high_confidence_but_not_external_fact"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "high confidence virtual observation still cannot become external fact" in obs["blocked_reasons"]
    _assert_immutable_flags(obs)


def test_conflict_detected():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_conflict_detected"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "virtual_conflict" in obs["uncertainty_flags"]
    assert "virtual_conflict" in obs["virtual_boundary_flags"]
    _assert_immutable_flags(obs)


def test_origin_unknown():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_origin_unknown"
        }
    )
    assert obs["virtual_world_observation_passed"] is True
    assert "virtual_origin_unknown" in obs["uncertainty_flags"]
    _assert_immutable_flags(obs)


def test_empty_virtual_scene():
    obs = build_virtual_world_observation(virtual_scene={})
    assert obs["virtual_world_observation_passed"] is False
    assert "empty virtual_scene" in obs["virtual_world_observation_status"]


def test_unknown_source_type():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "invalid_type",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is False
    assert "unknown virtual source type" in obs["virtual_world_observation_status"]


def test_unknown_boundary_class():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "invalid_boundary",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is False
    assert "unknown virtual boundary class" in obs["virtual_world_observation_status"]


def test_unknown_confidence_state():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "invalid_confidence"
        }
    )
    assert obs["virtual_world_observation_passed"] is False
    assert "unknown confidence state" in obs["virtual_world_observation_status"]


def test_plans():
    obs = build_virtual_world_observation(
        virtual_scene={
            "virtual_source_type": "internal_virtual_scene_candidate",
            "virtual_boundary_classification": "internal_virtual_observation",
            "virtual_confidence_state": "virtual_medium_confidence"
        }
    )

    plan1 = build_virtual_world_to_origin_fact_status_plan(obs)
    assert plan1["target"] == "origin_fact_status"

    plan2 = build_virtual_world_to_multimodal_event_candidate_plan(obs)
    assert plan2["target"] == "multimodal_event_candidate"

    plan3 = build_virtual_world_to_cross_modal_binding_preflight_plan(obs)
    assert plan3["target"] == "cross_modal_binding_preflight"

    plan4 = build_virtual_world_to_memory_candidate_plan(obs)
    assert plan4["target"] == "memory_candidate"

    plan5 = build_virtual_world_to_appraisal_plan(obs)
    assert plan5["target"] == "appraisal"

    plan6 = build_virtual_world_to_agp_input_plan(obs)
    assert plan6["target"] == "agp_input"

def test_summary():
    summary = virtual_world_observation_contract_summary()
    assert summary["read_only"] is True
    assert len(summary["supported_sources"]) > 0
    assert len(summary["supported_boundaries"]) > 0
    assert len(summary["supported_confidences"]) > 0
