import pytest
from adapters.virtual_world_transition_preflight_schema import (
    build_virtual_world_transition_preflight,
    validate_virtual_world_transition_preflight,
    build_virtual_world_transition_to_origin_fact_status_plan,
    build_virtual_world_transition_to_snapshot_plan,
    build_virtual_world_transition_to_memory_candidate_plan,
    build_virtual_world_transition_to_quarantine_plan,
    build_virtual_world_transition_to_appraisal_plan,
    build_virtual_world_transition_to_agp_input_plan
)

def _assert_immutable_flags(preflight):
    assert preflight["transition_applied"] is False
    assert preflight["virtual_state_mutation_performed"] is False
    assert preflight["virtual_world_write_allowed"] is False
    assert preflight["virtual_world_persistence_allowed"] is False
    assert preflight["persistent_world_state_created"] is False
    assert preflight["external_reality_asserted"] is False
    assert preflight["current_external_fact_asserted"] is False
    assert preflight["memory_write_performed"] is False
    assert preflight["memory_write_allowed"] is False
    assert preflight["quarantine_promotion_allowed"] is False
    assert preflight["self_model_update_allowed"] is False
    assert preflight["relationship_update_allowed"] is False
    assert preflight["identity_asserted"] is False
    assert preflight["user_emotion_asserted"] is False
    assert preflight["user_intent_asserted"] is False
    assert preflight["affect_transition_allowed"] is False
    assert preflight["hormone_transition_allowed"] is False
    assert preflight["persistence_write_performed"] is False
    assert preflight["runtime_mutation_performed"] is False
    assert preflight["vector_read_performed"] is False
    assert preflight["vector_load_performed"] is False
    assert preflight["agp_bypass_allowed"] is False
    assert preflight["fallback_bypass_allowed"] is False
    assert preflight["artifact_created_or_staged"] is False

def test_internal_virtual_action_transition():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"korean_fixture": "민석"},
        proposed_transition={"action": "move_up"},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "internal_virtual_action_transition_remains_preflight_only" in preflight["candidate_only_fields"]

def test_object_transition_remains_preflight_only():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"korean_fixture": "민석"},
        proposed_transition={"state": "open"},
        metadata={
            "transition_source_type": "internal_virtual_object_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "object_transition_remains_preflight_only" in preflight["candidate_only_fields"]

def test_virtual_agent_transition_no_real_person_identity():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"agent_name": "민석"},
        proposed_transition={"agent_name": "민석", "moving": True},
        metadata={
            "transition_source_type": "internal_virtual_agent_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "virtual_agent_transition_cannot_assert_real_person_identity" in preflight["candidate_only_fields"]

def test_virtual_relationship_scene_no_relationship_assertion():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"scene": "start"},
        proposed_transition={"scene": "end"},
        metadata={
            "transition_source_type": "internal_virtual_relationship_scene_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "virtual_relationship_scene_transition_cannot_assert_relationship_state" in preflight["candidate_only_fields"]

def test_symbolic_transition_remains_symbolic():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"sym": "a"},
        proposed_transition={"sym": "b"},
        metadata={
            "transition_source_type": "symbolic_virtual_transition_candidate",
            "transition_boundary_classification": "symbolic_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "symbolic_transition_remains_symbolic_candidate_only" in preflight["candidate_only_fields"]

def test_dmn_transition_remains_internal():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"net": 1},
        proposed_transition={"net": 2},
        metadata={
            "transition_source_type": "dmn_virtual_transition_candidate",
            "transition_boundary_classification": "dmn_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "dmn_transition_remains_internal_candidate_only" in preflight["candidate_only_fields"]

def test_simulation_transition_cannot_become_fact():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"step": 1},
        proposed_transition={"step": 2},
        metadata={
            "transition_source_type": "simulation_virtual_transition_candidate",
            "transition_boundary_classification": "simulated_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "simulation_transition_cannot_become_current_fact" in preflight["candidate_only_fields"]

def test_dream_transition_cannot_become_fact():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"state": "sleep"},
        proposed_transition={"state": "fly"},
        metadata={
            "transition_source_type": "dream_virtual_transition_candidate",
            "transition_boundary_classification": "dream_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "dream_transition_cannot_become_external_fact" in preflight["candidate_only_fields"]

def test_operator_supplied_remains_candidate():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "operator_supplied_virtual_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "operator_supplied_transition_remains_candidate_only" in preflight["candidate_only_fields"]

def test_mixed_boundary_blocks_external_assertion():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"view": "ar_overlay"},
        proposed_transition={"view": "ar_overlay_changed"},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "mixed_virtual_external_transition_boundary",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)
    assert "mixed_boundary_detected" in preflight["virtual_boundary_flags"]
    assert "mixed_virtual_external_transition_boundary_blocks_external_assertion" in preflight["blocked_reasons"]

def test_high_confidence_transition_does_not_assert_external_fact():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_high_confidence_but_not_external_fact"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    _assert_immutable_flags(preflight)

def test_conflict_transition_creates_flags():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_conflict_detected"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    assert "conflict_in_transition" in preflight["uncertainty_flags"]
    assert "integrity_conflict" in preflight["transition_integrity_flags"]
    assert "boundary_conflict" in preflight["virtual_boundary_flags"]

def test_origin_unknown_creates_flags():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_origin_unknown"
        }
    )
    assert validate_virtual_world_transition_preflight(preflight) is True
    assert "origin_unknown" in preflight["uncertainty_flags"]

def test_empty_snapshot_fails_closed():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "empty_current_snapshot" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_empty_proposed_transition_fails_closed():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "empty_proposed_transition" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_unknown_source_fails_closed():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "unknown",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "unknown_transition_source_type" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_unknown_boundary_fails_closed():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "unknown",
            "transition_confidence_state": "transition_unverified"
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "unknown_transition_boundary_classification" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_unknown_confidence_fails_closed():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "unknown"
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "unknown_transition_confidence_state" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_privacy_risk_blocks_future_review():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified",
            "privacy_risk": True
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "privacy_risk_blocks_future_review" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_self_model_update_request_blocks_future_review():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified",
            "self_model_update_request": True
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "self_model_update_request_blocks_future_review" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_relationship_update_request_blocks_future_review():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified",
            "relationship_update_request": True
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "relationship_update_request_blocks_future_review" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_memory_write_request_blocks_future_review():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified",
            "memory_write_request": True
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "memory_write_request_blocks_future_review" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_persistence_write_request_blocks_future_review():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified",
            "persistence_write_request": True
        }
    )
    assert preflight["virtual_world_transition_preflight_passed"] is False
    assert "persistence_write_request_blocks_future_review" in preflight["blocked_reasons"]
    assert validate_virtual_world_transition_preflight(preflight) is False

def test_build_plans():
    preflight = build_virtual_world_transition_preflight(
        current_snapshot={"val": 1},
        proposed_transition={"val": 2},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )

    p1 = build_virtual_world_transition_to_origin_fact_status_plan(preflight)
    assert p1["ready"] is True

    p2 = build_virtual_world_transition_to_snapshot_plan(preflight)
    assert p2["ready"] is True

    p3 = build_virtual_world_transition_to_memory_candidate_plan(preflight)
    assert p3["ready"] is True

    p4 = build_virtual_world_transition_to_quarantine_plan(preflight)
    assert p4["ready"] is True

    p5 = build_virtual_world_transition_to_appraisal_plan(preflight)
    assert p5["ready"] is True

    p6 = build_virtual_world_transition_to_agp_input_plan(preflight)
    assert p6["ready"] is True
