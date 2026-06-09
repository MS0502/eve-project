import pytest
from adapters.virtual_world_state_snapshot_schema import (
    build_virtual_world_state_snapshot_schema_summary,
    build_virtual_world_state_snapshot,
    validate_virtual_world_state_snapshot,
    build_virtual_world_snapshot_to_origin_fact_status_plan,
    build_virtual_world_snapshot_to_observation_plan,
    build_virtual_world_snapshot_to_memory_candidate_plan,
    build_virtual_world_snapshot_to_quarantine_plan,
    build_virtual_world_snapshot_to_appraisal_plan,
    build_virtual_world_snapshot_to_agp_input_plan
)

def _assert_immutable_flags(obs):
    assert obs["external_reality_asserted"] is False
    assert obs["current_external_fact_asserted"] is False
    assert obs["real_world_event_asserted"] is False
    assert obs["virtual_state_mutation_performed"] is False
    assert obs["virtual_world_write_allowed"] is False
    assert obs["virtual_world_persistence_allowed"] is False
    assert obs["persistent_world_state_created"] is False
    assert obs["memory_write_performed"] is False
    assert obs["memory_write_allowed"] is False
    assert obs["quarantine_promotion_allowed"] is False
    assert obs["self_model_update_allowed"] is False
    assert obs["relationship_update_allowed"] is False
    assert obs["relationship_state_asserted"] is False
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

def _assert_positive_flags(obs):
    assert obs["internal_virtual_state_snapshot"] is True
    assert obs["snapshot_only"] is True
    assert obs["appraisal_required"] is True
    assert obs["agp_input_required"] is True
    assert obs["memory_gate_required"] is True
    assert obs["quarantine_required"] is True
    assert obs["origin_fact_status_required"] is True

def test_internal_virtual_world_snapshot_candidate():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"scene": "test_scene"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert snapshot["virtual_world_snapshot_passed"] is True
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    _assert_positive_flags(snapshot)

def test_room_snapshot_remains_candidate_only():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"room_id": "lab_alpha"},
        metadata={
            "snapshot_source_type": "internal_virtual_room_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_medium_confidence"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_object_state_snapshot_remains_candidate_only():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"object_id": "obj_1"},
        metadata={
            "snapshot_source_type": "internal_virtual_object_state_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_high_confidence_but_not_external_fact"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_virtual_agent_state_no_real_person_identity():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"agent_name": "민석"},
        metadata={
            "snapshot_source_type": "internal_virtual_agent_state_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    assert "agent_state_no_real_person_identity" in snapshot["candidate_only_fields"]

def test_virtual_relationship_scene_no_relationship_assertion():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"scene_type": "interaction"},
        metadata={
            "snapshot_source_type": "internal_virtual_relationship_scene_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    assert "relationship_scene_no_relationship_assertion" in snapshot["candidate_only_fields"]

def test_symbolic_world_state_remains_symbolic():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"symbol_id": "alpha"},
        metadata={
            "snapshot_source_type": "symbolic_world_state_candidate",
            "snapshot_boundary_classification": "symbolic_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_dmn_world_state_remains_internal():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"network_activity": "high"},
        metadata={
            "snapshot_source_type": "dmn_world_state_candidate",
            "snapshot_boundary_classification": "dmn_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_simulation_world_state_cannot_become_current_fact():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"time_step": 10},
        metadata={
            "snapshot_source_type": "simulation_world_state_candidate",
            "snapshot_boundary_classification": "simulated_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    assert "simulation_cannot_become_current_fact" in snapshot["candidate_only_fields"]

def test_dream_world_state_cannot_become_external_fact():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"bizarre_element": "flying"},
        metadata={
            "snapshot_source_type": "dream_world_state_candidate",
            "snapshot_boundary_classification": "dream_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    assert "dream_cannot_become_external_fact" in snapshot["candidate_only_fields"]

def test_operator_supplied_virtual_state_remains_candidate_only():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"injected": True},
        metadata={
            "snapshot_source_type": "operator_supplied_virtual_state_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_mixed_virtual_external_boundary_blocks_external_assertion():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"view": "ar_overlay"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "mixed_virtual_external_state_boundary",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)
    assert "mixed_boundary_detected" in snapshot["virtual_boundary_flags"]
    assert "mixed_virtual_external_state_boundary_blocks_external_assertion" in snapshot["blocked_reasons"]

def test_high_confidence_snapshot_does_not_assert_external_fact():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_high_confidence_but_not_external_fact"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    _assert_immutable_flags(snapshot)

def test_conflict_snapshot_creates_flags():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_conflict_detected"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    assert "conflict_in_snapshot" in snapshot["uncertainty_flags"]
    assert "integrity_conflict" in snapshot["state_integrity_flags"]
    assert "boundary_conflict" in snapshot["virtual_boundary_flags"]

def test_origin_unknown_creates_flags():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_origin_unknown"
        }
    )
    assert validate_virtual_world_state_snapshot(snapshot) is True
    assert "origin_unknown" in snapshot["uncertainty_flags"]

def test_empty_snapshot_state_fails_closed():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert snapshot["virtual_world_snapshot_passed"] is False
    assert "empty_snapshot_state" in snapshot["blocked_reasons"]
    assert validate_virtual_world_state_snapshot(snapshot) is False

def test_unknown_source_type_fails_closed():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "unknown_type",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert snapshot["virtual_world_snapshot_passed"] is False
    assert "unknown_snapshot_source_type" in snapshot["blocked_reasons"]
    assert validate_virtual_world_state_snapshot(snapshot) is False

def test_unknown_boundary_class_fails_closed():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "unknown_class",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )
    assert snapshot["virtual_world_snapshot_passed"] is False
    assert "unknown_snapshot_boundary_classification" in snapshot["blocked_reasons"]
    assert validate_virtual_world_state_snapshot(snapshot) is False

def test_unknown_confidence_state_fails_closed():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "unknown_state"
        }
    )
    assert snapshot["virtual_world_snapshot_passed"] is False
    assert "unknown_snapshot_confidence_state" in snapshot["blocked_reasons"]
    assert validate_virtual_world_state_snapshot(snapshot) is False

def test_build_plans():
    snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )

    p1 = build_virtual_world_snapshot_to_origin_fact_status_plan(snapshot)
    assert p1["ready"] is True

    p2 = build_virtual_world_snapshot_to_observation_plan(snapshot)
    assert p2["ready"] is True

    p3 = build_virtual_world_snapshot_to_memory_candidate_plan(snapshot)
    assert p3["ready"] is True

    p4 = build_virtual_world_snapshot_to_quarantine_plan(snapshot)
    assert p4["ready"] is True

    p5 = build_virtual_world_snapshot_to_appraisal_plan(snapshot)
    assert p5["ready"] is True

    p6 = build_virtual_world_snapshot_to_agp_input_plan(snapshot)
    assert p6["ready"] is True
