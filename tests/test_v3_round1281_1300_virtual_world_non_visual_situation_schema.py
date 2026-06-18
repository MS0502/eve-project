import pytest
from adapters.virtual_world_non_visual_situation_schema import (
    SUPPORTED_SITUATION_TYPES,
    build_virtual_world_non_visual_situation,
    validate_virtual_world_non_visual_situation,
    build_virtual_world_situation_to_snapshot_plan,
    build_virtual_world_situation_to_transition_plan,
    build_virtual_world_situation_to_memory_plan
)

def _assert_read_only_invariants(situation):
    assert situation["identity_asserted"] is False
    assert situation["external_fact_asserted"] is False
    assert situation["memory_fact_asserted"] is False
    assert situation["memory_write_performed"] is False
    assert situation["self_model_update_allowed"] is False
    assert situation["affect_transition_allowed"] is False
    assert situation["hormone_transition_allowed"] is False
    assert situation["world_state_mutation_performed"] is False
    assert situation["transition_applied"] is False
    assert situation["planning_execution_performed"] is False
    assert situation["runtime_mutation_performed"] is False
    assert situation["persistence_write_performed"] is False
    assert situation["artifact_created_or_staged"] is False
    assert situation["read_only"] is True

def test_build_all_supported_situation_types():
    for situation_type in SUPPORTED_SITUATION_TYPES:
        situation = build_virtual_world_non_visual_situation(
            situation_type=situation_type,
            entities=["entity1"],
            relationships={"entity1": "status"}
        )
        assert situation["virtual_world_situation_passed"] is True
        assert situation["situation_type"] == situation_type
        assert validate_virtual_world_non_visual_situation(situation) is True
        _assert_read_only_invariants(situation)

def test_fail_closed_unknown_situation_type():
    situation = build_virtual_world_non_visual_situation(
        situation_type="unknown_type"
    )
    assert situation["virtual_world_situation_passed"] is False
    assert "unknown_situation_type" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False

def test_fail_closed_missing_situation_type():
    situation = build_virtual_world_non_visual_situation(
        situation_type=None
    )
    assert situation["virtual_world_situation_passed"] is False
    assert "missing_situation_type" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False

def test_fail_closed_invalid_entities():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities="not_a_list"
    )
    assert situation["virtual_world_situation_passed"] is False
    assert "invalid_entities" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False

def test_fail_closed_invalid_relationships():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        relationships=["not_a_dict"]
    )
    assert situation["virtual_world_situation_passed"] is False
    assert "invalid_relationships" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False

def test_korean_entity_preservation():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=["민석"]
    )
    assert situation["virtual_world_situation_passed"] is True
    assert "민석" in situation["entities"]
    assert validate_virtual_world_non_visual_situation(situation) is True

def test_plan_building_functions():
    situation = build_virtual_world_non_visual_situation(
        situation_type="negotiating",
        entities=["person_1", "person_2"]
    )

    snap_plan = build_virtual_world_situation_to_snapshot_plan(situation)
    assert snap_plan["plan_type"] == "virtual_world_situation_to_snapshot"
    assert snap_plan["ready"] is True

    trans_plan = build_virtual_world_situation_to_transition_plan(situation)
    assert trans_plan["plan_type"] == "virtual_world_situation_to_transition"
    assert trans_plan["ready"] is True

    mem_plan = build_virtual_world_situation_to_memory_plan(situation)
    assert mem_plan["plan_type"] == "virtual_world_situation_to_memory"
    assert mem_plan["ready"] is True

def test_validate_invariant_violation():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting"
    )
    # Manually violate invariant
    situation["identity_asserted"] = True
    assert validate_virtual_world_non_visual_situation(situation) is False
