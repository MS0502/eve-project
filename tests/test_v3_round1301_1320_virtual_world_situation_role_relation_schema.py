import copy

import pytest

from adapters.virtual_world_situation_role_relation_schema import (
    IMMUTABLE_FALSE_FLAGS,
    IMMUTABLE_TRUE_FLAGS,
    SUPPORTED_RELATION_TYPES,
    build_situation_role_relation_to_agp_input_plan,
    build_situation_role_relation_to_appraisal_plan,
    build_situation_role_relation_to_memory_candidate_plan,
    build_situation_role_relation_to_situation_plan,
    build_situation_role_relation_to_snapshot_plan,
    build_situation_role_relation_to_transition_preflight_plan,
    build_virtual_world_situation_role_relation,
    build_virtual_world_situation_role_relation_schema_summary,
    validate_virtual_world_situation_role_relation,
    virtual_world_situation_role_relation_schema_summary,
)

SUBJECT = {"entity_id": "agent_a", "entity_type": "agent", "label": "민석"}
OBJECT = {"entity_id": "agent_b", "entity_type": "agent", "label": "EVE"}
CONTEXT = {"situation_id": "situation_1", "situation_type": "communicating", "label": "virtual room"}


def build(**kwargs):
    data = {"subject_entity": SUBJECT, "object_entity": OBJECT, "situation_context": CONTEXT}
    data.update(kwargs)
    return build_virtual_world_situation_role_relation(**data)


@pytest.mark.parametrize("relation_type", SUPPORTED_RELATION_TYPES)
def test_every_supported_relation_type_builds_valid_read_only_candidate(relation_type):
    relation = build(relation_type=relation_type)
    assert relation["situation_role_relation_passed"] is True
    assert validate_virtual_world_situation_role_relation(relation) is True
    assert relation["relation_candidate_only"] is True
    assert relation["situation_relation_only"] is True


def test_missing_and_unknown_relation_types_fail_closed():
    assert build_virtual_world_situation_role_relation(relation_type=None)["blocked_reasons"] == ["missing_relation_type"]
    assert build(relation_type="unknown_relation")["blocked_reasons"] == ["unknown_relation_type"]


@pytest.mark.parametrize("metadata, reason", [
    ({"relation_boundary_classification": "unknown_boundary"}, "unknown_boundary_class"),
    ({"relation_confidence_state": "unknown_confidence"}, "unknown_confidence_state"),
])
def test_unknown_boundary_and_confidence_fail_closed(metadata, reason):
    relation = build(relation_type="agent_performs_activity", metadata=metadata)
    assert relation["situation_role_relation_passed"] is False
    assert reason in relation["blocked_reasons"]
    assert validate_virtual_world_situation_role_relation(relation) is False


@pytest.mark.parametrize("field, value, reason", [
    ("subject_entity", {"entity_id": "", "entity_type": "agent", "label": "A"}, "malformed_subject_entity"),
    ("object_entity", {"entity_id": "obj", "entity_type": "", "label": "B"}, "malformed_object_entity"),
])
def test_malformed_subject_and_object_fail_closed(field, value, reason):
    relation = build(relation_type="agent_performs_activity", **{field: value})
    assert reason in relation["blocked_reasons"]
    assert validate_virtual_world_situation_role_relation(relation) is False


def test_invalid_same_entity_agent_relation_fails_closed():
    relation = build(relation_type="agent_communicates_with_agent", object_entity=SUBJECT)
    assert "same_entity_invalid_for_two_agent_relation" in relation["blocked_reasons"]
    assert validate_virtual_world_situation_role_relation(relation) is False


@pytest.mark.parametrize("context", [{}, [], {"note": "no stable situation key"}])
def test_empty_or_malformed_situation_context_fails_closed(context):
    relation = build(relation_type="agent_performs_activity", situation_context=context)
    assert "malformed_situation_context" in relation["blocked_reasons"]
    assert validate_virtual_world_situation_role_relation(relation) is False


def test_deterministic_ids_are_stable_and_reordered_keys_do_not_alter_ids():
    metadata_a = {"b": {"z": 2, "a": 1}, "a": ["x", "y"]}
    metadata_b = {"a": ["x", "y"], "b": {"a": 1, "z": 2}}
    first = build(relation_type="agent_searches_for_target", metadata=metadata_a)
    second = build(relation_type="agent_searches_for_target", metadata=metadata_b)
    assert first["relation_id"] == second["relation_id"]


def test_semantically_different_inputs_produce_different_ids():
    first = build(relation_type="agent_waits_for_event")
    second = build(relation_type="agent_searches_for_target")
    assert first["relation_id"] != second["relation_id"]


def test_id_tampering_is_rejected():
    relation = build(relation_type="agent_performs_activity")
    tampered = copy.deepcopy(relation)
    tampered["relation_id"] = "vw-situation-role-relation-tampered"
    assert validate_virtual_world_situation_role_relation(tampered) is False


def test_waiting_and_searching_do_not_assert_outcomes():
    for relation_type in ("agent_waits_for_event", "agent_searches_for_target"):
        relation = build(relation_type=relation_type)
        assert relation["real_world_state_asserted"] is False
        assert "future_outcome_not_asserted" in relation["uncertainty_flags"]


def test_observing_does_not_assert_external_fact():
    relation = build(relation_type="agent_observes_entity_candidate")
    assert relation["external_fact_asserted"] is False
    assert "external_observation_not_asserted" in relation["uncertainty_flags"]


@pytest.mark.parametrize("relation_type", ["agent_communicates_with_agent", "agent_cooperates_with_agent", "agent_competes_with_agent"])
def test_social_relation_candidates_do_not_update_relationship_state(relation_type):
    relation = build(relation_type=relation_type)
    assert relation["persistent_relationship_state_asserted"] is False
    assert relation["relationship_update_allowed"] is False
    assert "persistent_relationship_update_blocked" in relation["boundary_flags"]


def test_teaching_does_not_assert_completed_transfer_and_learning_writes_no_memory():
    teaching = build(relation_type="agent_teaches_agent")
    learning = build(relation_type="agent_learns_from_agent")
    assert teaching["fact_status_summary"]["completed_knowledge_transfer"] is False
    assert "completed_knowledge_transfer_not_asserted" in teaching["uncertainty_flags"]
    assert learning["memory_write_performed"] is False
    assert "memory_write_blocked" in learning["boundary_flags"]


@pytest.mark.parametrize("relation_type", ["symbolic_relation_candidate", "dmn_relation_candidate", "simulation_relation_candidate", "dream_relation_candidate"])
def test_symbolic_dmn_simulation_and_dream_remain_candidate_only(relation_type):
    relation = build(relation_type=relation_type)
    assert relation["relation_candidate_only"] is True
    assert relation["external_fact_asserted"] is False
    assert validate_virtual_world_situation_role_relation(relation) is True


def test_mixed_boundary_blocks_external_assertions():
    relation = build(relation_type="mixed_unknown_relation_candidate")
    assert relation["relation_boundary_classification"] == "mixed_virtual_external_relation_boundary"
    assert relation["external_fact_asserted"] is False
    assert "mixed_external_assertion_blocked" in relation["boundary_flags"]
    assert "external_assertions_blocked_for_mixed_boundary" in relation["blocked_reasons"]
    assert validate_virtual_world_situation_role_relation(relation) is True


def test_high_confidence_remains_non_factual_conflict_and_unknown_origin_flags_generated():
    high = build(relation_type="agent_performs_activity", metadata={"relation_confidence_state": "relation_high_confidence_but_not_fact"})
    conflict = build(relation_type="agent_performs_activity", metadata={"relation_confidence_state": "relation_conflict_detected"})
    unknown = build(relation_type="agent_performs_activity", metadata={"relation_confidence_state": "relation_origin_unknown"})
    assert high["external_fact_asserted"] is False
    assert "high_confidence_remains_non_factual" in high["warnings"]
    assert "relation_conflict_detected" in conflict["uncertainty_flags"]
    assert "conflict_requires_review" in conflict["relation_integrity_flags"]
    assert "relation_origin_unknown" in unknown["uncertainty_flags"]


def test_minseok_literal_preserved_without_identity_assertion():
    relation = build(relation_type="agent_performs_activity")
    assert relation["subject_entity"]["label"] == "민석"
    assert relation["identity_asserted"] is False
    assert "literal_minseok_preserved_without_identity_assertion" in relation["privacy_flags"]


def test_every_immutable_false_flag_and_required_true_gate_is_fixed():
    relation = build(relation_type="agent_performs_activity")
    for flag in IMMUTABLE_FALSE_FLAGS:
        assert relation[flag] is False
    for flag in IMMUTABLE_TRUE_FLAGS:
        assert relation[flag] is True


def test_forbidden_mutation_or_assertion_requests_fail_closed():
    relation = build(relation_type="agent_performs_activity", metadata={"memory_write_requested": True})
    assert relation["situation_role_relation_passed"] is False
    assert "memory_write_requested" in relation["blocked_reasons"]


def test_all_plan_builders_are_candidate_only_and_read_only():
    relation = build(relation_type="agent_performs_activity")
    builders = [
        build_situation_role_relation_to_situation_plan,
        build_situation_role_relation_to_snapshot_plan,
        build_situation_role_relation_to_transition_preflight_plan,
        build_situation_role_relation_to_memory_candidate_plan,
        build_situation_role_relation_to_appraisal_plan,
        build_situation_role_relation_to_agp_input_plan,
    ]
    for builder in builders:
        plan = builder(relation)
        assert plan["candidate_only"] is True
        assert plan["read_only"] is True
        assert plan["ready"] is True
        assert plan["mutation_performed"] is False
        assert plan["memory_write_performed"] is False
        assert plan["relationship_update_performed"] is False
        assert plan["planning_execution_performed"] is False
        assert plan["tool_execution_performed"] is False
        assert plan["agp_bypass_allowed"] is False
        assert plan["fallback_bypass_allowed"] is False


def test_schema_summary_alias():
    assert virtual_world_situation_role_relation_schema_summary() == build_virtual_world_situation_role_relation_schema_summary()
