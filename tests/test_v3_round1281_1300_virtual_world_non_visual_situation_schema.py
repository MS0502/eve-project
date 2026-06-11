from adapters.virtual_world_non_visual_situation_schema import (
    READ_ONLY_FALSE_FLAGS,
    SUPPORTED_SITUATION_TYPES,
    build_virtual_world_non_visual_situation,
    build_virtual_world_situation_to_memory_plan,
    build_virtual_world_situation_to_snapshot_plan,
    build_virtual_world_situation_to_transition_plan,
    validate_virtual_world_non_visual_situation,
)


def _assert_always_false_flags(payload):
    for flag in READ_ONLY_FALSE_FLAGS:
        assert payload[flag] is False


def test_supported_situation_types_build_valid_read_only_schemas():
    for situation_type in SUPPORTED_SITUATION_TYPES:
        situation = build_virtual_world_non_visual_situation(
            situation_type=situation_type,
            entities=[{"entity_id": "eve_virtual_actor", "entity_role": "participant"}],
            relationships=[],
            situation_context={"origin": "virtual_world_schema_test"},
        )

        assert situation["virtual_world_non_visual_situation_valid"] is True
        assert situation["virtual_world_non_visual_situation_status"] == "read_only_validated"
        assert situation["situation_type"] == situation_type
        assert validate_virtual_world_non_visual_situation(situation) is True
        _assert_always_false_flags(situation)


def test_missing_situation_type_fails_closed():
    situation = build_virtual_world_non_visual_situation()

    assert situation["virtual_world_non_visual_situation_valid"] is False
    assert situation["virtual_world_non_visual_situation_status"] == "failed_closed"
    assert "missing situation type fails closed" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is True
    _assert_always_false_flags(situation)


def test_unknown_situation_type_fails_closed():
    situation = build_virtual_world_non_visual_situation(situation_type="teleporting_fact")

    assert situation["virtual_world_non_visual_situation_valid"] is False
    assert "unknown situation type fails closed" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is True
    _assert_always_false_flags(situation)


def test_invalid_entities_fail_closed():
    bad_entity_payloads = [
        "not-a-list",
        ["not-a-dict"],
        [{"entity_role": "participant"}],
        [{"entity_id": "eve"}, {"entity_id": "eve"}],
        [{"entity_id": "eve", "entity_role": ""}],
        [{"entity_id": "eve", "is_identity_fact": True}],
    ]

    for entities in bad_entity_payloads:
        situation = build_virtual_world_non_visual_situation(
            situation_type="waiting",
            entities=entities,
        )

        assert situation["virtual_world_non_visual_situation_valid"] is False
        assert validate_virtual_world_non_visual_situation(situation) is False
        _assert_always_false_flags(situation)


def test_invalid_relationships_fail_closed():
    entities = [
        {"entity_id": "eve", "entity_role": "participant"},
        {"entity_id": "operator", "entity_role": "observer"},
    ]
    bad_relationship_payloads = [
        "not-a-list",
        ["not-a-dict"],
        [{"target_entity_id": "operator", "relationship_type": "near"}],
        [{"source_entity_id": "eve", "relationship_type": "near"}],
        [{"source_entity_id": "eve", "target_entity_id": "operator"}],
        [{"source_entity_id": "missing", "target_entity_id": "operator", "relationship_type": "near"}],
        [{"source_entity_id": "eve", "target_entity_id": "missing", "relationship_type": "near"}],
        [{"source_entity_id": "eve", "target_entity_id": "operator", "relationship_type": "near", "relationship_asserted": True}],
    ]

    for relationships in bad_relationship_payloads:
        situation = build_virtual_world_non_visual_situation(
            situation_type="cooperating",
            entities=entities,
            relationships=relationships,
        )

        assert situation["virtual_world_non_visual_situation_valid"] is False
        assert validate_virtual_world_non_visual_situation(situation) is False
        _assert_always_false_flags(situation)


def test_valid_relationships_remain_non_assertive():
    situation = build_virtual_world_non_visual_situation(
        situation_type="communicating",
        entities=[
            {"entity_id": "eve", "entity_role": "speaker", "is_identity_fact": False},
            {"entity_id": "operator", "entity_role": "listener", "is_identity_fact": False},
        ],
        relationships=[
            {
                "source_entity_id": "eve",
                "target_entity_id": "operator",
                "relationship_type": "communicates_with",
                "relationship_asserted": False,
            }
        ],
    )

    assert situation["virtual_world_non_visual_situation_valid"] is True
    assert validate_virtual_world_non_visual_situation(situation) is True
    _assert_always_false_flags(situation)


def test_forbidden_requests_and_assertions_fail_closed_without_mutation_flags():
    situation = build_virtual_world_non_visual_situation(
        situation_type="planning",
        situation_context={
            "requests": [
                "memory_write",
                "self_model_update",
                "affect_transition",
                "hormone_transition",
                "world_state_mutation",
                "transition_apply",
                "planning_execution",
                "persistence_write",
                "artifact_create",
            ],
            "assertions": [
                "identity_assertion",
                "external_fact_assertion",
                "memory_fact_assertion",
            ],
        },
    )

    assert situation["virtual_world_non_visual_situation_valid"] is False
    assert len(situation["blocked_reasons"]) == 12
    _assert_always_false_flags(situation)


def test_stable_id_is_deterministic_for_same_input():
    kwargs = {
        "situation_type": "learning",
        "entities": [{"entity_id": "eve", "entity_role": "learner"}],
        "relationships": [],
        "situation_context": {"topic": "symbolic rehearsal"},
        "metadata": {"round": "1281_1300"},
    }

    first = build_virtual_world_non_visual_situation(**kwargs)
    second = build_virtual_world_non_visual_situation(**kwargs)

    assert first["situation_id"] == second["situation_id"]
    assert first == second


def test_snapshot_transition_and_memory_plans_are_read_only_for_valid_situation():
    situation = build_virtual_world_non_visual_situation(situation_type="working")

    for builder in (
        build_virtual_world_situation_to_snapshot_plan,
        build_virtual_world_situation_to_transition_plan,
        build_virtual_world_situation_to_memory_plan,
    ):
        plan = builder(situation)
        assert plan["plan_valid"] is True
        assert plan["plan_status"] == "read_only_plan_available"
        assert plan["plan_execution_allowed"] is False
        assert plan["plan_execution_performed"] is False
        assert plan["snapshot_write_performed"] is False
        assert plan["snapshot_applied"] is False
        assert plan["memory_candidate_created"] is False
        assert plan["memory_write_allowed"] is False
        _assert_always_false_flags(plan)


def test_plans_fail_closed_for_invalid_situation():
    situation = build_virtual_world_non_visual_situation(situation_type="unknown")

    snapshot_plan = build_virtual_world_situation_to_snapshot_plan(situation)
    transition_plan = build_virtual_world_situation_to_transition_plan(situation)
    memory_plan = build_virtual_world_situation_to_memory_plan(situation)

    for plan in (snapshot_plan, transition_plan, memory_plan):
        assert plan["plan_valid"] is False
        assert plan["plan_status"] == "failed_closed"
        assert plan["blocked_reasons"] == ["source situation is invalid"]
        _assert_always_false_flags(plan)


def test_validator_rejects_any_mutation_flag_flip():
    situation = build_virtual_world_non_visual_situation(situation_type="resting")
    assert validate_virtual_world_non_visual_situation(situation) is True

    for flag in READ_ONLY_FALSE_FLAGS:
        mutated = dict(situation)
        mutated[flag] = True
        assert validate_virtual_world_non_visual_situation(mutated) is False
