from adapters.virtual_world_non_visual_situation_schema import (
    FORBIDDEN_REQUEST_FIELDS,
    READ_ONLY_FALSE_INVARIANTS,
    SUPPORTED_SITUATION_TYPES,
    build_virtual_world_non_visual_situation,
    build_virtual_world_situation_to_memory_plan,
    build_virtual_world_situation_to_snapshot_plan,
    build_virtual_world_situation_to_transition_plan,
    validate_virtual_world_non_visual_situation,
)


def _entities():
    return [
        {"entity_id": "person_1", "entity_type": "person", "label": "민석"},
        {"entity_id": "place_1", "entity_type": "place", "label": "정류장"},
    ]


def _relationships():
    return [
        {
            "source_entity_id": "person_1",
            "target_entity_id": "place_1",
            "relationship_type": "waiting_at",
        }
    ]


def _assert_read_only_invariants(situation):
    for invariant in READ_ONLY_FALSE_INVARIANTS:
        assert situation[invariant] is False
    assert situation["read_only"] is True
    assert situation["snapshot_plan_candidate_only"] is True
    assert situation["transition_plan_candidate_only"] is True
    assert situation["memory_handoff_plan_candidate_only"] is True


def test_build_all_supported_situation_types():
    for situation_type in SUPPORTED_SITUATION_TYPES:
        situation = build_virtual_world_non_visual_situation(
            situation_type=situation_type,
            entities=_entities(),
            relationships=_relationships(),
        )
        assert situation["virtual_world_situation_passed"] is True
        assert situation["situation_type"] == situation_type
        assert validate_virtual_world_non_visual_situation(situation) is True
        _assert_read_only_invariants(situation)


def test_deterministic_content_derived_situation_id_is_stable_for_same_semantic_input():
    first = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=list(reversed(_entities())),
        relationships=_relationships(),
        metadata={"note": "민석 대기"},
    )
    second = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=_entities(),
        relationships=_relationships(),
        metadata={"note": "민석 대기"},
    )
    assert first["situation_id"] == second["situation_id"]
    assert first["situation_id"].startswith("vw-non-visual-")
    assert len(first["situation_id"].removeprefix("vw-non-visual-")) == 64


def test_deterministic_content_derived_situation_id_changes_for_different_semantic_input():
    waiting = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=_entities(),
        relationships=_relationships(),
    )
    traveling = build_virtual_world_non_visual_situation(
        situation_type="traveling",
        entities=_entities(),
        relationships=_relationships(),
    )
    assert waiting["situation_id"] != traveling["situation_id"]


def test_fail_closed_unknown_situation_type():
    situation = build_virtual_world_non_visual_situation(situation_type="unknown_type")
    assert situation["virtual_world_situation_passed"] is False
    assert "unknown_situation_type" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False


def test_fail_closed_missing_situation_type():
    situation = build_virtual_world_non_visual_situation(situation_type=None)
    assert situation["virtual_world_situation_passed"] is False
    assert "missing_situation_type" in situation["blocked_reasons"]
    assert validate_virtual_world_non_visual_situation(situation) is False


def test_fail_closed_malformed_entities():
    malformed_cases = [
        "not_a_list",
        ["민석"],
        [{"entity_id": "person_1", "entity_type": "person"}],
        [
            {"entity_id": "person_1", "entity_type": "person", "label": "민석"},
            {"entity_id": "person_1", "entity_type": "person", "label": "중복"},
        ],
    ]
    for entities in malformed_cases:
        situation = build_virtual_world_non_visual_situation(
            situation_type="waiting",
            entities=entities,
        )
        assert situation["virtual_world_situation_passed"] is False
        assert validate_virtual_world_non_visual_situation(situation) is False


def test_fail_closed_malformed_relationships_and_unknown_references():
    malformed_cases = [
        {"person_1": "waiting_at"},
        ["not_a_dict"],
        [{"source_entity_id": "person_1", "relationship_type": "waiting_at"}],
        [
            {
                "source_entity_id": "person_1",
                "target_entity_id": "missing_place",
                "relationship_type": "waiting_at",
            }
        ],
    ]
    for relationships in malformed_cases:
        situation = build_virtual_world_non_visual_situation(
            situation_type="waiting",
            entities=_entities(),
            relationships=relationships,
        )
        assert situation["virtual_world_situation_passed"] is False
        assert validate_virtual_world_non_visual_situation(situation) is False


def test_korean_entity_preservation():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=_entities(),
        relationships=_relationships(),
    )
    assert situation["virtual_world_situation_passed"] is True
    assert situation["entities"][0]["label"] == "민석"
    assert validate_virtual_world_non_visual_situation(situation) is True


def test_forbidden_requests_fail_closed():
    for forbidden_field in FORBIDDEN_REQUEST_FIELDS:
        situation = build_virtual_world_non_visual_situation(
            situation_type="waiting",
            entities=_entities(),
            relationships=_relationships(),
            metadata={forbidden_field: True},
        )
        assert situation["virtual_world_situation_passed"] is False
        assert forbidden_field in situation["blocked_reasons"]
        assert validate_virtual_world_non_visual_situation(situation) is False


def test_plan_building_functions_are_candidate_only_and_read_only():
    situation = build_virtual_world_non_visual_situation(
        situation_type="negotiating",
        entities=_entities(),
        relationships=_relationships(),
    )

    snap_plan = build_virtual_world_situation_to_snapshot_plan(situation)
    assert snap_plan["plan_type"] == "virtual_world_situation_to_snapshot"
    assert snap_plan["candidate_only"] is True
    assert snap_plan["read_only"] is True
    assert snap_plan["mutation_performed"] is False
    assert snap_plan["ready"] is True

    trans_plan = build_virtual_world_situation_to_transition_plan(situation)
    assert trans_plan["plan_type"] == "virtual_world_situation_to_transition"
    assert trans_plan["candidate_only"] is True
    assert trans_plan["transition_applied"] is False
    assert trans_plan["mutation_performed"] is False
    assert trans_plan["ready"] is True

    mem_plan = build_virtual_world_situation_to_memory_plan(situation)
    assert mem_plan["plan_type"] == "virtual_world_situation_to_memory"
    assert mem_plan["candidate_only"] is True
    assert mem_plan["memory_write_performed"] is False
    assert mem_plan["mutation_performed"] is False
    assert mem_plan["ready"] is True


def test_validate_invariant_violation_and_id_tamper():
    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=_entities(),
        relationships=_relationships(),
    )
    situation["identity_asserted"] = True
    assert validate_virtual_world_non_visual_situation(situation) is False

    situation = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=_entities(),
        relationships=_relationships(),
    )
    situation["situation_id"] = "tampered"
    assert validate_virtual_world_non_visual_situation(situation) is False
