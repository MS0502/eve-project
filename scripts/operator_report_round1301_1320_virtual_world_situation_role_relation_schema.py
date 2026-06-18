import json
import sys

from adapters.virtual_world_situation_role_relation_schema import (
    build_virtual_world_situation_role_relation,
    build_virtual_world_situation_role_relation_schema_summary,
    validate_virtual_world_situation_role_relation,
)


def main():
    subject = {"entity_id": "agent_a", "entity_type": "agent", "label": "민석"}
    obj = {"entity_id": "agent_b", "entity_type": "agent", "label": "EVE"}
    context = {"situation_id": "situation_1", "situation_type": "communicating", "label": "virtual room"}
    valid = build_virtual_world_situation_role_relation(
        relation_type="agent_communicates_with_agent",
        subject_entity=subject,
        object_entity=obj,
        situation_context=context,
    )
    invalid = build_virtual_world_situation_role_relation(
        relation_type="unknown_relation",
        subject_entity=subject,
        object_entity=obj,
        situation_context=context,
    )
    first = build_virtual_world_situation_role_relation(
        relation_type="agent_searches_for_target",
        subject_entity=subject,
        object_entity=obj,
        situation_context=context,
        metadata={"b": {"z": 2, "a": 1}, "a": ["x", "y"]},
    )
    second = build_virtual_world_situation_role_relation(
        relation_type="agent_searches_for_target",
        subject_entity={"label": "민석", "entity_type": "agent", "entity_id": "agent_a"},
        object_entity={"label": "EVE", "entity_id": "agent_b", "entity_type": "agent"},
        situation_context={"label": "virtual room", "situation_type": "communicating", "situation_id": "situation_1"},
        metadata={"a": ["x", "y"], "b": {"a": 1, "z": 2}},
    )
    tampered = dict(valid)
    tampered["relation_id"] = "tampered"
    result = {
        "valid_case_passed": valid["situation_role_relation_passed"] is True and validate_virtual_world_situation_role_relation(valid),
        "invalid_case_passed": invalid["situation_role_relation_passed"] is False and "unknown_relation_type" in invalid["blocked_reasons"],
        "deterministic_id_passed": first["relation_id"] == second["relation_id"] and validate_virtual_world_situation_role_relation(first),
        "tamper_detection_passed": validate_virtual_world_situation_role_relation(tampered) is False,
        "schema_summary": build_virtual_world_situation_role_relation_schema_summary(),
    }
    result["all_checks_passed"] = all(
        result[key]
        for key in (
            "valid_case_passed",
            "invalid_case_passed",
            "deterministic_id_passed",
            "tamper_detection_passed",
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
