import json
import sys

from adapters.virtual_world_non_visual_situation_schema import (
    build_virtual_world_non_visual_situation,
    validate_virtual_world_non_visual_situation,
)


def _valid_case():
    return build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=[
            {"entity_id": "person_1", "entity_type": "person", "label": "민석"},
            {"entity_id": "place_1", "entity_type": "place", "label": "정류장"},
        ],
        relationships=[
            {
                "source_entity_id": "person_1",
                "target_entity_id": "place_1",
                "relationship_type": "waiting_at",
            }
        ],
        metadata={"note": "민석 대기"},
    )


def run_report():
    valid = _valid_case()
    invalid = build_virtual_world_non_visual_situation(
        situation_type="waiting",
        entities=[{"entity_id": "person_1", "entity_type": "person", "label": "민석"}],
        relationships=[
            {
                "source_entity_id": "person_1",
                "target_entity_id": "missing",
                "relationship_type": "waiting_at",
            }
        ],
    )
    deterministic_a = _valid_case()
    deterministic_b = _valid_case()

    valid_case_passed = validate_virtual_world_non_visual_situation(valid) is True
    invalid_case_passed = (
        invalid.get("virtual_world_situation_passed") is False
        and validate_virtual_world_non_visual_situation(invalid) is False
    )
    deterministic_id_passed = (
        deterministic_a.get("situation_id") == deterministic_b.get("situation_id")
        and deterministic_a.get("situation_id") != invalid.get("situation_id")
    )
    all_checks_passed = valid_case_passed and invalid_case_passed and deterministic_id_passed

    report = {
        "valid_case_passed": valid_case_passed,
        "invalid_case_passed": invalid_case_passed,
        "deterministic_id_passed": deterministic_id_passed,
        "all_checks_passed": all_checks_passed,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(run_report())
