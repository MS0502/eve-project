import json
from adapters.virtual_world_transition_preflight_schema import (
    build_virtual_world_transition_preflight_schema_summary,
    build_virtual_world_transition_preflight,
    validate_virtual_world_transition_preflight
)

def main():
    summary = build_virtual_world_transition_preflight_schema_summary()

    valid_preflight = build_virtual_world_transition_preflight(
        current_snapshot={"foo": "bar"},
        proposed_transition={"foo": "baz"},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )

    valid_ok = validate_virtual_world_transition_preflight(valid_preflight)

    invalid_preflight = build_virtual_world_transition_preflight(
        current_snapshot={},
        proposed_transition={"foo": "baz"},
        metadata={
            "transition_source_type": "internal_virtual_action_transition_candidate",
            "transition_boundary_classification": "internal_virtual_transition",
            "transition_confidence_state": "transition_unverified"
        }
    )

    invalid_ok = not validate_virtual_world_transition_preflight(invalid_preflight)

    report = {
        "report_id": "operator_report_round1201_1220_virtual_world_transition_preflight_schema",
        "schema_summary": summary,
        "valid_case_passed": valid_ok,
        "invalid_case_passed": invalid_ok,
        "all_checks_passed": valid_ok and invalid_ok
    }

    print(json.dumps(report, separators=(',', ':')))

if __name__ == "__main__":
    main()
