import json
import sys
from adapters.virtual_world_state_snapshot_schema import (
    build_virtual_world_state_snapshot_schema_summary,
    build_virtual_world_state_snapshot,
    validate_virtual_world_state_snapshot
)

def main():
    summary = build_virtual_world_state_snapshot_schema_summary()

    # Test valid case
    valid_snapshot = build_virtual_world_state_snapshot(
        snapshot_state={"foo": "bar"},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )

    valid_ok = validate_virtual_world_state_snapshot(valid_snapshot)

    # Test invalid empty
    invalid_snapshot = build_virtual_world_state_snapshot(
        snapshot_state={},
        metadata={
            "snapshot_source_type": "internal_virtual_world_snapshot_candidate",
            "snapshot_boundary_classification": "internal_virtual_state",
            "snapshot_confidence_state": "snapshot_unverified"
        }
    )

    invalid_ok = not validate_virtual_world_state_snapshot(invalid_snapshot)

    report = {
        "report_id": "operator_report_round1181_1200_virtual_world_state_snapshot_schema",
        "schema_summary": summary,
        "valid_case_passed": valid_ok,
        "invalid_case_passed": invalid_ok,
        "all_checks_passed": valid_ok and invalid_ok
    }

    print(json.dumps(report, separators=(',', ':')))

if __name__ == "__main__":
    main()
