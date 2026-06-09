import json
from adapters.virtual_world_consistency_audit_schema import (
    build_virtual_world_consistency_audit_schema_summary,
    build_virtual_world_consistency_audit,
    validate_virtual_world_consistency_audit
)

def main():
    summary = build_virtual_world_consistency_audit_schema_summary()

    valid_audit = build_virtual_world_consistency_audit(
        observation_candidate={"obs": "data"},
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )

    valid_ok = validate_virtual_world_consistency_audit(valid_audit)

    invalid_audit = build_virtual_world_consistency_audit(
        observation_candidate=None,
        snapshot_candidate={"snap": "data"},
        metadata={
            "audit_source_type": "observation_snapshot_audit_candidate",
            "audit_boundary_classification": "internal_virtual_consistency_audit"
        }
    )

    invalid_ok = not validate_virtual_world_consistency_audit(invalid_audit)

    report = {
        "report_id": "operator_report_round1221_1240_virtual_world_consistency_audit_schema",
        "schema_summary": summary,
        "valid_case_passed": valid_ok,
        "invalid_case_passed": invalid_ok,
        "all_checks_passed": valid_ok and invalid_ok
    }

    print(json.dumps(report, separators=(',', ':')))

if __name__ == "__main__":
    main()
