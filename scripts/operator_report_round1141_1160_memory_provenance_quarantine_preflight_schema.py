import json
import sys
from adapters.memory_provenance_quarantine_preflight_schema import (
    build_memory_provenance_quarantine_preflight_schema_summary,
    build_memory_provenance_quarantine_preflight,
    validate_memory_provenance_quarantine_preflight,
    build_memory_provenance_to_origin_fact_status_plan,
    build_memory_provenance_to_quarantine_plan,
    build_memory_provenance_to_memory_write_plan,
    build_memory_provenance_to_appraisal_plan,
    build_memory_provenance_to_agp_input_plan
)

def main():
    report = {}

    report["schema_summary"] = build_memory_provenance_quarantine_preflight_schema_summary()

    # Valid preflight
    valid_preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"text": "I saw a tree"},
        provenance_context={"source": "camera"},
        metadata={
            "memory_candidate_id": "test_1",
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_partial"
        }
    )
    report["valid_preflight_example"] = valid_preflight
    report["valid_preflight_validation"] = validate_memory_provenance_quarantine_preflight(valid_preflight)

    # Missing provenance blocks future memory write
    missing_provenance_preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"text": "ghost memory"},
        provenance_context={},
        metadata={
            "memory_candidate_id": "test_2",
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_missing"
        }
    )
    report["missing_provenance_preflight"] = missing_provenance_preflight

    # Identity assertion blocked
    identity_assertion_preflight = build_memory_provenance_quarantine_preflight(
        memory_candidate={"text": "I am 민석"},
        provenance_context={"source": "speech"},
        metadata={
            "memory_candidate_id": "test_3",
            "memory_candidate_source_type": "sensory_observation_candidate",
            "provenance_state": "provenance_complete_unverified",
            "identity_assertion": True
        }
    )
    report["identity_assertion_preflight"] = identity_assertion_preflight

    # Plans
    report["plans"] = {
        "origin_fact_status_plan": build_memory_provenance_to_origin_fact_status_plan(valid_preflight),
        "quarantine_plan": build_memory_provenance_to_quarantine_plan(valid_preflight),
        "memory_write_plan": build_memory_provenance_to_memory_write_plan(valid_preflight),
        "appraisal_plan": build_memory_provenance_to_appraisal_plan(valid_preflight),
        "agp_input_plan": build_memory_provenance_to_agp_input_plan(valid_preflight)
    }

    report["status"] = "success"

    print(json.dumps(report, separators=(',', ':')))

if __name__ == "__main__":
    main()
