import json
import sys
from adapters.virtual_world_operator_intent_boundary_schema import (
    virtual_world_operator_intent_boundary_schema_summary,
    build_virtual_world_operator_intent_boundary
)

def run_report():
    summary = virtual_world_operator_intent_boundary_schema_summary()

    test_candidate = {
        "candidate_source_type": "explicit_operator_instruction_candidate",
        "instruction": "민석 says: Observe the virtual room.",
        "flags": ["operator_verified"]
    }
    test_context = {
        "boundary_classification": "explicit_operator_boundary",
        "confidence_state": "intent_explicit"
    }

    boundary = build_virtual_world_operator_intent_boundary(test_candidate, test_context)

    report = {
        "report_type": "operator_report_round1261_1280_virtual_world_operator_intent_boundary_schema",
        "schema_summary": summary,
        "sample_boundary": boundary
    }

    print(json.dumps(report))

if __name__ == "__main__":
    run_report()
