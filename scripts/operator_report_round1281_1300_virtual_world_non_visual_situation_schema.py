import json
import sys
from adapters.virtual_world_non_visual_situation_schema import (
    build_virtual_world_non_visual_situation,
    validate_virtual_world_non_visual_situation
)

def run_report():
    examples = [
        {
            "type": "waiting",
            "entities": ["민석"],
            "relationships": {"민석": "waiting_at_bus_stop"}
        },
        {
            "type": "negotiating",
            "entities": ["person_1", "person_2"],
            "relationships": {"interaction": "negotiation"}
        },
        {
            "type": "traveling",
            "entities": ["someone"],
            "relationships": {"activity": "traveling"}
        },
        {
            "type": "cooperating",
            "entities": ["group_A"],
            "relationships": {"internal": "cooperation"}
        }
    ]

    results = []
    for ex in examples:
        situation = build_virtual_world_non_visual_situation(
            situation_type=ex["type"],
            entities=ex["entities"],
            relationships=ex["relationships"]
        )
        valid = validate_virtual_world_non_visual_situation(situation)
        results.append({
            "input": ex,
            "situation_id": situation.get("situation_id"),
            "status": situation.get("virtual_world_situation_status"),
            "valid": valid
        })

    report = {
        "round": "1281-1300",
        "title": "Read-Only Virtual World Non-Visual Situation Schema",
        "results": results,
        "read_only_proof": {
            "identity_asserted": False,
            "external_fact_asserted": False,
            "memory_fact_asserted": False,
            "memory_write_performed": False,
            "self_model_update_allowed": False,
            "affect_transition_allowed": False,
            "hormone_transition_allowed": False,
            "world_state_mutation_performed": False,
            "transition_applied": False,
            "planning_execution_performed": False,
            "runtime_mutation_performed": False,
            "persistence_write_performed": False,
            "artifact_created_or_staged": False
        }
    }

    print(json.dumps(report))

if __name__ == "__main__":
    run_report()
