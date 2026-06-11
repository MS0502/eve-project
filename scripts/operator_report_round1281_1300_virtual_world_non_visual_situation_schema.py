import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.virtual_world_non_visual_situation_schema import (
    READ_ONLY_FALSE_FLAGS,
    SUPPORTED_SITUATION_TYPES,
    build_virtual_world_non_visual_situation,
    build_virtual_world_situation_to_memory_plan,
    build_virtual_world_situation_to_snapshot_plan,
    build_virtual_world_situation_to_transition_plan,
    validate_virtual_world_non_visual_situation,
)


def run_report():
    sample_situation = build_virtual_world_non_visual_situation(
        situation_type="cooperating",
        entities=[
            {"entity_id": "eve_virtual_actor", "entity_role": "participant", "is_identity_fact": False},
            {"entity_id": "operator_virtual_actor", "entity_role": "observer", "is_identity_fact": False},
        ],
        relationships=[
            {
                "source_entity_id": "eve_virtual_actor",
                "target_entity_id": "operator_virtual_actor",
                "relationship_type": "cooperates_with",
                "relationship_asserted": False,
            }
        ],
        situation_context={"origin": "operator_report_sample", "assertions": [], "requests": []},
        metadata={"round": "1281_1300", "read_only": True},
    )

    report = {
        "report_type": "operator_report_round1281_1300_virtual_world_non_visual_situation_schema",
        "schema_type": "virtual_world_non_visual_situation",
        "supported_situation_types": SUPPORTED_SITUATION_TYPES,
        "always_false_flags": READ_ONLY_FALSE_FLAGS,
        "sample_situation": sample_situation,
        "sample_validates": validate_virtual_world_non_visual_situation(sample_situation),
        "snapshot_plan": build_virtual_world_situation_to_snapshot_plan(sample_situation),
        "transition_plan": build_virtual_world_situation_to_transition_plan(sample_situation),
        "memory_plan": build_virtual_world_situation_to_memory_plan(sample_situation),
        "runtime_mutation_paths_added": False,
        "memory_self_affect_hormone_persistence_mutation_paths_added": False,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    run_report()
