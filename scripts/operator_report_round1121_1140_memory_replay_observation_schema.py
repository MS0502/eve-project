"""Operator report for Round1121-1140 memory replay observation schema."""

import json
import sys

from adapters.memory_replay_observation_schema import (
    memory_replay_observation_schema_summary,
    build_memory_replay_observation,
    build_memory_replay_to_cross_modal_binding_preflight_plan
)

def run() -> None:
    # 1. Summary
    summary = memory_replay_observation_schema_summary()

    # 2. Replay Example (episodic)
    episodic_obs = build_memory_replay_observation(
        memory_fragment={"text": "민석 laughed"},
        metadata={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )

    # 3. Blocked Example (low confidence blocks binding)
    low_conf_obs = build_memory_replay_observation(
        memory_fragment={"text": "maybe I saw a cat"},
        metadata={
            "replay_source_type": "episodic_replay_candidate",
            "replay_confidence_state": "replay_low_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )
    binding_plan = build_memory_replay_to_cross_modal_binding_preflight_plan(low_conf_obs)

    # 4. Failed closed Example (unknown source)
    failed_obs = build_memory_replay_observation(
        memory_fragment={"text": "test"},
        metadata={
            "replay_source_type": "unknown_source",
            "replay_confidence_state": "replay_medium_confidence",
            "replay_boundary_classification": "reconstructed_memory"
        }
    )

    report = {
        "summary": summary,
        "episodic_replay_example_passed": episodic_obs.get("memory_replay_observation_passed"),
        "episodic_replay_origin": episodic_obs.get("origin_summary", {}).get("origin"),
        "episodic_replay_fact_status": episodic_obs.get("fact_status_summary", {}).get("fact_status"),
        "low_confidence_blocks_binding": not binding_plan.get("valid"),
        "unknown_source_fails_closed": not failed_obs.get("memory_replay_observation_passed"),
        "all_assertion_flags_false": not episodic_obs.get("memory_truth_asserted"),
        "all_mutation_flags_false": not episodic_obs.get("memory_write_performed"),
        "korean_fixture_preserved": "민석" in str(episodic_obs.get("memory_fragment")),
        "next_implementation_recommendations": [
            "read_only_memory_provenance_and_quarantine_preflight_schema"
        ]
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run()
