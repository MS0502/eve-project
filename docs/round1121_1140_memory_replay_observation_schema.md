# Round1121-1140: Read-Only Memory Replay Observation Schema

## Track

`read_only_memory_replay_observation_schema`

## Scope

Round1121-1140 implements a strictly read-only schema for representing memory replay as an observation candidate. This contract defines how memory-like content retrieved or surfaced internally is structured before being evaluated by downstream systems.

This is a schema/contract/test-surface only. It does not implement memory retrieval, does not write memory, does not promote quarantine, does not update the self-model, and does not assert replayed memory as a current external reality. All assertions, mutations, model loading, and hardware activations remain strictly `False`.

## Supported Replay Source Types

- `episodic_replay_candidate`
- `semantic_replay_candidate`
- `narrative_replay_candidate`
- `self_model_replay_candidate`
- `relationship_replay_candidate`
- `dream_replay_candidate`
- `simulation_replay_candidate`
- `operator_supplied_replay_candidate`

## Supported Replay Confidence States

- `replay_unverified`
- `replay_low_confidence`
- `replay_medium_confidence`
- `replay_high_confidence_but_not_fact`
- `replay_conflict_detected`
- `replay_origin_unknown`

## Supported Replay Boundary Classes

- `reconstructed_memory`
- `symbolic_memory_trace`
- `narrative_memory_trace`
- `simulated_memory_like_content`
- `dream_memory_like_content`
- `mixed_replay_boundary`

## Key Invariants & Safeguards

- Empty `memory_fragment` or unknown type/state/class inputs fail closed.
- A replay observation can only be classified as reconstructed or memory-like content (`reconstructed_memory` = `True`, `replay_only` = `True`).
- It must not assert that the replay happened now or is currently an external fact.
- It must not bypass appraisal, memory gates, quarantine, or origin fact status checks.
- Specific blocks are in place:
  - `self_model` replay cannot update the self-model.
  - `relationship` replay cannot assert relationship state.
  - `dream` replay remains a dream (`dream_dmn` / `symbolic_visualization`).
  - `simulation` replay remains simulated (`simulation` / `simulated_future`).
- Conflicting states create uncertainty flags. Mixed boundaries create boundary flags. Low confidence blocks future cross-modal binding.

## Operator Command

```bash
PYTHONPATH=. python scripts/operator_report_round1121_1140_memory_replay_observation_schema.py
```

The command evaluates the guards, enforces invariants, and outputs a compact JSON proof payload.

## Next Implementation Recommendation

`read_only_memory_provenance_and_quarantine_preflight_schema`
