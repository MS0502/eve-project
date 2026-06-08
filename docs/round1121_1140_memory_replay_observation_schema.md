# Round 1121-1140: Memory Replay Observation Schema

## Objective
Add a pure read-only schema for representing memory replay as an observation candidate.
This provides EVE a safe envelope to process retrieved or reconstructed memories as input signals, without immediately believing them to be facts, writing back to memory, or triggering hormone/affect changes.

## Key Invariants
- `current_external_fact_asserted`: false
- `memory_truth_asserted`: false
- `memory_write_performed`: false
- `quarantine_promotion_allowed`: false
- `self_model_update_allowed`: false
- `affect_transition_allowed`: false
- `hormone_transition_allowed`: false
- `persistence_write_performed`: false

## Components
- `adapters/memory_replay_observation_schema.py`
- `tests/test_v3_round1121_1140_memory_replay_observation_schema.py`
- `scripts/operator_report_round1121_1140_memory_replay_observation_schema.py`

## Behaviors
- Empty memory fragment fails closed.
- Invalid replay source or confidence or boundary fails closed.
- Low confidence blocks future binding.
- Conflict creates uncertainty and boundary flags.
- Mixed boundaries creates boundary flags.
- Pure functional mapping, no mutations.
