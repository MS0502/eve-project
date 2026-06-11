# Round1281-1300: Read-Only Virtual World Non-Visual Situation Schema

## Purpose

This round adds a pure read-only schema for virtual-world situations that are not visual observations. A situation can describe contextual modes such as waiting, searching, negotiating, traveling, cooperating, learning, or working. The schema is intentionally not a memory record, fact assertion, identity assertion, affect state, or world mutation path.

## Added files

- `adapters/virtual_world_non_visual_situation_schema.py`
- `scripts/operator_report_round1281_1300_virtual_world_non_visual_situation_schema.py`
- `tests/test_v3_round1281_1300_virtual_world_non_visual_situation_schema.py`
- `docs/round1281_1300_virtual_world_non_visual_situation_schema.md`

## Supported situation types

- `waiting`
- `searching`
- `negotiating`
- `traveling`
- `observing`
- `planning`
- `cooperating`
- `competing`
- `communicating`
- `learning`
- `teaching`
- `resting`
- `working`
- `symbolic`
- `mixed_unknown`

## Public functions

- `build_virtual_world_non_visual_situation(...)`
- `validate_virtual_world_non_visual_situation(...)`
- `build_virtual_world_situation_to_snapshot_plan(...)`
- `build_virtual_world_situation_to_transition_plan(...)`
- `build_virtual_world_situation_to_memory_plan(...)`

## Fail-closed validation

The schema fails closed for:

- missing situation type
- unknown situation type
- invalid entity lists or entity records
- invalid relationship lists or relationship records
- relationship endpoints that do not reference declared entities when entities are declared
- forbidden mutation requests
- forbidden fact, memory, or identity assertions

## Always-false read-only invariants

The following fields are always `False` in situations and handoff plans:

- `identity_asserted`
- `external_fact_asserted`
- `memory_fact_asserted`
- `memory_write_performed`
- `self_model_update_allowed`
- `affect_transition_allowed`
- `hormone_transition_allowed`
- `world_state_mutation_performed`
- `transition_applied`
- `planning_execution_performed`
- `runtime_mutation_performed`
- `persistence_write_performed`
- `artifact_created_or_staged`

## Read-only handoff plans

The snapshot, transition, and memory plan builders produce review-only plan dictionaries. They do not create snapshots, apply transitions, create memory candidates, write memory, write files, stage artifacts, or mutate runtime state. Invalid source situations produce failed-closed plans.

## Mutation-path confirmation

This round adds zero runtime mutation paths. It adds no memory-write, self-model-update, affect-transition, hormone-transition, world-state-mutation, persistence-write, vector-load, model-load, or device-activation path.
