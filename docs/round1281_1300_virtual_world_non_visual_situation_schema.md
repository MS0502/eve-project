# Round 1281-1300: Read-Only Virtual World Non-Visual Situation Schema

## Overview

This round introduces a pure read-only schema representing virtual situations that are not images.
Unlike visual observations, these situations represent abstract states or activities involving entities and their relationships.

## Supported Situation Types

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

## Schema Invariants

The following fields must always be `False` to maintain read-only state and prevent side effects:

- `identity_asserted` = false
- `external_fact_asserted` = false
- `memory_fact_asserted` = false
- `memory_write_performed` = false
- `self_model_update_allowed` = false
- `affect_transition_allowed` = false
- `hormone_transition_allowed` = false
- `world_state_mutation_performed` = false
- `transition_applied` = false
- `planning_execution_performed` = false
- `runtime_mutation_performed` = false
- `persistence_write_performed` = false
- `artifact_created_or_staged` = false

## Required Functions

### `build_virtual_world_non_visual_situation`
Builds a situation payload. Fails closed if the situation type is unknown or missing, or if entities/relationships are invalid.

### `validate_virtual_world_non_visual_situation`
Validates that a situation payload adheres to the schema and invariants.

### Plan Builders
- `build_virtual_world_situation_to_snapshot_plan`
- `build_virtual_world_situation_to_transition_plan`
- `build_virtual_world_situation_to_memory_plan`

## Examples

- 민석 is waiting (type: `waiting`, entities: `['민석']`)
- two people are negotiating (type: `negotiating`, entities: `['person_1', 'person_2']`)
- someone is traveling (type: `traveling`)
- a group is cooperating (type: `cooperating`)
