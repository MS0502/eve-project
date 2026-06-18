# Round 1281-1300: Read-Only Virtual World Non-Visual Situation Schema

## Overview

This round introduces a pure read-only schema representing virtual situations that are not images.
Unlike visual observations, these situations represent abstract states or activities involving entities and their relationships.

This corrective patch removes nondeterministic situation IDs. `situation_id` is now derived from normalized semantic input with canonical JSON and `hashlib.sha256`, so the same semantic input produces the same ID and a different semantic input produces a different ID.

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

## Deterministic ID Rule

`build_virtual_world_non_visual_situation` normalizes the situation type, entity records, relationship records, and metadata into canonical JSON using sorted keys, UTF-8 Korean-preserving encoding, and compact separators. The canonical bytes are hashed with SHA-256 and exposed as:

```text
vw-non-visual-<64 hex sha256 digest>
```

No UUID, random, sampling, time, process, device, model, or runtime state is used to create the ID.

## Structural Validation

Entities must be a list of records with non-empty string fields:

- `entity_id`
- `entity_type`
- `label`

Relationship records must be a list of records with non-empty string fields:

- `source_entity_id`
- `target_entity_id`
- `relationship_type`

Each relationship endpoint must reference an existing `entity_id`. Korean text is preserved, including literal `민석` labels and metadata.

The schema fails closed for malformed entities, malformed relationships, duplicate entity IDs, missing type, and unknown type.

## Schema Invariants

The following fields must always be `False` to maintain read-only state and prevent side effects:

- `identity_asserted` = false
- `external_fact_asserted` = false
- `memory_fact_asserted` = false
- `memory_write_performed` = false
- `self_model_update_allowed` = false
- `relationship_update_allowed` = false
- `affect_transition_allowed` = false
- `hormone_transition_allowed` = false
- `world_state_mutation_performed` = false
- `transition_applied` = false
- `planning_execution_performed` = false
- `runtime_mutation_performed` = false
- `persistence_write_performed` = false
- `tool_execution_performed` = false
- `model_activation_performed` = false
- `device_activation_performed` = false
- `artifact_created_or_staged` = false
- `agp_bypass_allowed` = false
- `fallback_bypass_allowed` = false

Forbidden metadata request flags fail closed when set to `true`: memory write, self-model update, relationship update, affect/hormone transition, world mutation, transition application, planning execution, persistence, tool execution, model/device activation, and artifact creation.

## Candidate-Only Plans

Snapshot, transition, and memory handoff plan builders remain candidate-only and read-only. They do not apply transitions, write memory, persist files, activate tools, or mutate runtime/world state.

## Required Functions

### `build_virtual_world_non_visual_situation`

Builds a situation payload. Fails closed if the situation type is unknown or missing, if entities/relationships are invalid, if relationship references are unknown, or if forbidden mutation/activation requests are present.

### `validate_virtual_world_non_visual_situation`

Validates that a situation payload adheres to the schema, deterministic ID, relationship references, and read-only invariants.

### Plan Builders

- `build_virtual_world_situation_to_snapshot_plan`
- `build_virtual_world_situation_to_transition_plan`
- `build_virtual_world_situation_to_memory_plan`

## Examples

- 민석 is waiting at a bus stop: type `waiting`, entity label `민석`, relationship `waiting_at`
- two people are negotiating: type `negotiating`, person entity records, relationship `negotiating_with`
- someone is traveling: type `traveling`
- a group is cooperating: type `cooperating`
