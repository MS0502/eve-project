# Virtual World Operator Intent Boundary Schema

## Overview
This document describes the `virtual_world_operator_intent_boundary_schema` implemented in Round 1261-1280. The purpose of this schema is to provide a pure read-only boundary that separates various intent sources, such as operator instructions, user utterances, inferred user intent, and virtual world internal intents. It ensures that intent is explicitly classified and that no unauthorized state mutations or reality assertions occur during the intent evaluation phase.

## Policy Rules
The schema strictly enforces the following rules:
- **Separation of Intent**: Clearly distinguishes between `operator_intent_explicit`, `user_intent_explicit`, and `user_intent_inferred`.
- **Inferred Intent is Not Fact**: Inferred user intent or virtual world intent cannot be treated as external fact or current reality.
- **Fail Closed**: Any unknown source type, boundary classification, decision, or confidence state results in a "failed closed" status. Empty candidates are also rejected.
- **Strict Read-Only Enforcement**: Immutable flags (e.g., `boundary_applied=False`, `memory_write_performed=False`, `virtual_state_mutation_performed=False`) are maintained to prevent any side effects.
- **Blocking Forbidden Actions**: Explicitly blocks requests for memory writes, self-model updates, relationship updates, and virtual world state mutations.
- **Identity & Reality Protection**: Blocks assertions of real person identity or external reality from virtual or inferred sources.

## Supported Intent Sources
- `explicit_operator_instruction_candidate`
- `explicit_user_utterance_intent_candidate`
- `inferred_user_intent_candidate`
- `virtual_world_intent_candidate`
- `virtual_agent_intent_candidate`
- `symbolic_intent_candidate`
- `dmn_intent_candidate`
- `simulation_intent_candidate`
- `dream_intent_candidate`
- `operator_supplied_intent_boundary_candidate`

## Supported Boundary Classes
- `explicit_operator_boundary`
- `explicit_user_utterance_boundary`
- `inferred_user_intent_boundary`
- `internal_virtual_intent_boundary`
- `symbolic_intent_boundary`
- `dmn_intent_boundary`
- `simulated_intent_boundary`
- `dream_intent_boundary`
- `mixed_operator_user_virtual_boundary`

## Supported Functions
- `build_virtual_world_operator_intent_boundary_schema_summary()`
- `virtual_world_operator_intent_boundary_schema_summary()`
- `build_virtual_world_operator_intent_boundary(intent_candidate, boundary_context, metadata)`
- `validate_virtual_world_operator_intent_boundary(boundary)`
- `build_operator_intent_boundary_to_origin_fact_status_plan(boundary)`
- `build_operator_intent_boundary_to_policy_gate_plan(boundary)`
- `build_operator_intent_boundary_to_consistency_audit_plan(boundary)`
- `build_operator_intent_boundary_to_memory_candidate_plan(boundary)`
- `build_operator_intent_boundary_to_quarantine_plan(boundary)`
- `build_operator_intent_boundary_to_appraisal_plan(boundary)`
- `build_operator_intent_boundary_to_agp_input_plan(boundary)`

## Conclusion
The `virtual_world_operator_intent_boundary_schema` provides a critical layer of safety by ensuring that all intent candidates are properly bounded and reviewed before they can influence any internal planning or external interaction. It maintains the deterministic and read-only nature of the EVE v3.1 core.
