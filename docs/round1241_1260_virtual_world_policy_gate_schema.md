# Virtual World Policy Gate Schema

## Overview
This document describes the `virtual_world_policy_gate_schema` implemented in Round 1241-1260. The purpose of this schema is to evaluate virtual world candidates (such as observations, snapshots, transition preflights, or consistency audits) and determine if they are allowed to proceed to future review stages. It is a strictly read-only gate that does not enact any changes itself.

## Policy Rules
The schema ensures strict compliance with project constraints. Specifically:
- **No Side Effects**: Evaluation does not apply transitions, mutate virtual world state, generate persistent data, or assert any reality outside of the virtual environment boundary.
- **Fail Closed**: Unknown candidate types, policy decisions, risk levels, or boundary classifications cause the policy gate evaluation to fail closed. Empty candidates are also rejected.
- **Block External Assertions**: Mixed virtual/external boundaries immediately block requests asserting external reality. Dream or simulated candidates cannot become current external facts.
- **Strict Privacy & Risk Evaluation**: Identifying privacy flags or a "high" policy risk level definitively blocks candidates from further processing.
- **Strict Read-Only Enforcement**: Immutable read-only flags (e.g., `policy_applied=False`, `virtual_world_write_allowed=False`) are rigorously maintained regardless of the candidate's intent or internal evaluation.

## Supported Functions
The adapter exposes several functions to build the policy gate evaluation and translate its results into corresponding plans (such as origin_fact_status, appraisal, consistency_audit, quarantine, etc.):
- `build_virtual_world_policy_gate_schema_summary()`
- `virtual_world_policy_gate_schema_summary()`
- `build_virtual_world_policy_gate(candidate, policy_context, metadata)`
- `validate_virtual_world_policy_gate(gate)`
- `build_virtual_world_policy_to_origin_fact_status_plan(gate)`
- `build_virtual_world_policy_to_consistency_audit_plan(gate)`
- `build_virtual_world_policy_to_transition_preflight_plan(gate)`
- `build_virtual_world_policy_to_memory_candidate_plan(gate)`
- `build_virtual_world_policy_to_quarantine_plan(gate)`
- `build_virtual_world_policy_to_appraisal_plan(gate)`
- `build_virtual_world_policy_to_agp_input_plan(gate)`

## Conclusion
This component solidifies the safety boundaries around the virtual world schemas by establishing a deterministic, read-only policy gating mechanism preventing unintended modifications or state pollution.