# Round1221-1240 Virtual World Consistency Audit Schema

## Track

`read_only_virtual_world_consistency_audit_schema`

## Scope

Round1221-1240 adds a pure read-only schema that evaluates consistency between:
- a virtual world observation candidate
- a virtual world state snapshot candidate
- a virtual world transition preflight candidate

The audit validates these candidates and may say:
"consistent for future review,"
"inconsistent,"
"blocked by boundary conflict,"
"blocked by missing candidate,"
or "requires AGP/origin/fact-status review."

It must not say:
"apply this transition,"
"mutate virtual world state,"
"persist this state,"
"this happened in external reality,"
"write this to memory,"
"promote quarantine,"
"update EVE's self-model,"
"update relationship state,"
or "trigger affect/hormone changes."

This is a **schema/contract/test-surface only**. It does not:
- implement a virtual world engine
- apply transitions
- mutate virtual world state
- create persistent world state
- write memory
- promote quarantine
- update self-model
- update relationship state
- assert virtual state as external reality
- trigger affect/hormone transitions
- enable persistence
- load vectors
- run models
- bypass AGP/fallback

## Supported Audit Source Types

- `observation_snapshot_audit_candidate`
- `snapshot_transition_audit_candidate`
- `observation_transition_audit_candidate`
- `full_virtual_world_audit_candidate`
- `symbolic_virtual_audit_candidate`
- `dmn_virtual_audit_candidate`
- `simulation_virtual_audit_candidate`
- `dream_virtual_audit_candidate`
- `operator_supplied_virtual_audit_candidate`

## Supported Audit Boundary Classes

- `internal_virtual_consistency_audit`
- `symbolic_virtual_consistency_audit`
- `dmn_virtual_consistency_audit`
- `simulated_virtual_consistency_audit`
- `dream_virtual_consistency_audit`
- `mixed_virtual_external_consistency_boundary`

## Supported Audit Decisions

- `consistency_future_review_allowed`
- `consistency_review_required`
- `consistency_blocked_missing_observation`
- `consistency_blocked_missing_snapshot`
- `consistency_blocked_missing_transition`
- `consistency_blocked_origin_conflict`
- `consistency_blocked_fact_status_conflict`
- `consistency_blocked_boundary_conflict`
- `consistency_blocked_external_fact_assertion`
- `consistency_blocked_identity_assertion`
- `consistency_blocked_relationship_assertion`
- `consistency_blocked_self_model_update`
- `consistency_blocked_memory_write`
- `consistency_blocked_persistence_write`
- `consistency_blocked_virtual_state_mutation`
- `consistency_blocked_privacy_risk`
- `consistency_not_applicable`

## Required Behavior and Guards

- Missing current snapshot or observation/transition when required by the source immediately fails closed.
- Unknown types or boundary classes immediately fail closed.
- Audit cannot assert external reality.
- Audit cannot assert current external fact.
- Audit cannot mutate virtual world state.
- Audit cannot create persistent world state.
- Audit cannot allow virtual world write.
- Audit cannot apply transition.
- Audit cannot write memory.
- Audit cannot promote quarantine.
- Observation/snapshot/transition mismatch creates consistency flags.
- Origin, fact status, boundary conflict blocks future review.
- External fact, identity, relationship assertion blocks future review.
- Self-model, memory write, persistence write, virtual state mutation requests blocks future review.
- Privacy risk blocks future review.
- Symbolic audit remains symbolic/candidate-only.
- DMN audit remains internal/candidate-only.
- Simulation audit cannot become current fact.
- Dream audit cannot become external fact.
- Operator supplied audit remains candidate-only.

All side effects (mutation, memory write, quarantine promotion, vector load, persistence, etc.) remain permanently disabled (`False`).

## Operator Command

```bash
PYTHONPATH=. python scripts/operator_report_round1221_1240_virtual_world_consistency_audit_schema.py
```

This command runs validations, tests policy invariants, ensures no side-effects occur, and outputs a compact JSON report.

## Recommended Next Implementation Step

`read_only_virtual_world_policy_gate_schema`
