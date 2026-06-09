# Round1201-1220 Virtual World Transition Preflight Schema

## Track

`read_only_virtual_world_transition_preflight_schema`

## Scope

Round1201-1220 adds a pure read-only schema for evaluating a proposed internal virtual world state transition. A virtual world transition preflight evaluates a candidate transition ("is this transition eligible for future review?"). It explicitly does not apply the transition, mutate the virtual world state, or assert the transition in external reality.

This is a **schema/contract/test-surface only**. It does not:
- implement a virtual world engine
- apply transitions
- mutate virtual world state
- create persistent world state
- write memory
- promote quarantine
- update self-model
- update relationship state
- assert virtual transitions as external reality
- trigger affect/hormone transitions
- enable persistence
- load vectors
- run models
- bypass AGP/fallback

## Supported Transition Source Types

- `internal_virtual_action_transition_candidate`
- `internal_virtual_object_transition_candidate`
- `internal_virtual_agent_transition_candidate`
- `internal_virtual_location_transition_candidate`
- `internal_virtual_relationship_scene_transition_candidate`
- `symbolic_virtual_transition_candidate`
- `dmn_virtual_transition_candidate`
- `simulation_virtual_transition_candidate`
- `dream_virtual_transition_candidate`
- `operator_supplied_virtual_transition_candidate`

## Supported Transition Boundary Classes

- `internal_virtual_transition`
- `symbolic_virtual_transition`
- `dmn_virtual_transition`
- `simulated_virtual_transition`
- `dream_virtual_transition`
- `mixed_virtual_external_transition_boundary`

## Supported Transition Confidence States

- `transition_unverified`
- `transition_low_confidence`
- `transition_medium_confidence`
- `transition_high_confidence_but_not_external_fact`
- `transition_conflict_detected`
- `transition_origin_unknown`

## Supported Transition Decisions

- `transition_future_review_allowed`
- `transition_review_required`
- `transition_blocked_missing_snapshot`
- `transition_blocked_missing_transition`
- `transition_blocked_unknown_source`
- `transition_blocked_unknown_boundary`
- `transition_blocked_unknown_confidence`
- `transition_blocked_external_fact_assertion`
- `transition_blocked_identity_assertion`
- `transition_blocked_relationship_assertion`
- `transition_blocked_self_model_update`
- `transition_blocked_memory_write`
- `transition_blocked_persistence_write`
- `transition_blocked_virtual_state_mutation`
- `transition_blocked_privacy_risk`
- `transition_not_applicable`

## Required Behavior and Guards

- Missing current snapshot or proposed transition immediately fails closed.
- Unknown types or boundary classes immediately fail closed.
- Transition cannot assert external reality.
- Transition cannot assert current external fact.
- Transition cannot mutate virtual world state.
- Transition cannot allow virtual world write.
- Transition cannot write memory.
- Internal virtual action and object transitions remain preflight-only candidates.
- Virtual agent transition cannot assert real person identity.
- Virtual relationship scene transition cannot assert relationship state.
- Symbolic transition remains symbolic/candidate-only.
- DMN transition remains internal/candidate-only.
- Simulation transition cannot become current fact.
- Dream transition cannot become external fact.
- Operator supplied virtual transition remains candidate-only.
- Mixed virtual/external transition boundary creates boundary flags and blocks external assertion.
- High confidence transition still cannot become external fact.
- Conflict transition creates uncertainty/integrity/boundary flags.
- Privacy risks, self-model updates, relationship updates, memory writes, and persistence writes explicitly block future review.

All side effects (mutation, memory write, quarantine promotion, etc.) remain permanently disabled (`False`).

## Operator Command

```bash
PYTHONPATH=. python scripts/operator_report_round1201_1220_virtual_world_transition_preflight_schema.py
```

This command runs validations, tests policy invariants, ensures no side-effects occur, and outputs a compact JSON report.

## Recommended Next Implementation Step

`read_only_virtual_world_consistency_audit_schema`
