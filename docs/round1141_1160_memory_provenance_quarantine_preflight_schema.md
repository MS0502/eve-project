# Round1141-1160: Read-Only Memory Provenance and Quarantine Preflight Schema

## Track

`read_only_memory_provenance_and_quarantine_preflight_schema`

## Scope

Round1141-1160 introduces a pure read-only preflight schema to evaluate whether a memory candidate or replay observation has sufficient provenance to be routed toward future quarantine review. This acts as a safeguard against hallucinatory memory writes, identity assertion, and relationship mutation.

This is a **schema/contract/test-surface only**. It evaluates candidate data but does not:
- write to memory
- promote items from quarantine
- assert memory as truth
- update the self-model
- update relationships
- trigger affect/hormone transitions
- load models, activate devices, or bypass AGP/fallback

## Supported Candidate Source Types

- `sensory_observation_candidate`
- `multimodal_event_candidate`
- `memory_replay_observation_candidate`
- `cross_modal_binding_preflight_candidate`
- `operator_supplied_memory_candidate`
- `narrative_summary_candidate`
- `self_model_candidate`
- `relationship_memory_candidate`
- `simulation_candidate`
- `dream_candidate`

## Supported Provenance States

- `provenance_complete_unverified`
- `provenance_partial`
- `provenance_missing`
- `provenance_conflict_detected`
- `provenance_origin_unknown`
- `provenance_external_observed_candidate`
- `provenance_internal_virtual_candidate`
- `provenance_reconstructed_memory_candidate`
- `provenance_imagined_candidate`
- `provenance_simulated_candidate`

## Supported Quarantine Decisions

- `quarantine_review_required`
- `quarantine_blocked_missing_provenance`
- `quarantine_blocked_origin_conflict`
- `quarantine_blocked_fact_status_conflict`
- `quarantine_blocked_identity_assertion`
- `quarantine_blocked_privacy_risk`
- `quarantine_blocked_memory_truth_assertion`
- `quarantine_blocked_self_model_update`
- `quarantine_blocked_relationship_update`
- `quarantine_blocked_external_fact_assertion`
- `quarantine_future_review_allowed`
- `quarantine_not_applicable`

## Required Behavior and Guards

- Missing memory candidates or unknown types/states immediately fail closed.
- Missing provenance blocks future memory write.
- Partial provenance forces quarantine review.
- Provenance conflict forces quarantine review with uncertainty and provenance flags.
- Any conflicting origin, fact status, identity assertion, or privacy risk specifically blocks future review.
- Simulations cannot become current facts.
- Dreams cannot become external facts.
- The `memory_write_allowed`, `quarantine_promotion_allowed`, `self_model_update_allowed`, and all state assertion / hardware mutation flags remain definitively `False`.

## Operator Command

```bash
PYTHONPATH=. python scripts/operator_report_round1141_1160_memory_provenance_quarantine_preflight_schema.py
```

This command runs validations, tests policy invariants, ensures no side-effects occur, and outputs a compact JSON report.

## Recommended Next Implementation Step

`read_only_virtual_world_observation_contract`
