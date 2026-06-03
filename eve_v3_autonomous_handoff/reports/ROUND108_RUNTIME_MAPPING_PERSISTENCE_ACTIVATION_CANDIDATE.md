# Round108 Runtime Mapping Persistence Activation Candidate

## Status

- Candidate version: `v3_round108_runtime_mapping_persistence_activation_candidate`
- Default behavior: blocked/read-only unless explicit operator approval and `apply_candidate=True` are supplied.
- Runtime mapping default: `false`
- Enforcement default: `false`
- Persistence is **not** enabled by default.

## Prerequisites

Round108 uses these prior artifacts as hard prerequisites:

1. Round106 decision packet: `v3_round106_runtime_mapping_persistence_decision`
2. Round107 activation dry-run harness: `v3_round107_runtime_mapping_persistence_activation_dryrun`

The operator guard fails closed unless the Round106 decision is ready, the Round107 dry-run reports no mutation, and an explicit Round108 approval token is supplied.

## Guarded activation candidate behavior

The activation candidate is intentionally not a normal startup path. When the operator guard is satisfied and `apply_candidate=True` is explicitly requested, the candidate performs this ordered sequence:

1. Export state-debug before candidate mutation.
2. Emit `activation_precheck_started` audit event.
3. Create JSON checkpoint before any candidate mutation.
4. Emit `checkpoint_written` audit event.
5. Apply only the candidate runtime mapping flag mutation (`runtime_mapping_enabled=True`, `enforcement_enabled=False`).
6. Export state-debug after candidate mutation.
7. Emit `activation_candidate_applied` audit event.
8. Roll runtime mapping back to disabled flags.
9. Verify rollback and protected-state invariants.
10. Write rollback JSON and after-rollback state-debug export.
11. Emit `rollback_validation_passed` audit event when verification succeeds.

## Artifact surfaces

The candidate writes JSON/JSONL artifacts only when explicitly applied in an operator-approved run:

- `runtime_mapping_persistence_activation_candidate_checkpoint.json`
- `runtime_mapping_persistence_activation_candidate_rollback.json`
- `runtime_mapping_persistence_activation_candidate_audit_log.jsonl`
- `runtime_mapping_persistence_activation_candidate_state_debug_before.json`
- `runtime_mapping_persistence_activation_candidate_state_debug_after.json`
- `runtime_mapping_persistence_activation_candidate_state_debug_after_rollback.json`

## Mutation boundary

Allowed candidate mutation:

- Ephemeral `LexConceptMappingAdapter.runtime_mapping_enabled=True` only after checkpoint creation and audit emission.
- `LexConceptMappingAdapter.enforcement_enabled` remains `false`.
- The candidate immediately rolls back to disabled flags and verifies rollback.

Forbidden mutation remains forbidden:

- AGP bypass or threshold changes
- Category creation or concept commit mutation
- Concept memory mutation
- SA activation mutation
- EveSpecific vector mutation
- `vectors.npy` commit or fastText seed mutation
- Semantic memory or quarantine mutation
- Test weakening

## State-debug surface

`StateDebugAdapter` now exposes `lex_concept_mapping.runtime_mapping_persistence_activation_candidate` with:

- candidate version
- operator approval requirement
- checkpoint-before-mutation requirement
- rollback verification requirement
- audit-log requirement
- before/after state-debug export requirement
- disabled runtime/enforcement defaults

## Validation summary

Focused and prerequisite tests were run for the Round108 candidate and its Round106/Round107 prerequisites. The status JSON records exact commands and pass states.

## Boundary

Round108 is a guarded activation candidate, not default enablement. Any future production persistence activation must still be an explicit round with operator approval, checkpoint, rollback, audit log, state-debug exports, focused and adjacent validation, and no AGP/vector/category/memory mutation outside the checkpointed/audited boundary.
