# M2-C Bounded Migration and Dual-Read Candidate

## Status

- Pull request: #164
- Baseline: `cb6ddf755aa42d3227b2f2d47cc4215ef3b26a21`
- Authority: comparison-only candidate
- Runtime integration: none
- Human acceptance: not performed
- Legacy authority: retained
- Shadow authority: `shadow_only`

This document records an implementation candidate. It does not approve M2-C, start M2-D, activate dual read, grant recovery authority, or authorize persistence cutover.

## Bounded envelope

The candidate is limited to the single M1-B/C registered state envelope:

```text
legacy target:  ActivationAdapter.learn_pair
stream:         shadow:legacy.activation.learn_pair
state schema:   eve.shadow-projection.activation-learn-pair.v1
state fields:   calls, learned
```

It does not generalize to memory, goals, affect, vectors, models, scheduler state, raw-text capability edges, or any other legacy sidecar.

## Schemas

```text
legacy source declaration:  legacy.activation-learn-pair.snapshot.v1
state evidence:              eve.m2-c-state-evidence.v1
state serialization:         eve.canonical-json-state.v1
legacy evidence:             eve.m2-c-legacy-sidecar-evidence.v1
migration candidate:         eve.m2-c-migration-candidate.v1
dual-read report:            eve.m2-c-dual-read-report.v1
```

## Legacy evidence boundary

The caller supplies all of the following explicitly:

1. a source label;
2. detached source bytes;
3. the declared source schema version;
4. a separately decoded bounded snapshot.

The candidate hashes the source bytes but does not interpret them. It performs no pickle, gzip, archive, filesystem, or arbitrary-object deserialization. It does not discover sidecars or infer a source path. A decoded snapshot is compatible only when it exactly satisfies the accepted `{calls, learned}` projection contract.

State evidence revalidates nested bounded values even when its dataclass is constructed directly. The serialized snapshot must be the exact canonical JSON generated from the accepted projection state; noncanonical JSON, malformed nested entries, digest mismatch, and manifest mismatch fail closed.

This separation prevents migration evidence from becoming a new legacy reader or unsafe deserializer. Actual source acquisition and any future format-specific decoder require a separate reviewed capability and provenance contract.

## Recalculable state evidence

Both legacy and replayed shadow state retain:

- exact canonical state JSON;
- SHA-256 over that canonical JSON;
- a versioned structural manifest;
- canonical byte count;
- exact `calls` and `learned` entry counts;
- key domain and top-level key count;
- serialization schema and projection-state schema;
- manifest SHA-256.

This makes comparison metrics independently recomputable from the report payload and satisfies the v4.2 A10/A11 evidence boundary without requiring authority over the source sidecar.

## Migration candidate

A compatible assessment may produce an immutable content-addressed migration candidate. The candidate requires a positive source-byte count and is fixed to:

```text
authority:                 comparison_only
writes_performed:          false
runtime_integrated:        false
legacy_authority_retained: true
```

It contains no activation flag and cannot initialize a store, append an event, write a snapshot, install an observer, or alter a default.

## Dual-read comparison

The caller must explicitly provide an already initialized M2-A `SQLiteShadowStore` and an explicit initial bounded snapshot. The comparison:

1. records the shadow-store integrity digest before reading;
2. reads only the registered stream;
3. replays twice from independently created initial states through the accepted M1-C reducer;
4. requires identical repeated replay state and digest;
5. compares replay output with detached legacy evidence;
6. reports exact state mismatches and schema/integrity/replay incompatibilities;
7. records the shadow-store integrity digest after reading;
8. computes `state_changed` only from the two integrity digests;
9. records a transition hash over source, legacy state, replay state, findings, and before/after evidence;
10. marks any digest change as `shadow_store_changed_during_comparison`.

`writes_performed` remains `false` even when the store changed concurrently. A before/after difference proves that store state changed during the observation window; it does not prove that comparison-only code performed the write. The digest change therefore blocks a match without falsely granting or attributing write authority.

The report remains comparison evidence only. A matching result does not grant recovery authority or cutover eligibility. A mismatch or incompatibility fails closed and does not mutate either side.

## Focused evidence

`tests/test_v4_m2_c_dual_read.py` covers:

- deterministic source hashing and bounded snapshot normalization;
- unsupported schema, empty source, and malformed snapshot reporting;
- nested-state revalidation on direct evidence construction;
- rejection of noncanonical JSON even with matching digest and manifest;
- positive source-byte enforcement;
- content-addressed comparison-only migration candidates;
- rejection of incompatible migration input;
- matching repeated replay without logical store change;
- exact state mismatch reporting;
- concurrent store-change detection without false write attribution;
- visible uninitialized-store incompatibility without auto-creation;
- fail-closed replay-contract rejection;
- absence of file discovery, unsafe deserialization, store initialization, append, snapshot write, backup, or runtime integration calls.

## Prohibited effects

PR #164 must not:

- import or install the legacy runtime;
- read raw external text or expand M2-B capability authority;
- locate or decode pickle/gzip sidecars;
- initialize or create a SQLite database automatically;
- append events or write snapshots during comparison;
- become authoritative recovery;
- activate dual-read production behavior;
- change persistence defaults;
- transfer authority from legacy persistence;
- start M2-D;
- perform cutover;
- activate scheduler, model, vector, affect, goal, or memory authority.

## Promotion boundary

M2-C remains blocked from acceptance until its final exact head has:

- focused tests green;
- M0-A through M0-D byte identity;
- M2-B exact technical decisions still valid;
- forward gate green with exact same-PR registrations;
- full suite green;
- independently inspectable exact-head artifact;
- separate human review.

PR-body or review-comment-only changes do not require test repetition when the final head, workflow run, and artifact digest remain unchanged.
