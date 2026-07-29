# M3-C-F Bounded Goal-Lifecycle Writer Activation-Gate Design

## Status

Documentation-only M3-C continuation. No writer, database, sidecar, operator
command, configuration default, or production integration is added by this
slice. The separate M3-D milestone remains closed. M3-E remains closed.

Exact prerequisite:

```text
PR:             #222
exact head:     844191df9a29d63950ea1ba4f90035261af9418f
exact run:      30438518079
focused:        14 passed
full:           3,272 passed
artifact:       exact-head-validation-844191df9a29d63950ea1ba4f90035261af9418f
artifact SHA:   6e65b9da98ef8ce326187961bceec8e683b4d5ba0cdc9feef4e282ae815429ea
M2-E run:       30438518517
M2-E:           6/6 passed
merge SHA:      938dc3f9d00a8bd7fffe4e4cae38894531462947
```

The exact-head evidence above is immutable and reused. Chat, session,
operator-session, PR metadata, review metadata, and Draft/Ready transitions are
not validation invalidators.

## Purpose

M3-C-A through M3-C-E now provide:

1. a deterministic goal-candidate and selection design;
2. a deterministic selection kernel;
3. a deterministic lifecycle-transition kernel;
4. a canonical candidate-only event envelope and replay reducer;
5. a canonical v4 `EventEnvelope` binding plus isolated in-memory replay,
   rollback-prefix reconstruction, and checkpoint-resume equivalence.

Those results do **not** authorize a persistent lifecycle writer. This design
fixes the exact gate that a later implementation must satisfy before one
bounded M3-C lifecycle stream may write to the accepted v4-native SQLite
substrate.

## Fixed authority facts

```text
M2-E v4-native substrate authority:          active
M3 authority:                                open
legacy runtime authority:                    per-domain until separate migration
legacy goal-domain authority transferred:   false
M3-E authority open:                         false
```

The v4-native substrate may be authoritative for a new M3-C persistence stream
without making legacy `GoalManagement` non-authoritative. Persistence-substrate
authority and legacy-domain behavior authority are separate decisions.

The existing SQLite store accepts only canonical `shadow_only` envelopes. A
future bounded writer therefore keeps the envelope authority field
`shadow_only` while the separately verified M2-E authority state establishes
that the store is the authoritative persistence substrate for this v4-native
stream. Changing `EventEnvelope.authority` is forbidden.

## Versioned gate identities

```text
eve.m3-c-f.goal-lifecycle-writer-activation-design.v1
eve.m3-c-f.goal-lifecycle-writer-authorization.v1
eve.m3-c-f.goal-lifecycle-writer-preflight-receipt.v1
eve.m3-c-f.goal-lifecycle-writer-rollback-control.v1
```

## Exact stream contract

| Item | Fixed value |
|---|---|
| stream id | `m3c.goal_lifecycle` |
| event type | `m3c.goal_lifecycle_transition` |
| producer | `m3c.goal-lifecycle-binding` |
| producer version | `v1` |
| envelope authority | `shadow_only` |
| binding authority prerequisite | `candidate_only` from M3-C-E |
| sequence | one-based, contiguous, no gaps |
| causation | prior bound lifecycle event except sequence 1 |
| payload | byte-equivalent canonical M3-C-E bound payload |
| causal context | exact M3-C-E authority/binding context |

A writer must reject any event whose source transition, source envelope digest,
binding digest, authority-state digest, stream, event type, producer, sequence,
causation, payload, or causal context differs from the accepted M3-C-E
contract.

## Required activation packet

A future writer cannot activate from a boolean default. It requires one
immutable reviewed packet containing all of the following:

1. exact implementation head SHA;
2. exact M3-C-E prerequisite merge SHA and validation artifact SHA-256;
3. exact writer schema and policy versions;
4. exact event stream/type/producer constants;
5. explicit storage policy limits;
6. exact database-path ownership rule;
7. rollback control schema and operator procedure;
8. focused/full/forward/M2-E validation pins;
9. explicit `legacy_goal_authority_transferred=false`;
10. explicit `m3_e_authority_open=false`;
11. explicit human-reviewed authorization for the bounded writer only;
12. explicit exclusions for action, scheduling, speech, and legacy migration.

Missing, malformed, mismatched, or unreviewed packet fields fail closed. No
environment variable, import side effect, file presence, or runtime heuristic
may substitute for this packet.

## Database-path and initialization rule

A later rehearsal or activation must receive an explicit concrete path from its
reviewed caller. It must not:

- invent a default production path;
- use `:memory:` as persistence evidence;
- reuse a legacy goal database;
- scan the filesystem for a database;
- initialize on module import;
- create a writer from a read path automatically.

The first implementation after this design is restricted to a disposable
temporary-directory SQLite file created by the focused test or rehearsal
harness. A production path remains out of scope until a later explicit gate.

## Single-append transaction protocol

The first bounded writer implementation must process exactly one accepted
M3-C-E binding per call. The order is fixed:

1. resolve and verify active M2-E authority;
2. verify the immutable writer-authorization packet;
3. verify the M3-C-E binding and reconstruct its canonical `EventEnvelope`;
4. run store schema and integrity checks;
5. confirm expected stream head and exact next sequence;
6. begin one SQLite immediate transaction;
7. append one canonical envelope;
8. read back the inserted row before commit;
9. verify envelope digest, event id, sequence, chain digests, and byte count;
10. commit only after all checks pass;
11. read back the stream after commit;
12. replay with the M3-C-D reducer and compare the resulting lifecycle snapshot;
13. emit one immutable append receipt.

Any failure rolls back the transaction and emits no success receipt. Batch
append, retry loops, silent repair, sequence skipping, duplicate acceptance,
and partial success are forbidden in the first writer.

## Duplicate and conflict rule

The store's append-only contract already rejects duplicate `event_id`, duplicate
`(stream_id, sequence)`, unknown causation, and non-contiguous sequence. The
bounded writer must preserve those failures exactly.

A repeated request is not idempotent success. It is a deterministic conflict
unless a separate read-only caller first proves that the already-persisted
canonical envelope digest is exactly the requested digest and returns a
read-only `already_present` observation. That observation cannot claim a new
append or new transition.

## Readback and replay acceptance

A successful append receipt is valid only when all are true:

```text
transaction committed:                       true
inserted rows:                                1
readback envelope equals requested envelope: true
stream sequence advanced by exactly one:     true
chain digest advanced and verifies:          true
direct reducer snapshot == SQLite replay:    true
legacy goal authority transferred:           false
M3-E authority open:                         false
action/scheduler/speech authorized:           false
```

The writer may persist named lifecycle transitions only. It may not persist
continuous drive values, continuous scores, polling observations, or unchanged
evaluations.

## Snapshot rule

The writer does not invent a new snapshot system. It uses the existing SQLite
snapshot contract after the configured bounded interval. A snapshot must pin:

- stream id and through-sequence;
- through-event id and envelope digest;
- lifecycle reducer state schema;
- canonical reducer snapshot and digest;
- exact replay policy/version;
- event-chain digest at the snapshot boundary.

Full replay from genesis and snapshot-plus-suffix replay must produce the same
lifecycle snapshot digest.

## Rollback model

Rollback has two layers and neither deletes immutable events.

### Operational writer disable

A private reviewed rollback-control record disables new M3-C lifecycle appends.
It does not revoke the historical M2-E human authorization, delete the SQLite
history, transfer authority to legacy, or open M3-E.

### State recovery

Before any activation rehearsal, the harness must create a verified backup or
checkpoint. Recovery must:

1. disable the writer;
2. preserve the failed database as evidence;
3. restore into a separate path;
4. verify schema, migration history, event chain, and snapshots;
5. replay the lifecycle stream;
6. compare the restored snapshot digest with the pre-activation checkpoint;
7. keep the writer disabled until a new reviewed activation packet exists.

Rollback success is a verified state, not merely a command exit code.

## Failure matrix

| Failure | Required result |
|---|---|
| inactive/rolled-back M2-E authority | refuse before opening store |
| missing or mismatched authorization packet | refuse before opening store |
| malformed/tampered M3-C-E binding | refuse before transaction |
| schema or migration mismatch | refuse append, preserve database |
| duplicate event id or stream sequence | rollback transaction, conflict receipt |
| unknown causation | rollback transaction, conflict receipt |
| storage-policy limit exceeded | rollback transaction, bounded-capacity failure |
| pre-commit readback mismatch | rollback transaction, corruption failure |
| post-commit replay mismatch | disable writer, preserve database, require recovery |
| snapshot mismatch | disable writer, preserve database, require recovery |
| rollback verification mismatch | remain disabled; no reactivation |

## Observation and promotion ladder

A future writer must advance through separate reviewed slices:

1. **M3-C-G disposable SQLite rehearsal** — temporary path only, synthetic
   lifecycle chain, append/readback/replay/snapshot/rollback proof; no production
   hook.
2. **M3-C-H dormant writer integration** — production code path may exist but is
   unreachable by default and requires an absent authorization packet.
3. **M3-C-I bounded activation candidate** — exact-head artifact and explicit
   human review; still no legacy goal-domain authority transfer.
4. **M3-C-J observation window** — bounded lifecycle events, zero duplicate or
   replay divergence, verified rollback preservation.
5. A later separately named goal-domain migration gate may decide whether any
   legacy behavior authority changes. It is not implied by M3-C-G through J.

Each slice receives its own exact-head validation. Acceptance of one slice is
not authorization for the next.

## Explicit exclusions

This design does not authorize or implement:

- a SQLite writer;
- a production database path;
- runtime startup integration;
- automatic activation;
- batch append or retry;
- action execution;
- scheduling;
- speech generation;
- memory or affect mutation;
- legacy goal-domain migration or deletion;
- M3-E affect/goal cutover;
- phone witness or M3-B retention command replay.

## Acceptance ruling

M3-C-F is accepted only as the fixed design boundary for a later disposable
SQLite rehearsal. The immediate implementation target after acceptance is
**M3-C-G disposable SQLite rehearsal** and nothing broader.
