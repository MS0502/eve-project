# M3-C-E Goal-Lifecycle Substrate Binding and Rollback Preflight

## Status

Candidate-only M3-C continuation. This slice is **not** the separate M3-D
milestone and does not open M3-E.

Exact prerequisite:

```text
PR:             #221
exact head:     68b79d3be2d7781f03d30ed782df021409e0e296
exact run:      30424108697
focused:        12 passed
full:           3,258 passed
artifact:       exact-head-validation-68b79d3be2d7781f03d30ed782df021409e0e296
artifact SHA:   c8624fe797901f3cb6afe6af202cd3ab6fcb3f64ac39426ceadabf7bac131f8c
M2-E run:       30424108713
M2-E:           6/6 passed
merge SHA:      6b27629bb576a39d51d14e8350c2fca8bba5961b
```

The prerequisite validation is immutable and reused. Chat, session,
operator-session, PR metadata, review metadata, and Draft/Ready transitions are
not invalidators.

## Purpose

PR #221 established a canonical `candidate_only` lifecycle event candidate and
an immutable in-memory reducer. The next required boundary is to prove that the
candidate can be represented by the accepted v4 `EventEnvelope` contract and
replayed through the accepted event-kernel interface while preserving a
fail-closed rollback path.

This preflight therefore:

1. verifies that the already-authorized v4-native persistence substrate role is
   active and has not entered operational rollback;
2. binds each accepted M3-C-D candidate to a canonical `EventEnvelope`;
3. retains `shadow_only` envelope authority and `candidate_only` binding
   authority;
4. appends only to a newly constructed `InMemoryEventKernel`;
5. proves event-kernel replay equals direct M3-C-D replay;
6. proves a checkpoint prefix can be reconstructed identically after rollback;
7. proves replaying the suffix from that checkpoint reaches the same final
   snapshot as full replay.

## Authority boundary

The active M2-E substrate role is a prerequisite fact, not an append grant for
this goal-lifecycle domain.

```text
v4-native substrate role checked:          authoritative persistence substrate
binding authority:                         candidate_only
bound EventEnvelope authority:             shadow_only
authoritative append authorized:            false
authoritative append performed:             false
SQLite/file write performed:                false
live lifecycle writer installed:            false
production lifecycle integration:           false
action/scheduler/speech authority:           false
legacy goal-domain authority transferred:   false
M3-E authority open:                        false
```

An operationally rolled-back or otherwise inactive v4-native substrate state is
rejected. A technical preflight cannot reactivate it.

## Canonical binding

Each binding retains:

- source event-candidate and transition identity;
- source payload and envelope digests;
- contiguous one-based stream sequence;
- explicit prior-event causation;
- active substrate authority-state digest;
- fixed producer and producer version;
- fixed non-authority flags.

Payload and causal context are revalidated during replay. A changed transition,
source digest, event identity, authority-state digest, producer, stream, or
causal field fails closed.

## Replay and rollback rehearsal

The rehearsal uses disposable in-memory objects only:

1. append all bound envelopes to one isolated `InMemoryEventKernel`;
2. replay from the empty lifecycle snapshot;
3. compare the result with direct M3-C-D reducer replay;
4. append and replay the checkpoint prefix in a fresh kernel;
5. repeat the prefix in another fresh kernel and compare checkpoint digests;
6. replay the suffix from the checkpoint and compare with full replay.

The receipt records only canonical digests, counts, and booleans. It cannot
claim an authoritative append, SQLite write, live-writer installation,
production integration, legacy transfer, or M3-E opening.

## Explicit exclusions

- no import or construction of `SQLiteShadowStore`;
- no file, directory, database, network, clock, process, or thread access;
- no EventKernel or SQLite production hook;
- no default/configuration change;
- no legacy goal-runtime read or write;
- no action execution, scheduling, or speech;
- no goal-domain authority transfer;
- no M3-E affect/goal cutover;
- no phone command and no replay of M3-B sequences 1-5.

## Acceptance condition for this slice

Acceptance means only that canonical binding, isolated event-kernel replay,
checkpoint reconstruction, and checkpoint-resume equivalence are demonstrated
on the exact reviewed head. A later live lifecycle writer still requires a
separate reviewed gate with explicit persistence/write scope, failure handling,
rollback control, and exact-head evidence.
