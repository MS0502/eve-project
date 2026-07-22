# M2-D Bounded Recovery and Rollback Rehearsal Candidate

## Status

- Pull request: #165
- Baseline: `634a7c5c8e5b8083c34178d967e380903e54107e`
- Authority: `rehearsal_only`
- Runtime integration: none
- Human acceptance: not performed
- Legacy authority: retained
- Shadow authority: `shadow_only`
- Production dual read: disabled
- Authoritative recovery: disabled
- Cutover: not authorized

This document records a disconnected M2-D implementation and evidence candidate. It does not approve M2-D, start M2-E, activate recovery, enable production dual read, change persistence defaults, or transfer authority from the pre-kernel legacy runtime.

## Accepted M2-C prerequisite pin

The observation-window contract is fixed to the accepted M2-C evidence:

```text
PR:              164
accepted head:   3e7e484e91460f5cc46e7bc2e67bac4a5bb51d14
workflow:        29912952289
artifact SHA256: af4f69b75c1316033b695b79362f2968058f31a5e37785f82b5f30626db90c3a
```

Changing any pin invalidates the window instead of silently widening scope.

## Bounded envelope

The rehearsal is limited to the single M1-B/C and M2-C envelope:

```text
legacy target:  ActivationAdapter.learn_pair
stream:         shadow:legacy.activation.learn_pair
state schema:   eve.shadow-projection.activation-learn-pair.v1
state fields:   calls, learned
```

It does not cover memory, goals, affect, vectors, model weights, scheduler state, raw-text capabilities, or any other sidecar or runtime domain.

## Schemas

```text
observation window: eve.m2-d-observation-window.v1
scenario evidence:  eve.m2-d-scenario-evidence.v1
rehearsal packet:   eve.m2-d-rehearsal-packet.v1
```

Each scenario retains canonical raw checks and observations, before/after integrity-report digests, a transition hash, and an evidence digest. The packet binds all six scenario records and the accepted M2-C prerequisite. This makes every machine result independently recalculable from the generated packet.

## Defined observation window

The deterministic window contains:

- two contiguous accepted-stream success envelopes;
- a validated snapshot at sequence 1, leaving one replay suffix event;
- two equivalent valid snapshots so newest-corrupt fallback can be observed;
- an expected final `{calls, learned}` state with canonical JSON, SHA-256, and structural manifest;
- one next-sequence rollback-probe envelope that must change bounded state.

All identifiers and event contents are fixed by `scripts/audit/m2_d_rehearsal.py`. No wall clock, random source, UUID, external service, production path, or legacy sidecar discovery participates.

## Required scenarios

The packet order is fixed and complete:

1. `snapshot_restore`
   - restores from the newest valid snapshot;
   - replays the suffix;
   - repeats restore deterministically;
   - matches the independently pinned final state.
2. `full_replay_equivalence`
   - replays twice from independent initial states;
   - requires identical state and digest;
   - requires snapshot restore to equal full replay.
3. `corrupt_snapshot_fallback`
   - corrupts only a copied disposable database;
   - requires integrity failure visibility;
   - rejects the newest corrupt snapshot;
   - selects the preceding valid snapshot and reaches the same final state.
4. `corrupt_event_fail_closed`
   - corrupts only a copied disposable database;
   - requires event reads and restore to fail closed;
   - verifies failed reads do not further change integrity evidence.
5. `forced_termination`
   - starts an explicit child process against a copied disposable database;
   - begins an uncommitted transaction and exits with `os._exit(97)`;
   - reopens the database and proves the uncommitted event is absent;
   - requires committed history, integrity, and restored state to remain valid.
6. `rollback_rehearsal`
   - appends a bounded probe only to a disposable rollback target;
   - proves the probe changes state;
   - replaces that target with an independently verified M2-A backup;
   - proves the probe is removed and the baseline event/state envelope is restored.

## Execution boundary

`run_recovery_rehearsal` requires a caller-selected path that does not exist. Validation of the window and events occurs before the directory is created. The harness rejects an existing path and never discovers a database automatically.

All writes, corruption injection, forced termination, and file replacement occur only inside that new disposable workspace. Import and dataclass construction perform no I/O. There is no hook from `main.py`, streaming, live/autonomous loops, legacy persistence, lifecycle bridges, or production composition.

## Authority boundary

Every scenario and the final packet permanently fix:

```text
authority:                  rehearsal_only
shadow_authority:           shadow_only
legacy_authority_retained:  true
runtime_integrated:         false
production_dual_read:       false
authoritative_recovery:     false
cutover_authorized:         false
human_review_status:        required_not_performed
human_accepted:             false
```

A complete packet may set only `eligible_for_human_review=true`. It cannot accept itself or authorize M2-E.

## Exact-head evidence integration

`.github/workflows/exact-head-validation.yml` runs the deterministic rehearsal after focused tests and uploads `/tmp/m2-d-rehearsal.json` in the exact-head artifact. The generated JSON contains the complete canonical packet, not only a green verdict.

The existing exact-head gates remain unchanged in authority:

- exact target and clean worktree;
- changed-Python compilation;
- focused tests;
- M0-A through M0-D byte identity;
- M2-B extraction and exact decision validation;
- forward regression gate;
- full-suite collection and execution;
- final clean-worktree verification.

## Focused evidence

`tests/test_v4_m2_d_rehearsal.py` covers:

- complete six-scenario machine packet;
- deterministic identical packet across independent workspaces;
- exact M2-C prerequisite pins;
- recalculable corrupt-snapshot, corrupt-event, forced-termination, and rollback observations;
- inability of packet/scenario dataclasses to self-promote;
- validation failure before workspace creation;
- rejection and non-modification of existing workspaces;
- absence of runtime bridges, default activation, unsafe decoders, clocks, randomness, threads, or general-purpose model surfaces.

## Prohibited effects

PR #165 must not:

- install an observer or lifecycle bridge;
- read or decode a real legacy sidecar;
- discover or open a production database;
- enable production dual read;
- grant recovery authority;
- make the event store authoritative;
- change startup or persistence defaults;
- transfer legacy authority;
- modify scheduler, model, vector, affect, goal, memory, or expression authority;
- start M2-E automatically;
- perform persistence cutover.

## Promotion boundary

M2-D remains Draft until its final exact head has:

- focused tests green;
- deterministic generated rehearsal packet with six of six scenarios passing;
- M0-A through M0-D byte identity;
- M2-B exact technical decisions still valid;
- forward gate green with exact same-PR registrations;
- full suite green;
- independently inspectable exact-head artifact and ZIP digest;
- separate informed human acceptance pinned to that exact head, workflow, and artifact SHA-256.

PR-body, review-comment, label, or Draft/Ready-only changes do not require a new full run while the final head, workflow, artifact digest, and required validation scope remain unchanged. A rerun is allowed only after head change, artifact loss/corruption, digest mismatch, or validation-scope change.
