# EVE v4 Implementation Status

Active constitution: **EVE v4.1**
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**
Runtime status: **pre-kernel legacy runtime remains authoritative; M1-A/M1-B/M1-C/M1-D are shadow-only or declaration-only and not production-integrated**
Previous v3/v3.1 documents: historical reference only
M0 status: **closed**
Forward-gate status: **implemented and enforced by exact-head validation**
M1-A status: **completed by PR #146**
M1-B status: **completed by PR #147**
M1-C status: **completed by PR #148**
M1-D status: **completed by the merge carrying this STATUS update**
Current next step: **M1-E — bounded shadow-observation window and human-reviewed M1 acceptance evidence**
Frozen work: open implementation PRs #109, #86, #84, #82, #11, #7, and #4
Constitution merge baseline: `8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b`
Forward-gate merge baseline: `1ed1093cfec05b44848ad0d117e45885a5669b69`
M1-A merge baseline: `1a3da9aee41c0bed065bb0bbbcc2e8e577aa50f9`
M1-B merge baseline: `15e993780d4c2744047237f877f5add1f7f66339`
M1-C merge baseline: `2546548a4bf757d0fc7b915be1dac7749e7c9824`

## Current state

EVE v4.1 is the active constitutional authority. The existing application remains the **pre-kernel legacy runtime** and retains all current runtime and persistence authority.

M1-A provides an immutable canonical `shadow_only` event envelope, append-only in-memory kernel, and explicit reducer boundary. M1-B provides a separately invoked after-the-fact observer for one registered `ActivationAdapter.learn_pair` legacy funnel. M1-C provides a versioned immutable shadow projection, deterministic bounded reducer/replay, explicit equivalence reports, and immutable in-memory checkpoint/rollback values for that same single stream. M1-D provides immutable lifecycle-owner, disconnected bridge-registry, reviewed source/disposition, and redacted failure-propagation declarations for bounded chat, activity, memory, and goal domains.

None of M1-A through M1-D is connected to `main.py`, `language/streaming.py`, live/autonomous loops, production composition, persistence adapters, or default startup paths. M1-D names legacy source modules as evidence only and imports, constructs, calls, patches, or activates none of them. No SQLite database, file event store, durable snapshot, checkpoint artifact, sidecar, WAL, backup, migration, model/vector activation, scheduler, external effect, cutover, or production authority is introduced.

## M1-A implementation record

- `EventEnvelope`: frozen versioned schema, bounded canonical JSON, deterministic digest, caller-supplied identifiers/ordering, fixed `shadow_only` authority.
- `InMemoryEventKernel`: append-only in-memory ordering, duplicate-ID rejection, contiguous stream sequences, known-causation checks, immutable reads, and explicit reducer replay.
- No persistence, runtime hook, clock, thread, randomness, recovery, or legacy mutation authority.

## M1-B implementation record

The only registered target is:

```text
target_id:          legacy.activation.learn_pair
module:             adapters/activation_adapter.py
callable:           ActivationAdapter.learn_pair
source range:       103-105
M0-D disposition:   WRAP
observer stream:    shadow:legacy.activation.learn_pair
```

The observer requires the exact reviewed Python bound method before any snapshot or legacy call. It preserves the legacy return value or identical exception object, emits only after-the-fact in-memory candidates, and isolates observer failures as explicit immutable records. It is not installed in production.

Candidate types remain diagnostic only:

- `shadow.legacy_mutation_observed_candidate`;
- `shadow.legacy_mutation_failed_candidate`.

## M1-C implementation record

`core/shadow_projection.py` defines three versioned contracts:

```text
eve.shadow-projection.activation-learn-pair.v1
eve.shadow-projection-checkpoint.v1
eve.shadow-equivalence-report.v1
```

The bounded projection stores only immutable tuples for the registered stream's `calls` and `learned` state plus consumed sequence/event digest metadata.

The reducer accepts only the exact M1-B producer/version, target, stream, event types, causal-context shape, target metadata, and success/failure outcome contract. Every snapshot must keep `learned` as an ordered subsequence of `calls`. A transition then requires:

- one-based contiguous projection sequence;
- event `before` snapshot equal to current projection state;
- exactly one appended legacy-call record;
- unchanged call-log prefix;
- success → attempted pair appended exactly once to learned state;
- failure → learned state unchanged.

Malformed scope, state mismatch, sequence gap, impossible learned-state ordering, and invalid transition raise visible typed errors before a new projection state is returned. Because states are immutable, failed reduction leaves the prior projection unchanged.

Cross-stream causation metadata is preserved in the envelope but not reinterpreted by the bounded projection. M1-C ordering authority is limited to the registered stream sequence.

Equivalence comparison returns an immutable report containing deterministic projected and expected snapshot digests plus explicit `calls_mismatch` and/or `learned_mismatch` codes. A mismatch does not alter legacy state or projection state.

Checkpoint and rollback boundaries are immutable in-memory values only. A checkpoint contains a caller-supplied canonical ID, bounded projection state, and verified state digest. Restore/rollback performs no I/O and cannot roll forward to a future checkpoint.

M1-C does **not** claim:

- coverage beyond `ActivationAdapter.learn_pair`;
- reconstruction of the full legacy runtime;
- production observation or automatic replay;
- durable persistence or crash recovery;
- retry, suppression, recovery, or mutation authority;
- event-store cutover or general equivalence.

## M1-D implementation record

`core/shadow_lifecycle.py` defines four versioned declaration families:

```text
eve.shadow-lifecycle-owner.v1
eve.shadow-bridge-contract.v1
eve.shadow-bridge-registry.v1
eve.shadow-bridge-failure.v1
```

The exact reviewed domain/source/disposition registry is immutable:

```text
activity → adapters/agency_adapter.py → WRAP
chat     → language/streaming.py      → REWRITE
goal     → adapters/goal_adapter.py   → WRAP
memory   → adapters/memory_adapter.py → WRAP
```

Every bridge is fixed to:

```text
lifecycle_status:    declared_disconnected
integration_mode:    disconnected
default_enabled:     false
authority:           none
emitted_event_types: ()
required_capabilities: ()
persistence_mode:    none
retry_policy:        forbidden
suppression_policy:  forbidden
rollback_scope:      shadow_state_only
```

Every owner explicitly declares initialization, shutdown, interruption, failure propagation, provenance, and rollback responsibilities. Unknown owners, bridges, or domains fail closed. Missing, duplicate, cross-domain, source-spoofed, disposition-spoofed, authority-bearing, event-emitting, capability-bearing, persistence-bearing, retrying, or suppressing declarations are rejected at construction.

`BridgeFailureSignal` records bridge, owner, domain, stage, error type, and SHA-256 of the error message without retaining the raw message. It requires original failure propagation and forbids retry, suppression, or legacy-authority changes. Direct construction cannot spoof bridge/owner/domain ownership.

M1-D imports, constructs, calls, patches, or activates no reviewed legacy module. It grants no runtime integration, event emission, persistence, recovery, scheduler, external-effect, or cutover authority.

## Merged source-of-truth evidence

- `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md`: 7,842 in-memory mutation sites; 283 direct-write sites. Its 13,341 total is an evidence-entry count, not an object count.
- `docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md`: merged canonical failure figures broad 614, silent 597, silent broad 525.
- `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md`: legacy persistence plus gzip/pickle sidecar evidence; no cutover contract was implemented by M0-C.
- `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md`: 63 axes = 26 mutable legacy + 37 read-only registry; 59 `MAPPED`, 4 `PROPOSED-DROP`, 0 `UNRESOLVED`.
- `docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md`: 1,225 component evidence entries; 75 life-loop entries; integrated pre-M0-D failure baseline broad 614, silent 607, silent broad 532.
- `docs/audit/M0_D_MODULE_DISPOSITION.md`: 288 runtime modules; KEEP 30, WRAP 78, REWRITE 6, EXPERIMENTAL 172, DEPRECATE 2, REMOVE 0.

## Figure provenance and discrepancy register

| Measure | Merged M0-B snapshot `eea70c...` | Integrated pre-M0-D tree `fe10cd...` | Treatment |
|---|---:|---:|---|
| broad handlers | 614 | 614 | Same value, different regeneration context |
| silent handlers | 597 | 607 | Preserve both; do not silently normalize |
| silent broad handlers | 525 | 532 | Preserve both; do not silently normalize |

Count semantics remain fixed:

- 13,341 = M0-A total evidence entries, not objects;
- 1,225 = M0-D component evidence entries, not modules or owners;
- 288 = module disposition units;
- 75 = life-loop entries; taxonomy occurrence totals may exceed 75 because one callable may map to multiple categories.

## Dual-gate status

### Historical audit gate

M0-A/B/C are pinned by PR #141; exact-head path handling is corrected by PR #143. `.github/workflows/exact-head-validation.yml` regenerates M0-A through M0-D at base and head and requires byte identity.

### Forward regression gate

The frozen v4.1 current-tree baseline remains:

```text
baseline SHA:         8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b
fingerprint digest:   5c01be8cf2de84e82ef1cf1e7e786fb1d0b00a27cd1f050862a7f83fc21ca055
unique fingerprints:  7,236
occurrences:          10,702
```

Baseline category occurrences:

```text
mutation:          8,049
direct_write:        319
silent_broad:         525
adaptive_numeric:   1,639
raw_capability:       170
```

The gate enforces **unregistered delta = 0**. It rejects unregistered findings, new parse errors, baseline drift, stale or over-counted registrations, metadata mismatches, and wrong-PR provenance.

Reviewed additions are registered by introducing PR:

- PR #145: forward scanner and focused gate tests;
- PR #146: M1-A kernel and focused tests;
- PR #147: M1-B observer and focused tests;
- PR #148: M1-C projection and focused tests;
- PR #149: M1-D lifecycle declarations and focused tests.

PR #149 registers 17 fingerprints / 24 occurrences: 8 / 12 in `core/shadow_lifecycle.py` and 9 / 12 in `tests/test_v4_m1_d_shadow_lifecycle.py`. It adds no registered direct-write, silent-broad, or raw-capability finding. Total registered additions are 234 occurrences. Registration is review evidence, not automatic runtime authority.

## Governance registry

### Frozen-PR dispositions

| Disposition | PRs | Meaning |
|---|---|---|
| `REWRITE-AS-V4-CONTRACT` | #109, #86, #84, #82 | Preserve evidence and tests, then restate under v4 contracts; do not merge the frozen branch. |
| `ABSORB-INTO-M1` | #11, #7, #4 | Preserve safety and validation requirements as M1 inputs; do not merge the obsolete activation bundle. |

## Milestone Registry

This registry may be adjusted by a reviewed STATUS update without a constitutional amendment. It may refine sequencing and evidence requirements but may not weaken the constitution, bypass exact-head validation, or make promotion/cutover automatic.

### M1 — Event kernel and shadow acceptance

| ID | Purpose | Entry | Exit |
|---|---|---|---|
| M1-A | Define and implement the minimal event-kernel envelope, causal metadata, reducer boundary, and append-only in-memory contract without persistence authority. | v4.1 merged; forward scanner active; exact-head green. | Kernel contract tests pass; pre-kernel runtime authority unchanged; no persistence/default activation. |
| M1-B | Add a shadow observer around registered legacy mutation funnels, emitting non-authoritative diagnostic candidates only. | M1-A accepted; target funnels registered. | Candidate coverage and no-side-effect proofs pass; no candidate has recovery, persistence, or mutation authority. |
| M1-C | Implement deterministic reducers, replay checks, and event/state equivalence for the bounded M1 envelope. | M1-B stable; schemas versioned. | Replay reconstructs bounded shadow state; failures are visible and rollbackable. |
| M1-D | Map kernel lifecycle ownership, failure propagation, and registered bridges for chat, activity, memory, and goals without cutover. | M1-C equivalence evidence accepted. | Every M1 bridge has owner, shutdown, error, provenance, and rollback contracts. |
| M1-E | Run the defined shadow observation window and produce M1 shadow-acceptance evidence. | M1-D complete; exact-head validation green. | Human review accepts shadow criteria; this grants eligibility to open v4.2 review only. |

### M2 — Persistence, capability edges, migration, and cutover

| ID | Purpose | Entry | Exit |
|---|---|---|---|
| M2-A | Add append-only SQLite shadow persistence, schema versions, snapshots, integrity checks, and bounded backup policy with legacy authority retained. | Accepted M1 kernel envelope. | Shadow writes and restores are reproducible; legacy remains authoritative. |
| M2-B | Mechanically extract and approve the read-capability edge manifest from source/raw stores through cognition to expression/generation. | M2-A schemas stable; source-store boundaries identified. | Every approved edge has capability, provenance, quarantine, quotation, and denial semantics; no unknown raw-text edge remains. |
| M2-C | Implement migration tooling and dual-read comparison for the bounded state envelope, including legacy sidecar evidence. | M2-A/B accepted; migration schemas versioned. | Dual-read equivalence and incompatibility reporting pass without changing authority. |
| M2-D | Rehearse snapshot restore, replay equivalence, corrupt-state handling, forced termination, and rollback under a defined observation window. | M2-C dual-read stable. | Rehearsal evidence is complete and independently reviewed; rollback succeeds. |
| M2-E | Conduct explicit human-reviewed persistence cutover, making the event store authoritative and legacy sidecars read-only evidence. | M2-D accepted; exact-head validation and approval complete. | Cutover head approved; post-cutover observation window passes; rollback remains available. |

### M3 — Affect, drives, goals, and continuity

| ID | Purpose | Entry | Exit |
|---|---|---|---|
| M3-A | Version the 63-axis migration schema and target drives/appraisal/derived-emotion contracts from the merged Affect Migration Plan. | Required M2 persistence schemas available. | All 63 axes have executable mapping/drop preservation contracts matching 59/4/0 rulings. |
| M3-B | Run read-only/shadow affect projections with provenance and no live behavioral authority. | M3-A accepted. | Projection equivalence, bounds, and no-side-effect evidence pass over the observation window. |
| M3-C | Integrate bounded drive and goal proposals behind explicit validation and event-kernel authority. | M3-B stable; goal owners mapped. | Proposals are causal, reviewable, rollbackable, and cannot directly trigger speech or external action. |
| M3-D | Prove identity, memory, replay, and historical interpretation continuity across mapped and proposed-drop axes. | M3-C schemas stable. | Original values/provenance remain readable; memory and identity continuity tests pass. |
| M3-E | Authorize affect/goal cutover through separate human review, versioned activation, and rollback. | M3-D accepted; M2-E authority available. | New affect/goal state becomes authoritative only on approved head and completed observation window. |

### M4 — Autonomous life and activity scheduler

| ID | Purpose | Entry | Exit |
|---|---|---|---|
| M4-A | Define the activity-scheduler contract, lifecycle ownership, interruption, resumption, and resource bounds. | M1/M2 kernel and persistence authority stable. | Scheduler contract approved; no timer-driven speech trigger is introduced. |
| M4-B | Map all 75 life-loop entries to explicit owners, scheduler behavior, retained taxonomy, and disposition. | M4-A accepted. | Every entry has owner and mapping; `no-v4-equivalent` items receive explicit decisions. |
| M4-C | Run autonomous activity in shadow mode, separating Vital/Cognitive/Goal/Activity/Learning/Memory/Social work from Expression. | M4-B complete; M3 goal contracts available where needed. | Shadow activity shows bounded causality and no unauthorized expression or external effects. |
| M4-D | Prove long-running continuity, interruption/resumption, crash recovery, privacy separation, and failure visibility. | M4-C observation window stable. | Continuity and recovery evidence pass under bounded resource and rollback policies. |
| M4-E | Conduct human-reviewed autonomous-life acceptance and bounded activation. | M4-D accepted; exact-head green. | Approved capabilities activate on a versioned head; no claim of subjective consciousness is made. |

## Promotion rule

M1-E acceptance grants only eligibility to open a human-reviewed v4.2 amendment review. Promotion is never automatic. v4.2 requires its own exact-head validation and explicit approval.

## Current next step

Begin **M1-E** only after M1-D merges. M1-E must:

1. define a versioned, deterministic, bounded observation-window schema and explicit acceptance criteria;
2. use only explicit test/dry-run invocation with no production hook, default activation, clock-derived ordering, scheduler, or autonomous loop;
3. cover the registered `ActivationAdapter.learn_pair` candidate/reducer stream and all four M1-D lifecycle declarations without claiming unimplemented bridge execution;
4. record candidate success/failure visibility, sequence continuity, replay equivalence, checkpoint/rollback evidence, lifecycle ownership coverage, and zero unauthorized effects;
5. preserve legacy outputs, exception identity, call ordering, state, persistence behavior, and defaults;
6. produce an immutable human-review packet whose acceptance is explicit and never automatic;
7. add no SQLite/file event persistence, retry, suppression, recovery, external effect, scheduler activation, cutover, or v4.2 authority;
8. register every justified scanner finding in the same PR;
9. pass focused M1-E tests, the forward gate, historical audit invariance, collection, and the full suite.
