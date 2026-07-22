# EVE v4 Implementation Status

Active constitution: **EVE v4.2**
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**
Runtime status: **pre-kernel legacy runtime remains authoritative; M1 mechanism evidence is human-accepted, while M1-A through M1-E remain shadow/declaration/evidence-only with no production integration**
Previous v3/v3.1/v4.1 documents: historical reference only
M0 status: **closed**
Forward-gate status: **implemented and enforced by exact-head validation**
M1-A status: **completed by PR #146**
M1-B status: **completed by PR #147**
M1-C status: **completed by PR #148**
M1-D status: **completed by PR #149**
M1-E status: **completed and explicitly human-accepted by PR #158; immutable machine packet remains non-authoritative**
M1 status: **closed for mechanism verification; coverage remains deferred to A2/M2 dual-read and cutover**
v4.2 review status: **opened and closed by the v4.2 amendment; `v4_2_review_opened = true`; no outstanding constitutional objection**
v4.2 review consensus: **6 objections accepted plus 1 refinement requiring supersession/revocation artifacts to pass the same exact-head and human-review regime as the decisions they change**
M2-A status: **implemented as a bounded, explicit, disconnected SQLite shadow-persistence candidate in PR #161; not yet human-accepted or merged, with no runtime, recovery, cutover, or legacy-authority change**
M3-A drive-dynamics design status: **documentation-only parallel candidate in PR #169; binds the 63-axis Affect Migration Plan to 8 drives, 32 semantic states, and 48 bidirectional named transitions; no runtime integration, no scheduler or M3-E authority, and integration eligibility only after persistence cutover**
PR #158 gate wording retained for historical verification: **M2-A remains blocked until v4.2 approval**. This amendment satisfies that constitutional prerequisite only; it does not start M2-A.
Current next step: **complete exact-head validation and human review of PR #161; M2-B remains blocked until M2-A is accepted with its schemas and restore evidence stable**
Frozen work: open REWRITE PRs #109, #86, #84, and #82 remain untouched; absorbed PRs #11, #7, and #4 are closed
Constitution merge baseline: `8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b`
Forward-gate merge baseline: `1ed1093cfec05b44848ad0d117e45885a5669b69`
M1-A merge baseline: `1a3da9aee41c0bed065bb0bbbcc2e8e577aa50f9`
M1-B merge baseline: `15e993780d4c2744047237f877f5add1f7f66339`
M1-C merge baseline: `2546548a4bf757d0fc7b915be1dac7749e7c9824`
M1-D merge baseline: `dadc9be7ea67aa9a7f95499d2c874677b00cbcbb`
M1-E machine-evidence merge baseline: `76e7df1d6bd0194ccd1925fc1b906a359b0c5aef`
M1 controlled-evidence merge baseline: `847621bcd61634958ce505108ade491c50ced0d4`
M1 expanded-evidence merge baseline: `7c4573e628e5ac51d0d64ad1040078741f3630e0`
M1 accepted evidence head: `560b9b54f3237d63762b81da38e7c25c36922214`
v4.2 amendment baseline: `40a2a42da235d6ac97867c20a57620830a35fecd`

## Current authority

EVE v4.2 is the active constitutional authority. The existing application remains the **pre-kernel legacy runtime** and retains all current runtime and persistence authority.

M1-A provides an immutable canonical `shadow_only` event envelope, append-only in-memory kernel, and explicit reducer boundary. M1-B provides a separately invoked after-the-fact observer for one registered `ActivationAdapter.learn_pair` legacy funnel. M1-C provides a versioned immutable shadow projection, deterministic bounded reducer/replay, explicit equivalence reports, and immutable in-memory checkpoint/rollback values for that same single stream. M1-D provides immutable lifecycle-owner, disconnected bridge-registry, reviewed source/disposition, and redacted failure-propagation declarations for bounded chat, activity, memory, and goal domains. M1-E provides a deterministic in-memory evaluator and immutable machine-review packet for explicitly supplied M1-B through M1-D evidence.

None of M1-A through M1-E is connected to `main.py`, `language/streaming.py`, live/autonomous loops, production composition, persistence adapters, or default startup paths. M1-D names legacy source modules as evidence only. M1-E imports or calls no legacy module and installs no observer or bridge. No SQLite database, file event store, durable snapshot, checkpoint artifact, sidecar, WAL, backup, migration, model/vector activation, scheduler, external effect, cutover, or production authority is introduced by v4.2.

PR #161 separately introduces `core/sqlite_shadow_store.py` as the M2-A candidate. Import and construction perform no I/O. A caller must explicitly initialize a concrete SQLite path and explicitly append immutable `shadow_only` envelopes or validated snapshots. The module is not imported by `main.py`, legacy persistence, live/autonomous loops, production composition, M1 observers, or lifecycle bridges. It grants no dual-read, authoritative recovery, migration cutover, scheduler, model/vector activation, or production persistence authority.

## M1 implementation records

### M1-A — event kernel

- `EventEnvelope`: frozen versioned schema, bounded canonical JSON, deterministic digest, caller-supplied identifiers and ordering, fixed `shadow_only` authority.
- `InMemoryEventKernel`: append-only in-memory ordering, duplicate-ID rejection, contiguous stream sequences, known-causation checks, immutable reads, and explicit reducer replay.
- No persistence, runtime hook, clock, thread, randomness, recovery, or legacy mutation authority.

### M1-B — registered legacy observer

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

### M1-C — bounded projection and replay

`core/shadow_projection.py` defines:

```text
eve.shadow-projection.activation-learn-pair.v1
eve.shadow-projection-checkpoint.v1
eve.shadow-equivalence-report.v1
```

The reducer accepts only the exact M1-B producer/version, target, stream, event types, causal-context shape, target metadata, and success/failure outcome contract. Every snapshot keeps `learned` as an ordered subsequence of `calls`.

A valid transition requires one-based contiguous projection sequence; event `before` equal to current projection state; exactly one appended legacy-call record; unchanged call-log prefix; success appending the attempted pair exactly once; and failure leaving learned state unchanged. Malformed scope, mismatch, sequence gap, impossible ordering, and invalid transition fail closed. Checkpoint and rollback remain immutable in-memory values with no durable recovery authority.

### M1-D — disconnected lifecycle contracts

`core/shadow_lifecycle.py` defines:

```text
eve.shadow-lifecycle-owner.v1
eve.shadow-bridge-contract.v1
eve.shadow-bridge-registry.v1
eve.shadow-bridge-failure-signal.v1
```

The immutable reviewed registry is:

```text
activity → adapters/agency_adapter.py → WRAP
chat     → language/streaming.py      → REWRITE
goal     → adapters/goal_adapter.py   → WRAP
memory   → adapters/memory_adapter.py → WRAP
```

Every bridge is fixed to disconnected, default-disabled, no-authority, no-event, no-capability, no-persistence, no-retry, no-suppression behavior with shadow-state-only rollback. Every owner declares construction, shutdown, interruption, failure propagation, provenance, and rollback responsibilities. Invalid declarations fail closed.

### M1-E — machine shadow-acceptance evidence

`core/shadow_acceptance.py` defines:

```text
eve.m1-shadow-observation-window.v1
eve.m1-legacy-preservation-evidence.v1
eve.m1-shadow-acceptance-criterion.v1
eve.m1-shadow-acceptance-packet.v1
```

The immutable packet contains ten machine criteria:

```text
event_count_exact
success_failure_visible
observer_failure_visible
sequence_contiguous
replay_equivalent
checkpoint_restore_verified
rollback_verified
lifecycle_registry_complete
legacy_behavior_preserved
zero_unauthorized_effects
```

A complete machine packet may set only `eligible_for_human_review=true`. Its schema permanently fixes `human_review_status=required_not_performed`, `human_accepted=false`, `v4_2_eligible=false`, `authority=shadow_only`, `runtime_integrated=false`, and `persistence_mode=none`. It cannot accept itself, activate a bridge, grant persistence or recovery authority, perform cutover, or open v4.2 automatically.

### M1 human acceptance — external decision record

PR #158 records the separate decision in `docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.json` and `docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md`. The canonical JSON record SHA-256 is `aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`. It pins expanded evidence head `560b9b54f3237d63762b81da38e7c25c36922214`, raw artifact SHA-256 `3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac`, exact-head run `29826184624`, and artifact ZIP SHA-256 `5482da68f38e5d66400d6a32b948d559ce1dd6ce7ec80fe77de08659b8f9d0b9`.

The PR #158 record sets `human_accepted=true`, `m1_closed=true`, and `v4_2_eligible=true`. At creation it left `v4_2_review_opened=false` and `m2_started=false`. The immutable machine packet remains fixed and is not rewritten. The v4.2 amendment now records the later review state in this STATUS document rather than modifying the PR #158 artifacts.

The accepted scope is mechanism verification only. Historical coverage is deferred to A2/M2 dual-read and cutover; 527 unobserved historical sites remain tracked debt for progressive correction at WRAP.

## v4.2 constitutional amendment record

The v4.2 amendment is governance-only and changes exactly `AGENTS.md`, `docs/EVE_DESIGN_v4.md`, and this STATUS document. It creates no runtime, test, data, scanner, enforcement, database, persistence, model, vector, checkpoint, or generated artifact change.

The review accepted all six objections raised against the initial C1-C4 draft and accepted one additional refinement. C1-C4 remain draft-lineage aliases only; the constitutional lineage continues as A9-A12:

1. **A9 — Discrete-transition granularity.** PR #152 records 6 standalone deterministic tick steps with 0 events, maximum 1 event in one logical step, and 1.0 event per observed legacy mutating call. PR #153 separately records 4 standalone tick steps with 0 events and a live-thread tick observation with 0 events before discrete mutation. Continuous samples do not emit events; versioned named-state changes may emit in either direction; duplicate emission while the same state persists is prohibited.
2. **A10 — Evidence recalculability.** Raw observations or immutable SHA-256/schema-pinned references bound into the same package must permit independent recomputation of every claimed metric. Access and redaction rules must preserve authorized recomputability. Green verdicts alone are insufficient.
3. **A11 — Mutation-state fidelity.** Actual before/after values are required. Large state may use an identical-method canonical digest plus revalidatable structural manifest. `state_changed` is computed, not manually asserted; every record carries a transition hash and replay result.
4. **A12 — Append-only decision records.** Machine packets and human decisions are immutable. Correction, replacement, supersession, or revocation uses a separate digest-linked append-only artifact. That artifact must pass the same exact-head validation and human-review regime as the decision it changes.

This amendment records `v4_2_review_opened=true` and closes that review on constitutional merge. It does not set `m2_started=true`.

## M2-A implementation candidate — PR #161

`core/sqlite_shadow_store.py` defines the bounded durable contracts:

```text
eve.sqlite-shadow-store.v1
eve.sqlite-shadow-migration.v1
eve.sqlite-shadow-snapshot.v1
eve.sqlite-shadow-append-receipt.v1
eve.sqlite-shadow-snapshot-receipt.v1
eve.sqlite-shadow-integrity-report.v1
eve.sqlite-shadow-restore-report.v1
eve.sqlite-shadow-backup-receipt.v1
```

The candidate provides explicit file initialization, a WAL request with visible fallback reporting, explicit SQLite transactions, immutable migration history, update/delete rejection triggers for durable tables, canonical envelope digests, a chained durable event digest, computed before/after count and chain evidence, readback verification before commit, bounded event/byte/snapshot/backup policy, periodic snapshot eligibility, snapshots bound to the current stream head, newest-valid-snapshot selection with corrupt-snapshot fallback, repeated deterministic restore verification, SQLite plus logical integrity reports, and verified bounded backups. Historical events are never pruned; storage-limit exhaustion rejects the new append instead of deleting prior history.

The candidate remains limited to the accepted M1 event-envelope contract and caller-supplied pure reducer/state codecs. It does not install the M1-B observer, connect an M1-D bridge, read legacy sidecars, compare dual reads, become the recovery authority, alter defaults, or perform cutover. Those boundaries remain assigned to later M2 milestones and separate human-reviewed decisions.

`tests/test_v4_m2_a_sqlite_shadow_store.py` provides focused evidence for explicit creation, WAL/schema/migration contracts, append ordering and hash-chain fidelity, atomic rollback, append-only enforcement, bounded storage, validated snapshots, corrupt-snapshot fallback, repeated replay, reopen after an uncommitted write, integrity-failure visibility, bounded backups, and absence of production integration.

## M3-A drive-dynamics documentation-only candidate — PR #169

`docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md` fixes a versioned bounded continuous equation and the complete parameter set for `energy`, `safety`, `affiliation`, `curiosity`, `agency`, `coherence`, `competence`, and `expression`. It defines 32 semantic states, 24 hysteresis boundaries, 48 bidirectional named transitions, exact cooldowns, the transition-candidate lifecycle, and the A9 no-duplicate proof.

`scripts/audit/m3_a_drive_dynamics_check.py` is a standard-library static checker. It parses the merged 63-axis Affect Migration Plan directly and requires all 59 `MAPPED` axes to land in versioned drive/appraisal/derived-emotion targets while all 4 `PROPOSED-DROP` axes retain historical preservation with no future behavioral target. Focused tests verify deterministic output, exact catalog coverage, and the authority boundary.

This candidate is documentation-only and runs in parallel with the M2-E observation-window work. It performs no runtime integration, affect or drive mutation, event emission, persistence access, scheduler integration, goal or expression integration, cutover, production-default change, or M3-E authorization. The legacy runtime remains authoritative. Integration eligibility only after persistence cutover means that this design cannot activate or promote itself before a separate approved cutover and later reviewed M3 implementation.

## Merged source-of-truth evidence

- `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md`: 7,842 in-memory mutation sites; 283 direct-write sites. Its 13,341 total is an evidence-entry count, not an object count.
- `docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md`: merged canonical failure figures broad 614, silent 597, silent broad 525.
- `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md`: legacy persistence plus gzip/pickle sidecar evidence; no cutover contract was implemented by M0-C.
- `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md`: 63 axes = 26 mutable legacy + 37 read-only registry; 59 `MAPPED`, 4 `PROPOSED-DROP`, 0 `UNRESOLVED`.
- `docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md`: 1,225 component evidence entries; 75 life-loop entries; integrated pre-M0-D failure baseline broad 614, silent 607, silent broad 532.
- `docs/audit/M0_D_MODULE_DISPOSITION.md`: 288 runtime modules; KEEP 30, WRAP 78, REWRITE 6, EXPERIMENTAL 172, DEPRECATE 2, REMOVE 0.
- `docs/audit/M1_CONTROLLED_OBSERVATION_EVIDENCE.md` from PR #152: standalone deterministic ticks 6 with events during tick steps 0; maximum events in one logical step 1; events per observed legacy call 1.0.
- `docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_EVIDENCE.md` and raw JSON from PR #153: standalone tick steps 4 with events 0; live tick count at barrier 1 and events from live tick before discrete mutation 0; raw observations sufficient to recompute every report metric.
- `docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md` and JSON from PR #158: immutable machine packet retained separately from explicit human acceptance, with exact evidence and workflow artifact pins.

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

No v4.2 source-of-truth discrepancy was found among PRs #152, #153, and #158. The tick evidence is intentionally reported as standalone 6 + standalone 4 + a separate live-thread observation, not silently summed into one count.

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

Reviewed additions are registered by introducing PR: #145 forward scanner; #146 M1-A; #147 M1-B; #148 M1-C; #149 M1-D; #150 M1-E; #151 evidence-gap documentation; #152 controlled evidence; #153 corrected expanded evidence; #158 external human acceptance; and candidate PR #161 M2-A SQLite shadow persistence. Registration is review evidence, not automatic runtime authority.

## Governance registry

### Frozen-PR dispositions

| Disposition | PRs | Meaning |
|---|---|---|
| `REWRITE-AS-V4-CONTRACT` | #109, #86, #84, #82 | Preserve evidence and tests, then restate under v4 contracts; do not merge the frozen branch. |
| `ABSORB-INTO-M1` | #11, #7, #4 | Closed after their safety and validation requirements were absorbed into M1; do not reopen or merge the obsolete activation bundles. |

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
| M2-A | Add append-only SQLite shadow persistence, schema versions, snapshots, integrity checks, and bounded backup policy with legacy authority retained. | Accepted M1 kernel envelope and active v4.2 constitution. | Shadow writes and restores are reproducible; legacy remains authoritative. |
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

M1-E machine evidence did not itself grant promotion. PR #158 separately closed M1 mechanism verification and made the project eligible to open v4.2 review. The v4.2 amendment has now opened and closed that review through its own exact-head validation, human review, and constitutional merge. PR #161 starts only the bounded M2-A implementation candidate. It does not activate a runtime bridge, dual-read path, recovery authority, cutover, or any production capability.

## Current next step

M2-A is now implemented only as the separate candidate in PR #161. Until that exact head is independently validated, human-reviewed, and merged:

1. M2-A remains `shadow_only`, disconnected, and non-authoritative;
2. no observer, bridge, default persistence path, scheduler, recovery behavior, dual-read, cutover, or production hook may be activated;
3. the pre-kernel legacy runtime and legacy persistence remain authoritative;
4. M2-B and later M2 work remain blocked;
5. the 527 unobserved historical sites remain tracked debt, not safe coverage;
6. A9-A12 bind all M2 evidence, acceptance, supersession, and revocation artifacts.
