# EVE v4 Implementation Status

Active constitution: **EVE v4.1**
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**
Runtime status: **pre-kernel legacy runtime remains authoritative; M1-A kernel is shadow-only and disconnected**
Previous v3/v3.1 documents: historical reference only
M0 status: **closed**
Forward-gate status: **implemented and enforced by exact-head validation**
M1-A status: **completed by the merge carrying this STATUS update**
Current next step: **M1-B — registered legacy-funnel shadow observer**
Frozen work: open implementation PRs #109, #86, #84, #82, #11, #7, and #4
Constitution merge baseline: `8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b`
Forward-gate merge baseline: `1ed1093cfec05b44848ad0d117e45885a5669b69`

## Current state

EVE v4.1 is active constitutional authority. The existing application remains the **pre-kernel legacy runtime** and retains all current runtime authority.

M1-A adds a minimal event-kernel contract in `core/event_kernel.py`. It is intentionally disconnected from `main.py`, `language/streaming.py`, live/autonomous loops, persistence adapters, and every legacy mutation funnel. It creates no file, SQLite database, snapshot, sidecar, thread, clock, network request, model/vector load, or production default. It cannot observe or alter live runtime behavior.

The M1-A kernel accepts only immutable canonical envelopes marked `shadow_only`, stores them only in process memory, and exposes append/read/replay boundaries. This is an implementation of the kernel contract, not event-store cutover, persistence authority, or evidence that legacy mutations are already represented by events.

No shadow observer, event/state equivalence proof, persistence cutover, affect conversion, capability-edge manifest, activity scheduler, or autonomous-life activation exists yet. M1-B is the next milestone.

## M1-A implementation record

Implemented surfaces:

- `EventEnvelope`: frozen versioned schema, canonical bounded JSON payload and causal context, deterministic digest, caller-supplied identifiers and ordering, and fixed `shadow_only` authority.
- `AppendReceipt`: immutable in-memory append evidence with envelope digest.
- `InMemoryEventKernel`: append-only in-memory ordering, duplicate-ID rejection, per-stream contiguous sequence checks, known-causation checks, immutable read views, and explicit reducer replay.
- focused tests: canonicalization, deep immutability boundary, authority rejection, identifier/version validation, finite/JSON-only payloads, size/depth bounds, append atomicity, duplicate/sequence/causation failure, independent streams, reducer failure propagation, and static absence of persistence/clock/thread/random/runtime integration.

Fail-closed constraints:

- malformed or noncanonical envelope → reject;
- authority other than `shadow_only` → reject;
- duplicate event ID → reject before append;
- non-one-based or noncontiguous stream sequence → reject before append;
- unknown or self causation → reject;
- non-callable reducer or reducer returning `None` → reject;
- reducer exception → propagate visibly;
- new scanner findings → require same-PR reviewed registration.

Explicit exclusions:

- no legacy-runtime hook or observer;
- no event emission from chat, live, autonomous, memory, affect, goal, or persistence paths;
- no SQLite, file, checkpoint, snapshot, sidecar, WAL, backup, restore, migration, or cutover;
- no recovery or mutation authority for the kernel;
- no claim that current legacy state is replayable from M1-A envelopes.

## Merged source-of-truth evidence

- `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md`: 7,842 in-memory mutation sites; 283 direct-write sites. Its 13,341 total is an evidence-entry count, not an object count.
- `docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md`: merged canonical failure figures broad 614, silent 597, silent broad 525.
- `docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md`: legacy persistence plus gzip/pickle sidecar evidence; no cutover contract was implemented by M0-C.
- `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md`: 63 axes = 26 mutable legacy + 37 read-only registry; 59 `MAPPED`, 4 `PROPOSED-DROP`, 0 `UNRESOLVED`.
- `docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md`: 1,225 component evidence entries; 75 life-loop entries; pre-M0-D failure baseline broad 614, silent 607, silent broad 532.
- `docs/audit/M0_D_MODULE_DISPOSITION.md`: 288 runtime modules; KEEP 30, WRAP 78, REWRITE 6, EXPERIMENTAL 172, DEPRECATE 2, REMOVE 0.

## Figure provenance and discrepancy register

The source-of-truth duty found no contradiction between the v4.1 amendment task and the merged documents. It did find provenance-specific figures that must remain separate:

| Measure | Merged M0-B snapshot `eea70c...` | Integrated pre-M0-D tree `fe10cd...` | Treatment |
|---|---:|---:|---|
| broad handlers | 614 | 614 | Same value, different regeneration context |
| silent handlers | 597 | 607 | Preserve both; do not silently normalize |
| silent broad handlers | 525 | 532 | Preserve both; do not silently normalize |

The M0-B figures come from `M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md`. The integrated figures come from the A/B/C retrospective in `M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md`. This difference is evidence for audit snapshot pinning, not permission to rewrite either historical baseline.

Count semantics are fixed:

- 13,341 = M0-A total evidence entries, not objects.
- 1,225 = M0-D component evidence entries, not modules or owners.
- 288 = module disposition units.
- 75 = life-loop entries; taxonomy occurrence totals may exceed 75 because one callable may map to multiple categories.

## Dual-gate status

### Historical audit gate

M0-A/B/C are pinned by PR #141; exact-head path handling is corrected by PR #143. `.github/workflows/exact-head-validation.yml` regenerates M0-A through M0-D at base and head and requires byte identity.

### Forward regression gate

Implemented by:

- `scripts/audit/forward_regression_gate.py`;
- `docs/audit/FORWARD_ADDITIONS_MANIFEST.json`;
- `docs/audit/FORWARD_REGRESSION_GATE.md`;
- `tests/audit/test_forward_regression_gate.py`;
- the enforced forward-gate step in `.github/workflows/exact-head-validation.yml`.

The frozen v4.1 current-tree baseline is:

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

The gate enforces **unregistered delta = 0**, not absolute delta = 0. It rejects:

- unregistered new mutation or direct-write findings;
- unregistered adaptive/numeric findings;
- new silent+broad handlers;
- new raw-external-text-to-expression/generation candidates;
- new parse errors that would hide AST detector coverage;
- baseline digest drift;
- malformed, stale, metadata-mismatched, over-counted, or wrong-PR registrations.

M1-A registers exactly two reviewed groups under PR #146:

- `core/event_kernel.py`: kernel-owned in-memory append/canonical-digest findings; disposition `M1_A_EVENT_KERNEL`;
- `tests/test_v4_m1_a_event_kernel.py`: focused verification findings; disposition `TEST_EVIDENCE`.

The M1-A registration includes no direct-write, silent-broad, or raw-capability addition. Registration is evidence for review, not automatic authority.

## Governance registry

### Frozen-PR dispositions

Copied from `docs/audit/M0_D_MODULE_DISPOSITION.md`:

| Disposition | PRs | Meaning |
|---|---|---|
| `REWRITE-AS-V4-CONTRACT` | #109, #86, #84, #82 | Preserve evidence and tests, then restate under v4 contracts; do not merge the frozen branch. |
| `ABSORB-INTO-M1` | #11, #7, #4 | Preserve safety and validation requirements as M1 inputs; do not merge the obsolete activation bundle. |

### Required infrastructure

- `.github/workflows/exact-head-validation.yml`
- `scripts/audit/forward_regression_gate.py`
- `docs/audit/FORWARD_ADDITIONS_MANIFEST.json`
- historical snapshot pinning established by PR #141
- exact-head invariance correction established by PR #143

## Milestone Registry

This is the provisional registry of IDs referenced by the constitution. It may be adjusted by a reviewed STATUS update without a constitutional amendment. A STATUS update may refine sequencing and evidence requirements but may not weaken the constitution, bypass exact-head validation, or make promotion/cutover automatic.

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

Begin **M1-B** only after M1-A merges. M1-B must:

1. select a minimal bounded set of legacy mutation funnels from M0-A and register each target;
2. observe after-the-fact state-transition evidence without replacing or controlling the legacy mutation;
3. emit only `shadow_only` envelopes into the in-memory kernel;
4. preserve legacy return values, exceptions, ordering, state, persistence behavior, and defaults byte-for-byte where observable;
5. grant no observer recovery, retry, suppression, persistence, or mutation authority;
6. surface observer failure explicitly without converting it into a legacy-runtime failure or silent+broad handler;
7. pass focused no-side-effect/coverage tests, the forward gate, historical audit invariance, collection, and the full suite.
