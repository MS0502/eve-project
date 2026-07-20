# EVE v4 Implementation Status

Active constitution: **EVE v4.1**
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**
Runtime status: **pre-kernel legacy runtime; no v4 runtime implementation or activation claim**
Previous v3/v3.1 documents: historical reference only
M0 status: **closed**
Current next step: **forward-regression scanner and forward-additions manifest infrastructure**
Frozen work: open implementation PRs #109, #86, #84, #82, #11, #7, and #4
Constitution baseline: `646191c082a18e817d114cdff4a2d81b404fda33`

## Current state

The v4.1 amendment is documentation and governance only. It changes no runtime code, test, data, model, vector, checkpoint, persistence state, scanner, enforcement tool, default, or frozen branch. No event kernel, event-store authority, persistence cutover, capability-edge manifest, affect conversion, scheduler, or autonomous-life activation is implemented by v4.1.

The existing implementation is designated the **pre-kernel legacy runtime**. Terms such as “event” or “authoritative event store” describe future accepted architecture, not the authority of current mutations, sidecars, debug exports, or diagnostic envelopes.

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

Count semantics are also fixed:

- 13,341 = M0-A total evidence entries, not objects.
- 1,225 = M0-D component evidence entries, not modules or owners.
- 288 = module disposition units.
- 75 = life-loop entries; taxonomy occurrence totals may exceed 75 because one callable may map to multiple categories.

## Dual-gate status

### Historical audit gate

Implemented infrastructure exists. M0-A/B/C are pinned by PR #141; exact-head path handling is corrected by PR #143. `.github/workflows/exact-head-validation.yml` regenerates M0-A through M0-D at base and head and requires byte identity.

### Forward regression gate

Not yet implemented. Until the scanner and forward-additions manifest exist, the following are review-enforced:

- no unregistered new mutation or direct-write entry;
- no unregistered adaptive/numeric state owner, artifact writer, learned-state repository, or vector/weight persistence path;
- no new silent+broad exception handler;
- no new raw-external-text capability edge into expression/generation;
- justified kernel or audit additions must be registered in the same PR that introduces them.

The forward scanner is the first post-v4.1 infrastructure PR. It must not modify the frozen historical audit outputs.

## Governance registry

### Frozen-PR dispositions

Copied from `docs/audit/M0_D_MODULE_DISPOSITION.md`:

| Disposition | PRs | Meaning |
|---|---|---|
| `REWRITE-AS-V4-CONTRACT` | #109, #86, #84, #82 | Preserve evidence and tests, then restate under v4 contracts; do not merge the frozen branch. |
| `ABSORB-INTO-M1` | #11, #7, #4 | Preserve safety and validation requirements as M1 inputs; do not merge the obsolete activation bundle. |

### Required infrastructure

- `.github/workflows/exact-head-validation.yml`
- historical snapshot pinning established by PR #141
- exact-head invariance correction established by PR #143
- future forward-regression scanner and forward-additions manifest

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

Build the small forward-regression scanner infrastructure PR. It must:

1. preserve frozen M0-A through M0-D outputs byte-identically;
2. run mutation/direct-write, adaptive/numeric, silent+broad, and raw-capability detectors against the current tree;
3. compare against frozen forward manifests;
4. support same-PR reviewed forward-additions registration;
5. change no runtime behavior or production authority.

After that infrastructure merges, begin M1-A.
