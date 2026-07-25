# M2-E Habitat Fix 2 — A11 Canonical-Size Repair

## Scope and authority

Preflight base is `1b5e8ffb268a47d215bcc6f6ef1235b147559bb7`. This repair is limited to the append-only shadow event/snapshot store, M2-E habitat verification, chaos coverage, and STATUS. It does not transfer runtime authority, authorize cutover, open M3-C/M3-E, or change a production default.

## Incident diagnosis

The operator-private A1 evidence on 2026-07-25 named the previously silent failure:

```text
InvalidEventEnvelope: event_material exceeds canonical size limit
context: append_snapshot_backup
sequence boundary: 280
```

The operator reports two deterministic occurrences at the same boundary:

- 2026-07-24 11:34: original freeze after sequence 279 at 11:29 under the fixed 300-second stimulus cadence; the pre-A1 broad path did not retain the exception identity.
- 2026-07-25 18:1x: the guarded A1 path retained the named `InvalidEventEnvelope` at the same sequence-280 boundary and `append_snapshot_backup` context.

The first occurrence's exception class was not directly recorded and is not retroactively fabricated. The repeated sequence/timing boundary plus the now-visible exception and the exact code path identify the same deterministic canonical-size boundary.

Repository inspection refines the original hypothesis: the growing material is not confined to the snapshot table. The habitat `_event(sequence)` includes cumulative full `before` and `after` shadow snapshots in every event payload. The logical event payload still fits the event-envelope limit at sequence 280, but the SQLite persistence representation wraps that canonical payload JSON as a string inside `event_material`; escaping overhead makes that persistence material cross the unchanged 65,536-byte canonical limit at sequence 280. New snapshots also carried full state inline and would eventually encounter the same design boundary as state continues to grow.

**Growth-source finding:** the approximately linear event growth is produced by the M2-E habitat driver's `_snapshot_for(sequence)` recovery-state material embedded in each synthetic event's `before`/`after`; it is not growth introduced by `core.event_kernel.EventEnvelope`, which only validates the payload it is given.

The same driver material reaches the unchanged logical `EventEnvelope` payload limit at sequence 304 (sequence 303 still fits). That is intentionally **not** compacted by this repair: sequence 304 remains fail-closed under the existing event contract. The fixed M2-E window quota is 288, so the reviewed observation window ends before that logical boundary. The existing one-row pending-commit reconciliation remains a robustness path for a store that has exactly one more durable row than `window_state.json`; code inspection shows the specific seq-280 exception is raised before insertion, so rows=279 is the expected incident state.

## A11 repair

The active v4.2 A11 mutation-state fidelity rule permits large persisted state to be represented by a digest over a versioned canonical representation plus a revalidatable structural manifest. This patch applies that rule at persistence boundaries without weakening or replacing the logical event contract.

1. `MAX_CANONICAL_JSON_BYTES` remains exactly 65,536. There is no threshold increase.
2. A new append-only `content_materials` table stores canonical JSON material by SHA-256 digest.
3. When the normal persisted `event_material` fits, its v1 inline representation is preserved. Only the SQLite persistence representation that would exceed the canonical limit switches to a compact `payload_reference` containing the content digest and structural manifest. Reads re-resolve and verify the material before reconstructing the original immutable `EventEnvelope`; the logical envelope digest and payload are unchanged.
4. Large snapshots, and subsequent snapshots once the A11 content store is active, place full state material in the content-addressed store. The snapshot row contains only the content reference while `state_digest` remains the digest of the full canonical state. The structural manifest binds serialization schema/version, SHA-256, canonical byte count, top-level key domain, and applicable collection counts.
5. Replay-generated states use the same unbounded-but-canonical JSON representation and SHA-256 method as snapshot content. Snapshot equivalence therefore remains a comparison of full-state digests, not reference-object digests.
6. Exact legacy v1 stores receive an additive schema extension that creates only the append-only content table/triggers. The frozen v1 migration history remains byte-for-byte unchanged at its single original row; legacy inline snapshot rows remain readable through an explicit format branch. No old snapshot is rewritten or deleted.
7. `core/event_kernel.py`, `core/shadow_projection.py`, `core/sqlite_shadow_store.py`, and the reviewed A1 guarded runtime remain unchanged. The habitat wrapper changes only its SQLite store binding to the A11 persistence implementation.
8. Missing/corrupt content material, digest mismatch, manifest mismatch, malformed storage schema, and append-only trigger/schema drift fail closed.

## Required recovery/chaos proofs

Focused coverage proves:

- an exact legacy-v1 store is additively extended and an existing inline snapshot remains readable;
- a synthetic snapshot whose full canonical state exceeds 65,536 bytes succeeds through digest+manifest content addressing;
- an actually oversized `EventEnvelope` payload is still rejected by the unchanged event-kernel limit;
- the old inline `event_material` path still works through sequence 279;
- sequence 280 crosses that old persistence-material boundary, is stored by content reference, reads back as the identical historical logical `EventEnvelope`, and passes integrity;
- a frozen `window_state.json` at 279 with durable row 280 passes the existing reviewed one-pending-row reconciliation and resumes at event count 280;
- the fixed quota is 288 and its logical event still uses the original full `before`/`after` payload;
- sequence 303 still fits the original logical payload, while sequence 304 is rejected by the unchanged `EventEnvelope` canonical-size limit rather than compacted.

## Operational gate

Until this repair is merged and exact-head validation is green, the phone habitat window remains frozen. After merge, the reviewed operator sequence is unchanged in principle: stop the supervisor, update `main`, run the explicit `resume --reviewed` gate, then restart the supervisor. A successful resume continues the bounded M2-E shadow observation window only; it is not cutover authorization.
