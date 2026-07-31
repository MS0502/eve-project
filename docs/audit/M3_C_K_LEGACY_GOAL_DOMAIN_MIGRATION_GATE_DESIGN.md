# M3-C-K Legacy Goal-Domain Migration-Gate Design

## Status

Documentation-only continuation after the completed M3-C-J bounded private-device
observation window. This slice does not integrate a runtime path, transfer legacy
goal authority, migrate or delete legacy state, enable a writer, authorize action,
scheduling, or speech, or open M3-E.

Exact prerequisite:

```text
PR:                    #234
exact head:            c1cb149637077953f3ffc5d1bb3719b24072d2e6
exact run:             30608878495
focused:               No focused tests selected
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,332 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-c1cb149637077953f3ffc5d1bb3719b24072d2e6
artifact SHA-256:      a621c2e60e7d998a806396a0fa05531e97d5c1b0bee5bac90a279642051e2f25
M2-E run:              30608878468
M2-E:                  6/6 passed
merge SHA:             108e840f9acf688b0751902df4472dbe98c373f4
public-review digest:  3954f979a16d7f71892debcef442b60c67bcb75f47131d30e2126e04ada670ff
```

That evidence is immutable and reused. The completed private-device command,
controlled-input readiness, PR #211 phone witness, retained sequences 1 through
5, production database, sidecars, private journal, bundle, backup, and restore
evidence must not be rerun, deleted, recreated, or accessed by this design or CI.

## Purpose

M3-C-G through M3-C-J proved that one explicitly reviewed, bounded v4-native
goal-lifecycle stream can append and replay canonical named lifecycle
transitions on a private device while preserving rollback evidence. That result
is necessary but not sufficient to change behavior authority.

The legacy goal domain remains authoritative. This design fixes the separate
gate required before any later slice may even propose transferring that domain
to the v4-native goal-selection and lifecycle path.

## Fixed authority facts

```text
v4-native persistence substrate authoritative: true
M3 authority open:                             true
M3-C-J bounded observation window complete:    true
M3-C lifecycle writer currently enabled:       false
runtime integration performed:                 false
legacy goal-domain authority transferred:      false
legacy migration authorized:                   false
action authority:                              false
scheduler authority:                           false
speech authority:                              false
M3-E authority open:                           false
```

A successful persistence observation does not prove behavioral equivalence,
runtime ownership, safe cutover, or permission to delete legacy state.

## Constitutional basis

This gate applies the active v4.2 requirements that:

- legacy authority remains until a separately reviewed migration gate;
- migration preserves provenance and continuity;
- event-store cutover, domain migration, and downstream behavior authority are
  separate decisions;
- replay equivalence, dual-read evidence, rollback rehearsal, bounded
  observation, exact-head validation, and explicit human review are required;
- original evidence and legacy state are not silently rewritten or deleted;
- action, scheduling, speech, and M3-E require their own later authorities.

## Exact domain boundary

The future migration package must mechanically inventory the complete legacy
goal-domain authority surface before any runtime hook is added. At minimum it
must identify, with path and callable evidence:

1. every constructor and owner of legacy `GoalManagement` state;
2. every read of current goal, candidate, priority, status, or lifecycle state;
3. every mutation, persistence write, restore, reset, and deletion path;
4. every timer, loop, scheduler, action, speech, memory, affect, and UI consumer;
5. every fallback and exception path that can preserve or replace a goal;
6. every test that asserts legacy goal behavior or persistence;
7. every startup, shutdown, recovery, and reload path;
8. every bridge between legacy goal state and M3-C candidate/lifecycle state.

The inventory must report exact counts and zero unclassified authority edges.
A partial inventory cannot authorize shadow comparison or migration.

## Canonical comparison unit

A later dual-read comparator must consume one immutable, versioned comparison
input containing:

- exact source observation and provenance digest;
- exact eight-drive sample and schema version;
- exact bounded candidate set and candidate identities;
- exact prior v4 selection/lifecycle snapshot;
- exact legacy goal-state snapshot or canonical digest plus structural manifest;
- logical step, cooldown state, and transition predicate versions;
- exact implementation and mapping versions;
- explicit authority fields fixing legacy as authoritative and v4 as
  shadow-only.

Legacy and v4 outputs must be normalized into a comparison record without
forcing false equality. The record must distinguish:

```text
exact_equivalent
mapped_equivalent
expected_design_difference
unexplained_divergence
legacy_only_behavior
v4_only_behavior
comparison_unavailable
```

Every non-exact ruling requires a versioned mapping rule and evidence. Unknown
or unclassified differences fail closed as `unexplained_divergence`.

## Dual-read execution order

A future production-origin shadow comparator must preserve this order:

1. capture one immutable input and legacy before-state;
2. execute the legacy goal path exactly once as the sole behavior authority;
3. record actual legacy after-state through before/after values or canonical
   digest plus structural manifest;
4. evaluate the v4 goal-selection and lifecycle path read-only against the same
   input;
5. prohibit the v4 result from action, scheduling, speech, memory mutation,
   affect mutation, or legacy mutation;
6. emit one canonical comparison candidate;
7. validate mapping, provenance, state fidelity, and replay;
8. retain the comparison only through a separately reviewed bounded evidence
   path;
9. leave legacy behavior and persistence authoritative regardless of verdict.

No retry loop, silent repair, best-effort fallback, or duplicate-as-success is
allowed.

## Required migration evidence package

A later migration-candidate decision must bind all of the following:

1. exact domain inventory and zero unclassified authority edges;
2. exact legacy-to-v4 state and lifecycle mapping tables;
3. versioned comparison schema and deterministic canonicalization;
4. bounded production-origin dual-read observation window;
5. raw observations or immutable content-addressed references sufficient to
   recompute every metric;
6. actual before/after state evidence and transition hashes;
7. exact counts for each comparison verdict;
8. zero unexplained divergence in the accepted window;
9. replay equivalence for retained v4 lifecycle history;
10. restart, recovery, corrupt-state, and corrupt-snapshot behavior;
11. verified rollback that restores legacy-only behavior authority without
    deleting v4 evidence;
12. verified preservation of legacy persistence as read-only migration evidence;
13. storage growth and bounded backup results on the private device;
14. exact-head focused/full/forward/M2-E validation pins;
15. explicit human-reviewed authorization limited to the named goal domain;
16. explicit false flags for action, scheduler, speech, legacy deletion, and
    M3-E unless separately authorized.

Machine-green checks or elapsed time alone cannot satisfy the human decision.

## Transfer packet

Any future authority transfer must be a separate append-only decision artifact,
not a boolean default or interpretation of this design. The packet must bind:

```text
schema: eve.m3-c-goal-domain-authority-transfer.v1
exact implementation head
exact migration-evidence package digest
exact legacy/v4 owner identities
exact domain and excluded edges
cutover and rollback commands
legacy preservation policy
observation-window metrics and raw-evidence references
human reviewer identity and explicit decision
activation time or event boundary
all downstream authority flags
```

Missing, malformed, mismatched, stale, or unreviewed fields fail closed. An
authority-transfer artifact cannot authorize action, scheduling, speech, affect
cutover, M3-E, or deletion unless those authorities are separately and
explicitly granted.

## Rollback and preservation

Rollback must be possible without rewriting history:

1. disable any v4 goal-domain behavior routing;
2. restore legacy as the sole behavior authority;
3. preserve v4 lifecycle events, comparison records, failed databases, and
   decision artifacts as evidence;
4. verify legacy state and persistence continuity;
5. verify no action, scheduler, speech, memory, or affect side effect escaped
   during the candidate period;
6. require a new separately reviewed transfer packet before reactivation.

Legacy files and sidecars must not be deleted at transfer time. Deletion, if
ever considered, is a later independent retention decision after a separately
specified preservation interval and restore proof.

## Failure matrix

| Failure | Required result |
|---|---|
| incomplete authority inventory | remain legacy-authoritative; no runtime hook |
| unclassified legacy/v4 mapping | fail comparison; no migration candidate |
| source or before-state mismatch | discard candidate; preserve evidence |
| unexplained divergence | fail window acceptance; remain legacy-authoritative |
| v4 mutation or downstream effect during shadow | disable candidate path; preserve evidence |
| replay or state-fidelity mismatch | fail closed; require defect correction and new tree |
| database, snapshot, or chain corruption | disable candidate path; recover separately |
| rollback verification failure | remain disabled; no authority transfer |
| artifact loss or digest mismatch | invalidate dependent decision evidence |
| missing explicit human transfer decision | legacy remains authoritative |

## Promotion ladder

The later migration work is split into separately reviewed slices:

1. **M3-C-K migration-gate design** — this document only.
2. **M3-C-L read-only dual-read comparator preflight** — synthetic or immutable
   fixture inputs only; no production hook, persistence write, or authority.
3. **M3-C-M dormant production-origin shadow tap** — code path exists but is
   unreachable by default and requires absent exact-reviewed authorization.
4. **M3-C-N bounded private-device dual-read observation window** — legacy acts;
   v4 observes only; zero unexplained divergence and verified rollback required.
5. **M3-C-O explicit goal-domain authority-transfer decision** — separate
   append-only human decision; no automatic transfer from earlier success.
6. **M3-C-P bounded transfer implementation and rollback rehearsal** — only
   within the exact accepted packet; legacy evidence preserved.

Acceptance of one slice is not authorization for the next. M3-D remains a
separate milestone. M3-E remains closed.

## Immediate next target

After this design is accepted, the only eligible implementation target is
**M3-C-L read-only dual-read comparator preflight** using synthetic or immutable
fixture inputs. It must not import or call production runtime orchestration,
open the private M3-C-J database, install a writer, mutate legacy goal state,
route actions, schedule work, generate speech, or open M3-E.

## Acceptance ruling

M3-C-K is accepted only as a fixed documentation boundary. It records that the
M3-C-J observation window is complete and reusable evidence while preserving
all private artifacts and every still-closed authority. It does not itself
create a migration candidate or transfer any authority.
