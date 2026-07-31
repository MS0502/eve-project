# M3-C-M Dormant Production-Origin Shadow Tap

## Status

Dormant implementation slice after the accepted M3-C-L pure comparator. The
active legacy goal call sites can host an in-memory shadow wrapper, but the
wrapper is unreachable by default. No implementation pin or authorization pin
is committed, constructed, loaded, or injected by the default engine.

```text
production call-site definitions present: true
default engine tap instance: false
implementation pin committed or active: false
authorization pin committed or active: false
tap reachable by default: false
legacy authoritative call count: exactly one
legacy goal authority: true
v4 goal authority: shadow only
state capture by default: false
v4 evaluation by default: false
comparison by default: false
retention / persistence write: false
event append: false
private database access: false
action / scheduler / speech authority: false
legacy migration or authority transfer: false
M3-E authority open: false
```

## Exact prerequisite

```text
PR:                    #236
base:                  d9c1cf8f615872b6a59ea7e950ccb9ceeb629133
exact head:            73e2fdcf9e4006c726a27304f39c7efb0826bc9f
exact run:             30620402653
focused:               16 passed
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,348 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-73e2fdcf9e4006c726a27304f39c7efb0826bc9f
artifact SHA-256:      44678184335a8a5f4c25efd3b0e7085914554e7bf54bdc81409b1b966606e065
M2-E run:              30620402663
M2-E:                  6/6 passed
merge SHA:             dd524a820a58947f0b589cd0cd521ee35eda73da
```

The artifact digest was independently recomputed from the downloaded ZIP and
matched the GitHub artifact digest. This evidence is immutable and reused. PR
#235 and every earlier phone, retained-sequence, readiness, private-device,
database, journal, bundle, backup, and restore execution remain single-use and
must not be rerun because work continued in another chat, shell, branch, or PR.

## Purpose

M3-C-K requires production-origin dual-read evidence before any goal-domain
migration candidate. M3-C-L supplied the pure comparator on fixtures only.
M3-C-M adds only the dormant, non-retaining execution wrapper and the exact two
legacy call-site seams needed by a later separately authorized bounded window:

1. `GoalAdapter.observe_meaning` to `GoalManagement.goal_set`;
2. `GoalAdapter.tick` to `GoalManagement.tick`.

This slice does not start an observation window. It deliberately ships without
an exact M3-C-M implementation pin and without an M3-C-N authorization pin.

## Default behavior

`main.build_full_engine()` continues to construct `GoalAdapter` without a tap.
When `production_origin_shadow_tap` is absent, the adapter directly invokes the
legacy callable exactly once. The M3-C-M module is imported lazily only after a
tap object is explicitly injected.

A tap with any missing component also remains dormant:

- missing implementation pin;
- missing authorization pin;
- missing legacy mapping table;
- missing v4 evaluator;
- malformed evaluator identity;
- any exact digest or manifest mismatch.

In all such cases, the authoritative legacy callable still executes exactly
once. No before-state capture, after-state capture, v4 evaluation, comparator
call, retry, retention, or side effect occurs.

## Exact-reviewed activation contract

A future M3-C-N operator package would have to supply all four objects together:

1. `ShadowTapImplementationPin` bound to the accepted exact M3-C-M head, run,
   artifact digest, merge SHA, comparator version, and call-site manifest;
2. `ShadowTapAuthorizationPin` bound to that implementation pin, the exact
   legacy mapping table digest, exact v4 evaluator digest, a reviewed
   authorization artifact, and the M3-C-N shadow-only scope;
3. `LegacyGoalMappingTable` with one exact tuple for each observed legacy goal
   code, category digest, and legacy status;
4. a v4 evaluator with an exact reviewed evaluator digest.

The authorization pin rejects persistence, event append, action, scheduling,
speech, migration, authority transfer, or M3-E flags. No environment variable,
default boolean, elapsed time, machine-green check, or prior window success can
replace these exact objects.

## Production-origin execution order

Only after all pins bind exactly does one call follow this order:

1. capture a canonical before-state snapshot of the in-memory legacy owner;
2. invoke the legacy callable exactly once as the sole behavior authority;
3. capture the canonical after-state snapshot;
4. reject structural-manifest drift;
5. resolve the exact legacy mapping tuple;
6. evaluate the v4 B/C kernels read-only against the same comparison identity;
7. call the accepted M3-C-L comparator;
8. return the comparison receipt in memory only.

There is no retry. A before-capture failure falls back to one legacy call. Any
failure after the legacy call preserves its result and returns a blocked shadow
execution without repeating or undoing legacy behavior.

## Snapshot and privacy boundary

The state snapshot includes deterministic state and structural digests,
active-count, and the top goal category digest/status. Raw goal identifiers,
categories, sources, reasons, and history text are SHA-256 transformed before
canonicalization. The wrapper exposes no filesystem or database path and holds
no receipt history.

Snapshot capture does not call `GoalManagement.get_state()`, `active_goals()`,
progress evaluation, lifecycle mutation, hormone stimulation, or persistence.
It reads the already-existing in-memory state only.

## No retention and no authority

`ShadowTapExecution` is a frozen one-call result. It fixes these facts to false:

```text
event append
persistence write
action authorization
scheduler authorization
speech authorization
legacy goal authority transfer
legacy migration authorization
M3-E authority
```

The tap object has no observation list, queue, journal, database, output path,
or retry buffer. Even an `exact_equivalent` comparison remains ephemeral and
cannot count toward an observation window until M3-C-N separately defines and
reviews a bounded evidence-retention path.

## Failure matrix

| Condition | Result |
|---|---|
| default engine or no tap | direct legacy call exactly once |
| missing implementation pin | legacy once; no capture or evaluation |
| missing authorization pin | legacy once; no capture or evaluation |
| missing mapping or evaluator | legacy once; no capture or evaluation |
| exact pin or evaluator mismatch | legacy once; no capture or evaluation |
| before capture failure | legacy once; blocked shadow result |
| after capture failure | preserve legacy result; no v4 evaluation |
| structural manifest drift | preserve legacy result; block comparison |
| exact mapping absent | preserve legacy result; no v4 evaluation |
| v4 or comparator failure | preserve legacy result; no retry or retention |
| successful comparison | return in memory only; legacy remains authoritative |

## Focused proof

The focused suite proves:

1. the call-site manifest contains only `goal_set` and `tick`;
2. the default engine injects no tap;
3. absent or mismatched pins run legacy exactly once with zero observation;
4. the authorized synthetic path uses genuine M3-C-B and M3-C-C receipts and
   the accepted M3-C-L comparator;
5. missing legacy mapping fails after one legacy call and before v4 evaluation;
6. snapshots are deterministic and contain category digests rather than raw
   private goal text;
7. any downstream authority in an authorization pin fails closed;
8. `GoalAdapter` preserves exactly one legacy invocation with and without a
   dormant injected tap;
9. the core module has no I/O, network, SQLite, retention, action, scheduling,
   or speech surface;
10. the PR #236 reuse pin prevents duplicate prerequisite execution across chat
    and shell changes.

## Promotion boundary

The next separately reviewed target is **M3-C-N bounded private-device dual-read
observation window**. M3-C-N must create a new exact implementation pin from the
accepted M3-C-M evidence, a separate human-reviewed shadow-only authorization
artifact, a complete reviewed mapping/evaluator package, bounded retention and
rollback rules, and a private-device operator command.

M3-C-M acceptance alone does not authorize M3-C-N execution. It does not permit
access to the existing M3-C-J database or any retained phone artifacts, and it
does not transfer goal authority or open M3-E.

## Acceptance ruling

M3-C-M is accepted only as dormant, unreachable-by-default code with exact
fail-closed pin contracts and no retention. Legacy remains the sole goal-domain
behavior authority under every default and failure path.
