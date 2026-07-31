# M3-C-N Bounded Private-Device Goal Dual-Read Window Preflight

## Status

Pure implementation and contract preflight after accepted M3-C-M. This slice
makes the bounded private-device observation window reviewable but not
executable.

```text
accepted M3-C-M evidence fixed: true
raw 40-character exact head and merge SHA bound: true
M3-C-M v1 compatibility pin projection defined: true
bounded window policy defined: true
private path separation contract defined: true
rollback contract defined: true
digest-only record chain defined: true
human-gate review receipt defined: true
active authorization checked in: false
operator command checked in: false
execution available in this slice: false
default runtime integration: false
existing M3-C-J database access: false
raw text retention: false
legacy goal authority transfer: false
legacy migration authorization: false
action / scheduler / speech authorization: false
M3-E authority: false
```

## Accepted prerequisite

```text
PR:                    #237
base:                  dd524a820a58947f0b589cd0cd521ee35eda73da
exact head:            ca9e8a13ae0308060fa0c0505a2b1b4a6558b3a4
exact run:             30635460387
focused:               10 passed
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,358 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-ca9e8a13ae0308060fa0c0505a2b1b4a6558b3a4
artifact SHA-256:      9b72851c017af41c6c8d423d3ccd79a94b8a4fe10e4ee432f0b64750e0c0117c
M2-E run:              30635460203
M2-E:                  6/6 passed
merge SHA:             1ebd3e27ad4582c67b9b2f072ebd58c625af2057
main after merge:      identical
```

The downloaded artifact ZIP was independently hashed before #237 merge. That
accepted evidence is immutable and is not rerun because work moves to another
chat, shell, branch, PR, Draft/Ready state, review, or metadata update.

## M3-C-M v1 pin compatibility correction

`ShadowTapImplementationPin` v1 named two fields `exact_head` and `merge_sha`
but validates them as 64-character SHA-256 values. Real Git commit identities
in this repository are 40-character hexadecimal SHAs. Constructing the v1 pin
with the actual accepted head or merge SHA therefore fails.

M3-C-N does not weaken the v1 validator or silently put a different value in the
raw evidence. `AcceptedM3CMImplementationEvidence` stores and validates the real
40-character base, exact head, and merge SHA, then derives a wire-compatible v1
pin by SHA-256-digesting the two Git identities. Future authorization binds both:

1. the raw accepted-evidence digest containing the actual Git SHAs; and
2. the compatibility shadow-pin digest consumed by the M3-C-M tap.

This closes the identity gap without changing accepted #237 code or pretending
a 64-character digest is a Git commit SHA.

## Window policy

The reviewed default policy is:

```text
minimum observations:                 4
maximum observations:                 16
maximum unexplained divergences:      0
maximum unavailable comparisons:     0
single-use private device only:       true
exact sequence and digest chain:      true
raw text retention:                   false
existing private database access:     false
```

All seven M3-C-L verdicts may be retained because unexplained or unavailable
behavior must remain visible. The following verdicts block human migration-gate
review eligibility:

- `unexplained_divergence`
- `comparison_unavailable`
- `legacy_only_behavior`
- `v4_only_behavior`

A clean window becomes eligible only for a later human review of the M3-C-K
gate. Eligibility is not migration authorization and does not transfer goal
authority.

## Private path separation

The future private-device operator must supply digest bindings for four new,
distinct local paths:

1. reviewed operator input;
2. new empty M3-C-N working store;
3. baseline backup;
4. separate restore target.

It must also supply a sorted set of prior private path digests. Every active
M3-C-N path must differ from every prior digest. This specifically prevents
reuse or access of the M3-C-J database, its sidecars, journal, bundle, backup,
or restore paths. Raw private paths are never checked in.

## Rollback contract

Rollback requires all of the following:

1. disable the shadow tap before rollback;
2. preserve only the public review bundle;
3. delete the new working store after review;
4. restore only into a separate path;
5. never rewrite legacy goal state.

The rollback plan binds the path-binding digest and the baseline legacy state
digest. It grants no runtime, action, scheduler, speech, migration, or M3-E
authority.

## Retained record contract

Only an actual `ShadowTapExecution` with all of these properties may become a
window record:

```text
status:                         comparison_ready_in_memory_only
legacy authoritative calls:     exactly 1
state capture performed:         true
v4 evaluation performed:        true
comparison performed:           true
M3-C-L receipt present:          true
event append by tap:             false
persistence write by tap:        false
```

The retained record contains only digests, sequence, verdict, and fixed false
authority flags. It retains no raw goal text, raw path, private input, candidate
text, evaluator source, or mapping contents. Records form a strict chain from a
fixed zero genesis digest. Duplicate execution or production-observation
digests fail closed.

## Authorization packet

A later private-device packet must bind:

- the exact future window implementation head;
- accepted #237 raw evidence digest;
- M3-C-M v1 compatibility pin digest;
- reviewed legacy mapping digest;
- reviewed v4 evaluator digest;
- window policy digest;
- private path-binding digest;
- rollback digest;
- human-reviewed authorization artifact digest.

It may authorize only single-use private-device shadow observation and bounded
private digest retention. It must keep these false:

```text
existing private database access
default runtime integration
raw text retention
action
scheduler
speech
legacy goal authority transfer
legacy migration authorization
M3-E authority
```

No authorization packet is active in this slice. Both checked-in active pin
slots remain `None`.

## Operator manifest

The pure module publishes the future operator manifest only. It names the
future entrypoint and required private-local inputs but deliberately marks
execution unavailable. No script, CLI, filesystem access, database access,
network access, event append, or production engine injection is included.

The later operator must require:

- reviewed authorization JSON;
- reviewed legacy mapping JSON;
- reviewed v4 evaluator package;
- new private operator input JSON;
- a new empty window-store path;
- separate backup and restore paths;
- prior private path digests.

## CI and private-device boundary

CI proves schemas, exact evidence, compatibility projection, bounds, path
separation, rollback, digest-only record conversion, verdict accounting, chain
validation, and absence of I/O/operator surfaces using synthetic immutable
objects only.

CI must not:

- inject the tap into `main.build_full_engine()`;
- open or simulate the user's existing M3-C-J database;
- rerun phone witness #211;
- rerun retained sequences 1–5;
- rerun readiness or completed private-device operators;
- create a new production observation and claim it is private-device evidence;
- activate or pin the future operator.

## Promotion boundary

The next target is **M3-C-O exact-reviewed private-device operator pin and
command**. M3-C-O must use the accepted M3-C-N exact head and artifact, create a
separate explicit human-reviewed authorization artifact, implement the dormant
single-use command, and prove on disposable synthetic paths that it blocks
existing-path reuse before touching any private location.

M3-C-O implementation acceptance still will not prove a real observation
window. Actual private-device execution must occur once on the user's device
with new reviewed local material and must produce separately reviewed public
receipts before any M3-C-K migration-gate candidate can exist.

## Acceptance ruling

M3-C-N is acceptable only as a pure, non-executable preflight. It closes the
M3-C-M Git-identity pin gap, defines bounded digest-only evidence and rollback,
and leaves every behavior and authority boundary closed.
