# M3-C-P Private-Device Goal Window Authorization Pin Preflight

## Status

This slice accepts the exact M3-C-O implementation evidence from merged PR #239
and defines the only admissible local authorization binding and scoped pin
session. It does not check in a concrete private package, local review digest,
path binding, rollback digest, active authorization pin, or launch command.

```text
accepted M3-C-O implementation evidence: true
active local authorization: false
concrete private binding: false
checked-in M3-C-O operator pins open: false
actual private-device execution: false
default runtime integration: false
existing M3-C-J database access: false
raw private text/path retention: false
legacy goal authority transfer: false
legacy migration authorization: false
action / scheduler / speech authorization: false
M3-E authority: false
```

## Accepted PR #239 evidence

```text
PR:                    #239
base:                  9a26f6040679013066425887c3bcee5a2846a025
exact head:            57da278ce01e04257efc8a84933092715b371dec
exact run:             30643858724
focused:               8 passed
M0 invariance:         byte-identical
M2-B:                  valid; errors 0
full:                  3,377 passed
forward gate:          0 / 0 / 0 / 0
artifact:              exact-head-validation-57da278ce01e04257efc8a84933092715b371dec
artifact SHA-256:      097f8025b587bd77156eb966fb4cbf584f0a437b2fdbbbfaff1d7c4200a88068
artifact verification: GitHub artifact digest == independently downloaded archive hash
M2-E run:              30643857677
M2-E:                  6/6 passed
squash merge:          f0a01b8e138dd1111c323dd54bf92c8527eb5b30
main after merge:      identical
```

`docs/audit/M3_C_O_PR239_VALIDATION_REUSE_PIN.json` is the durable cross-chat
reuse record. PR #239 full suite and M2-E must not run again because a chat,
shell, branch, PR, Draft/Ready, review, or descendant-work state changed. PR
#238 and all earlier phone, retained-sequence, readiness, private-device,
database, sidecar, journal, bundle, backup, and restore evidence remain equally
immutable.

## Why this is a separate slice

M3-C-O deliberately left both checked-in operator pins absent. Its promotion
boundary requires a later exact pin to bind all of these values at once:

- accepted M3-C-O implementation head and artifact;
- one concrete canonical private package digest;
- one local human-review artifact digest;
- one exact private path-binding digest;
- one exact rollback digest;
- one accepted mapping digest;
- one reviewed evaluator digest;
- one reviewer identity and authorization digest.

Only the first item is public and available after PR #239. The remaining values
must originate from one real local review on the private device. Fabricating or
checking in placeholders would create a false authorization, so this preflight
pins implementation evidence while leaving the concrete local pin absent.

## Immutable implementation evidence

`M3COImplementationEvidence` fixes the exact PR #239 base, head, validation run,
focused/full counts, M0/M2-B/forward results, M2-E run, artifact name and digest,
and squash merge SHA. Construction fails if any field differs from the accepted
record.

Its digest becomes the implementation-evidence root for every future local pin.
A private authorization cannot target a branch tip, merge SHA, projection, or
artifact other than the accepted exact M3-C-O implementation head and archive.

## Exact private binding

`M3CPOperatorAuthorizationBinding` contains only public-safe digests and identity:

```text
implementation head
authorization digest
canonical private package digest
local review artifact digest
private path-binding digest
rollback digest
legacy mapping digest
reviewed evaluator digest
reviewer id
```

The binding contains no raw private category, goal text, path, probe material,
drive sample, mapping contents, baseline contents, or existing private evidence.
`binding_from_private_package()` may derive this digest envelope only after the
M3-C-O package has already passed its canonical schema and internal binding
checks.

## Local reviewed authorization pin

`M3CPLocalReviewedAuthorizationPin` must bind:

- the accepted implementation-evidence digest;
- the exact private binding digest;
- every constituent digest independently;
- the same reviewer identity;
- single-use private-device shadow-observation scope only.

It rejects every downstream grant:

```text
existing M3-C-J path reuse
raw private text/path publication
legacy goal authority transfer
legacy migration
action
scheduler
speech
M3-E
```

The checked-in value remains:

```text
_ACTIVE_LOCAL_REVIEWED_AUTHORIZATION_PIN = None
```

Therefore this preflight cannot activate M3-C-O.

## Scoped pin session

A later isolated local-review pin may use `reviewed_operator_pin_session()` only
after its exact binding matches the active local pin. The session:

1. verifies every binding field;
2. refuses if either M3-C-O pin seam is already open;
3. opens the accepted implementation head and exact authorization digest;
4. yields to one synchronous caller;
5. restores both prior values in `finally`.

The session performs no file, database, network, engine, operator, event, action,
scheduler, speech, migration, or M3-E operation. This PR does not connect the
session to the explicit operator command.

## Duplicate-test boundary

The branch began with the merged PR #239 reuse pin before new implementation
work. Consequently:

- #239 exact validation/full/M2-E are not rerun for this PR;
- #238 exact validation/full/M2-E are not rerun;
- phone witness #211 is not rerun;
- retained sequences 1–5 are not rerun;
- controlled-input readiness is not rerun;
- the completed private-device operator is not rerun;
- existing DB/sidecar/journal/bundle/backup/restore evidence is not opened;
- only new M3-C-P code receives focused and final-head validation.

Discovery heads must stop before full suite until all new M2-B decisions and
forward additions are registered. The final fully registered head may run its
new full suite once.

## Focused-test boundary

Focused tests use synthetic digest strings and monkeypatch only. They prove:

- PR #239 implementation evidence is exact and complete;
- the checked-in local authorization pin remains absent;
- both M3-C-O operator pins remain absent by default;
- every downstream authority bit fails closed;
- every private binding digest must match;
- the session opens exactly two pins and restores them;
- exceptions restore both pins;
- an already-open seam refuses reentry;
- the module has no I/O, database, subprocess, network, engine-build, or operator
  execution surface;
- PR #239 reuse remains durable across chat and shell changes.

These tests are not local review and are not private-device evidence.

## Next promotion boundary

The next admissible slice requires one real locally reviewed canonical private
package. It may add exactly one concrete `M3CPLocalReviewedAuthorizationPin`
bound to that package and the accepted implementation evidence. That pin PR
must still not issue the operator command.

Only after the pin is exact-head validated and merged may one separate clean
descendant checkout launch the M3-C-O command once. A completed or partial
attempt is immutable and cannot be retried by changing chat, shell, branch, or
paths. Its public digest receipt must undergo a separate human-gate review before
any migration candidate is proposed.

## Acceptance ruling

M3-C-P is acceptable only as an implementation-evidence and authorization-schema
preflight. It preserves the absent local pin, performs no private observation,
leaves legacy as sole goal authority, and keeps migration and all downstream
authority closed.
