# M3-C-Q Local Goal-Window Authorization Capture

## Status

This slice adds the local-only capture harness required to turn one already
canonical, human-reviewed M3-C-O private package into one single-use private
M3-C-P authorization-pin file. It does not install that pin or execute the
operator.

```text
canonical private package required: true
explicit local human review required: true
single-use private pin output: true
active local authorization installed: false
operator pin opened: false
operator execution: false
actual private-device observation: false
default runtime integration: false
existing M3-C-J database access: false
raw private text/path public: false
legacy goal authority transfer: false
legacy migration authorization: false
action / scheduler / speech authorization: false
M3-E authority: false
```

## Accepted prerequisites

PR #240 accepted the exact M3-C-O implementation evidence and the default-absent
local authorization schema.

```text
PR #240 exact head:     b3c46051c5b6a57f794989e4c5fb896798843cb2
exact run:              30645219048
focused:                7 passed
M0:                     byte-identical
M2-B:                   valid; errors 0
full:                   3,384 passed
forward:                0 / 0 / 0 / 0
artifact SHA-256:       acd59130586981e3e657114ce80e5beb7a56758944d22a555e466f312b67c811
M2-E:                   30645219174; 6/6 passed
squash merge:           21783116020289fa891ef37837a828331bf7a6d4
```

The branch begins with
`docs/audit/M3_C_P_PR240_VALIDATION_REUSE_PIN.json`. PR #240, PR #239, PR #238,
phone witness #211, retained sequences 1–5, controlled-input readiness, the
completed private-device operator, and all existing DB/sidecar/journal/bundle/
backup/restore evidence are immutable. Chat, shell, branch, PR, Draft/Ready,
review, or descendant-work changes do not authorize reruns.

## Input contract

The explicit command accepts:

- a clean exact checkout head;
- one absolute private package path outside the repository;
- the independently reviewed canonical package digest;
- one absolute new private pin-output path outside the repository;
- explicit `--reviewed` confirmation.

The private package must already satisfy every M3-C-O canonical schema and
binding rule. The command does not build or repair a package and does not infer
human review from file existence.

Before reading the package, the command requires:

- a clean exact Git checkout;
- explicit `--reviewed`;
- no active M3-C-P local authorization pin;
- both M3-C-O module pin seams closed.

The package file must be private on POSIX systems. Its canonical digest must
match `--expected-package-digest` exactly.

## Capture operation

After package validation, the command:

1. derives an `M3CPOperatorAuthorizationBinding` containing only exact digests;
2. builds one `M3CPLocalReviewedAuthorizationPin` bound to accepted PR #239
   implementation evidence;
3. requires the private output parent to already exist;
4. refuses any existing output;
5. creates the output with exclusive mode;
6. writes canonical JSON and fsyncs it;
7. applies mode 0600 on POSIX;
8. reads the file back and reconstructs the exact pin;
9. verifies the pin and pin digest;
10. emits a public-safe capture receipt containing digests only.

There is no overwrite, repair, delete, retry, activation, engine build, operator
call, event append, action, scheduling, speech, authority transfer, migration,
or M3-E work.

## Private pin artifact

The private artifact contains the exact M3-C-P pin mapping, including reviewer
identity and all constituent digests. It contains no raw goal category, goal
text, private path, drive sample, candidate contents, mapping contents, baseline
contents, or existing private evidence.

The output is single-use. A completed or partial first attempt leaves the path
occupied. A later chat or shell cannot authorize deleting or overwriting it.

## Public-safe receipt

The printed receipt contains:

```text
accepted implementation head and evidence digest
binding and authorization digests
package/review/path/rollback/mapping/evaluator digests
reviewer-id digest
pin digest
private output path digest
private pin file SHA-256
launch repository head
```

It explicitly records:

```text
active local authorization installed: false
operator pin opened: false
operator executed: false
existing M3-C-J database accessed: false
raw private text/path public: false
legacy authority transferred: false
legacy migration authorized: false
action / scheduler / speech authorized: false
M3-E authority open: false
```

The plaintext private path and reviewer identity are not printed.

## Focused-test boundary

Focused tests use synthetic digest strings and pytest temporary paths only. They
prove:

- the pin binds accepted implementation evidence;
- every downstream authority remains false;
- canonical exclusive output and readback verification;
- POSIX private permissions;
- no plaintext output path or reviewer identity in the public receipt;
- missing human review fails before a write;
- an existing output refuses overwrite;
- relative paths and missing parents fail closed;
- extra fields and authority escape fail closed;
- the explicit script never installs a pin, opens a session, builds an engine,
  or executes the M3-C-O operator;
- PR #240 reuse remains durable.

These tests are not local human review and are not private-device evidence.

## Promotion boundary

After this capture harness is exact-head validated and merged, one real private
device may use it to create the concrete local pin artifact. The artifact and
public-safe capture receipt must then be reviewed before a later isolated PR can
check in the exact public digest pin. That later PR still must not execute the
operator.

The actual M3-C-O command remains a separate one-time descendant action after
that exact public pin is merged. Any completed or partial run remains immutable
and requires a separate human-gate receipt review before migration can even be
proposed.

## Acceptance ruling

M3-C-Q is acceptable only as a local authorization artifact capture harness. It
creates no active authorization, performs no observation, preserves legacy as
sole goal authority, and keeps migration and all downstream authority closed.
