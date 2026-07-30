# M3-C-J Private-Device Operator Preflight

## Status

This slice defines and tests the single-use private-device command required to
produce a real bounded M3-C-J observation bundle. The checked-in operator
implementation and authorization pins remain absent, so the command cannot pass
its first authorization gate on main after this preflight alone.

```text
reviewed operator implementation pin: absent
reviewed operator authorization digest: absent
private production database accessed by this PR: false
production append executed by this PR: false
real M3-C-J window started by this PR: false
phone command issued by this PR: false
```

## Reused exact prerequisite

The exact-reviewed observation evaluator authorization from PR #229 is reused
without rerun:

```text
PR:             #229
exact head:     532c595158ee68eb3268f75414bf6eaa23a79ffb
exact run:      30451436253
focused:        11 passed
full:           3,315 passed
forward gate:   0 / 0 / 0 / 0
artifact:       exact-head-validation-532c595158ee68eb3268f75414bf6eaa23a79ffb
artifact SHA:   e488f98d0d60a4572ea1f64c383ee8f3a0d91d23b22477c431695b16e9d9d12d
M2-E run:       30451436272
M2-E:           6/6 passed
merge SHA:      361ed88be399ed7650a946b58e713bc14253384e
```

PR #228, PR #227, PR #225, the #211 phone witness, and retained sequences 1
through 5 remain immutable prerequisites and are not rerun.

## Single-use invariant

The operator requires the reviewed database file and all SQLite sidecars to be
absent before initialization. Once any execution creates the database, a second
execution fails before a second append. The command never deletes or repairs an
existing database to make a retry possible.

This is the cross-chat duplicate-execution boundary:

```text
new chat:                    not an invalidator
new operator session:        not an invalidator
existing reviewed DB file:   hard rerun refusal
existing WAL/SHM/journal:    hard rerun refusal
existing output journal:     hard rerun refusal
existing private bundle:     hard rerun refusal
```

A failed or partial database is preserved as evidence and requires a separately
reviewed recovery decision.

## Operator authorization packet

The future exact-reviewed packet binds:

- the exact operator implementation head;
- PR #229 exact validation, artifact, M2-E run, and merge SHA;
- the exact reviewed observation evaluator packet;
- the exact reviewed bounded writer packet;
- the private production database-path digest, not its plaintext path;
- the 32-event observation-window cap;
- exactly four lifecycle events for one candidate:
  `absent -> proposed -> validated -> eligible -> selected`;
- explicit human review;
- a required empty baseline, backup, writer disable, and separate restore;
- explicit false values for runtime integration, action, scheduling, speech,
  legacy migration, legacy goal-domain transfer, and M3-E.

The packet permits four bounded production-path appends only after its own exact
review pin exists and the operator supplies all private materials explicitly.

## Private goal input

The canonical private input contains one reviewed `GoalCandidate` and exactly one
`DriveSample` for each of the eight M3-A drives in canonical order. The candidate
must deterministically produce an initial selection and the exact four named
lifecycle transitions.

The input also asserts:

```text
candidate human reviewed:                 true
drive samples human reviewed:             true
new window material:                      true
#211 phone witness replayed:               false
retained sequences replayed:              false
legacy goal authority transfer requested: false
M3-E requested:                            false
```

Raw candidate, drive samples, nonce, and path plaintext remain in the private
bundle. The public review contains cryptographic bindings and bounded receipts,
not the raw input.

## Exact execution sequence

After a later operator pin, the command performs only this sequence:

1. verify the exact clean repository head;
2. build and verify the exact operator, window, and writer packets in memory;
3. resolve explicit private paths outside the repository;
4. refuse any existing DB, sidecar, output, backup, restore, or journal;
5. parse canonical private input and private nonce;
6. create an empty SQLite database with the reviewed bounded policy;
7. verify zero events, zero snapshots, and the genesis chain;
8. create and verify baseline backup ordinal 1;
9. derive one exact four-transition binding chain;
10. append four events with the reviewed bounded writer;
11. verify final database integrity and reducer replay;
12. apply the reviewed writer-disable rollback control;
13. restore the baseline backup into a separate directory;
14. verify the restored empty baseline;
15. evaluate the complete M3-C-J window;
16. atomically write private bundle, public review, and completion journal.

The command does not build the full conversational engine, poll a phone runtime,
replay prior witnesses, run retained sequences, install a startup hook, or mutate
legacy goal authority.

## Output files

All output paths remain outside the repository with private permissions.

```text
m3_c_j_operator_journal_private_v1.json
m3_c_j_private_device_bundle_v1.json
m3_c_j_public_review_v1.json
backups/shadow-backup-00000001.sqlite3
restore/goal_lifecycle_baseline.sqlite3
```

The journal is written before database access with stage
`authorized_before_database_access` and replaced only after all checks pass with
stage `complete_writer_disabled_restore_verified`.

## Focused-test boundary

Focused tests use pytest temporary paths and test-only exact pins. They prove:

- canonical input round-trip and HMAC binding;
- exact four-transition derivation;
- rejection of below-threshold and scope-escape inputs;
- empty baseline, verified backup, four appends, writer disable, final integrity,
  evaluator acceptance, and separate restore;
- second-run refusal with the first four events preserved;
- wrong path-digest refusal before database creation;
- absent checked-in operator pins and import-time side-effect freedom.

Temporary-path receipts are not retained production observations and cannot close
the real M3-C-J window.

## Promotion ladder

1. Merge this preflight after its own exact-head validation.
2. Create a separate exact-reviewed operator implementation/digest pin.
3. Review a concrete private input file and exact command packet.
4. Execute that command exactly once on the private device.
5. Retain and review the public output and its private companion digest.
6. Pin the accepted window receipt in a separate repository PR.

No later step may rerun #211 or retained sequences 1 through 5.
