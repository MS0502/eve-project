# M3-C-J Bounded Goal-Lifecycle Observation-Window Preflight

## Status

This slice implements the deterministic, dormant evaluator required before an
M3-C-J production-path observation window can start. It does not pin the
evaluator's own reviewed exact head, construct a production writer, access the
private database path, append a lifecycle event, execute an operator command, or
start the real window.

```text
reviewed window implementation pin: absent
reviewed window authorization digest: absent
real private database accessed: false
production lifecycle append executed: false
M3-C-J observation window started: false
legacy goal authority transferred: false
M3-E authority open: false
```

## Immutable M3-C-I prerequisite

The accepted M3-C-I tree is reused without rerun:

```text
PR:             #227
exact head:     bec44a796834e037c41fbb941d090de416cf1e23
exact run:      30447974882
focused:        16 passed
full:           3,304 passed
forward gate:   0 / 0 / 0 / 0
artifact:       exact-head-validation-bec44a796834e037c41fbb941d090de416cf1e23
artifact SHA:   650d11a611b9ae8dcf49fe540b117a26e49fedab5576c366f332eda9d7b92f0f
M2-E run:       30447974661
M2-E:           6/6 passed
merge SHA:      51f682e00059698cbb301a75983e11dd4812f574
```

Chat changes, operator-session changes, and PR metadata changes are not
validation invalidators. The first #227 discovery head never ran the full suite;
the final registered exact head ran it exactly once. PR #225, the #211 phone
witness, and retained sequences 1 through 5 were not rerun.

## Purpose

M3-C-I made the bounded writer reachable only when a private caller supplies the
exact reviewed packet, exact private absolute-path digest, and exact storage
policy. M3-C-J must not infer success merely because an append can occur. It must
accept a bounded window only when independent immutable evidence proves all of
the following:

1. a verified pre-window database baseline and backup exist;
2. the writer is disabled while the baseline is captured;
3. every observed event is an exact production-path writer receipt;
4. sequences, counts, and event-chain digests advance contiguously;
5. no append receipt, event envelope, or lifecycle transition identity repeats;
6. final database integrity, chain head, and reducer replay agree;
7. rollback preserves the failed database and restores the pre-window state into
   a separate path;
8. action, scheduling, speech, runtime integration, legacy migration, legacy
   goal-authority transfer, and M3-E remain false.

## Versioned contracts

```text
eve.m3-c-j.goal-lifecycle-observation-authorization.v1
eve.m3-c-j.goal-lifecycle-observation-baseline.v1
eve.m3-c-j.goal-lifecycle-rollback-preservation.v1
eve.m3-c-j.goal-lifecycle-observation-receipt.v1
```

## Authorization packet

The packet binds:

- the exact M3-C-J evaluator implementation head;
- the exact M3-C-I validation artifact and merge SHA;
- the reviewed M3-C-I writer authorization digest and implementation head;
- the private database-path digest, never its plaintext path;
- the exact bounded storage policy;
- a maximum of 32 observed lifecycle events;
- explicit human review;
- explicit false values for every authority outside observation.

The packet cannot activate from an environment variable, file presence, branch
name, import side effect, or heuristic. This preflight intentionally keeps both
checked-in active window pins `None`, so evaluation fails before reading any
evidence until a later exact-review pin slice.

## Baseline evidence

A real window baseline must be captured before the writer is enabled and must
pin:

```text
database path digest
starting stream sequence
starting total event count
starting event-chain digest
starting lifecycle reducer snapshot digest
integrity report digest
verified backup SHA-256
backup path digest
writer disabled: true
legacy goal authority transferred: false
M3-E authority open: false
```

## Accepted append evidence

Each receipt must be an immutable `DormantWriterAppendReceipt` from the exact
reviewed writer and private path. The evaluator rejects disposable or test-only
receipts. It also rejects any receipt that lacks commit, precommit readback,
postcommit readback, one-step sequence advance, verified chain advance, direct
reducer equivalence, or SQLite write proof.

The evaluator performs no append itself. `production_append_executed_by_evaluator`
therefore remains false even when it evaluates receipts produced by a separately
reviewed private caller.

## Zero-acceptance failure rule

The accepted window receipt fixes both counters at zero:

```text
duplicate_acceptance_count: 0
conflict_acceptance_count:  0
```

A repeated receipt, event-envelope digest, transition identity, sequence,
noncontiguous before/after count, or broken chain link fails closed. No retry,
silent repair, idempotent-success conversion, or event skipping is authorized.

## Replay and integrity rule

The final database integrity report must be valid, report the exact total event
count implied by the baseline plus the observed receipts, and expose the final
chain head from the last receipt. The independently supplied final reducer
snapshot digest must equal the last append receipt's verified replay digest.

## Rollback-preservation rule

Rollback evidence must prove:

- the writer is disabled;
- the failed database is preserved as evidence;
- the verified pre-window backup is the baseline backup;
- restoration occurs into a path distinct from both production and backup paths;
- restored schema and logical integrity are valid;
- restored replay is verified;
- restored reducer snapshot equals the pre-window baseline snapshot;
- legacy goal authority and M3-E remain unchanged.

## Focused-test boundary

Focused tests use only temporary directories and test-only monkeypatched pins.
They create an empty SQLite baseline, create a verified backup, append four
synthetic lifecycle transitions, verify the final chain and reducer digest, copy
the baseline backup into a separate restore path, and prove the restored stream
is the original empty baseline.

Those focused receipts are not retained production observations and cannot close
M3-C-J. The production private path, #211 phone witness, and retained sequences 1
through 5 are not read or replayed.

## Explicit exclusions

```text
production writer construction by evaluator: false
production database access by evaluator:      false
production append by evaluator:               false
runtime startup integration:                  false
action/scheduler/speech authority:            false
continuous drive or score persistence:        false
legacy goal-domain authority transfer:        false
legacy migration:                             false
M3-E authority:                               false
phone command in this preflight:               false
M3-C-J real window started:                   false
```

## Promotion ladder

1. Merge this dormant evaluator after its own exact-head validation.
2. Create a separate exact-reviewed pin for the evaluator head and authorization
   digest. That pin still performs no production append.
3. Only then create a narrowly scoped private-device operator packet that captures
   the baseline, enables only the bounded writer for named lifecycle transitions,
   collects at most 32 receipts, disables the writer, and verifies rollback.
4. Close M3-C-J only after the retained private-device evidence satisfies this
   evaluator with zero duplicate/conflict acceptance and zero replay divergence.
5. Any later legacy goal-domain migration remains a separately named gate.
