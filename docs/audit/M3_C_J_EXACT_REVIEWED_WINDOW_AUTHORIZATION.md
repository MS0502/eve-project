# M3-C-J Exact-Reviewed Observation Evaluator Authorization

## Status

This slice pins the exact validated M3-C-J preflight evaluator and one immutable
observation-only authorization packet. It authorizes deterministic evaluation of
separately produced evidence. It does not access the private database, construct
a writer, append an event, capture a baseline, create or restore a backup, start
a runtime hook, or open the real observation window.

```text
reviewed evaluator implementation head:
a567f05fbee22b98fe02b612cbd42ff17de34182

reviewed authorization digest:
803780b19f0c496adb0a3a68ba32bd296a356a8ea3eeaf2fe6a33cb3476510fb

maximum accepted window events: 32
```

## Exact prerequisite reused without rerun

```text
PR:             #228
exact head:     a567f05fbee22b98fe02b612cbd42ff17de34182
exact run:      30449875769
focused:        11 passed
full:           3,315 passed
forward gate:   0 / 0 / 0 / 0
artifact:       exact-head-validation-a567f05fbee22b98fe02b612cbd42ff17de34182
artifact SHA:   c2ac6feb56461fb6e81954484383c92ced6b24b0178cfda92ee9824e44378dd3
M2-E run:       30449875832
M2-E:           6/6 passed
merge SHA:      3bee78e8fbe7053be55b9c1608c03b37e8d0cb5b
```

The first two #228 discovery heads never ran the full suite. The final registered
head ran it exactly once. PR #227, PR #225, the #211 phone witness, and retained
sequences 1 through 5 were not rerun.

## Packet binding

The authorization digest binds:

- the exact #228 evaluator implementation head;
- the exact M3-C-I implementation validation, artifact, M2-E run, and merge SHA;
- the exact reviewed writer authorization and implementation head;
- the private production database-path digest, not its plaintext path;
- the exact reviewed bounded storage policy;
- a hard maximum of 32 observed lifecycle events;
- explicit human review;
- explicit false values for production append authorization, runtime integration,
  action, scheduling, speech, legacy migration, legacy goal-authority transfer,
  and M3-E.

Any change to these fields changes the packet digest and fails closed.

## Reachability boundary

`active_reviewed_observation_window_authorization_packet()` constructs the one
reviewed packet entirely in memory and verifies its canonical SHA-256.
`verify_active_observation_window_authorization()` accepts only the exact reviewed
implementation head and digest.

Neither function accepts an environment variable, branch name, file-presence
signal, default path, home-directory expansion, mutable configuration, or import
side effect. Neither function constructs a writer or store.

## What becomes true

```text
reviewed evaluator implementation pin present: true
reviewed observation authorization digest present: true
exact packet can be built and verified in memory: true
bounded evidence evaluator can accept a later exact packet: true
```

## What remains false

```text
production database path plaintext public: false
production database accessed by this slice: false
production writer constructed by this slice: false
production lifecycle append executed: false
pre-window baseline captured: false
production backup created: false
production restore executed: false
M3-C-J real observation window started: false
runtime startup integration: false
action/scheduler/speech authority: false
legacy goal-domain authority transferred: false
legacy migration authorized: false
M3-E authority open: false
phone command issued or replayed: false
retained sequences 1 through 5 replayed: false
```

## Next boundary

The next separately reviewed step is a private-device operator packet. It must:

1. verify the exact checked-in evaluator and writer packets;
2. resolve the already reviewed private absolute path without publishing it;
3. capture a disabled-writer baseline and verified backup;
4. enable only named bounded lifecycle appends;
5. retain at most 32 immutable append receipts;
6. disable the writer and verify final integrity/replay;
7. restore the baseline into a separate path and verify rollback preservation;
8. emit a canonical evidence bundle for offline evaluation.

That operator packet must not replay #211 or retained sequences 1 through 5, and
must not transfer legacy goal authority or open M3-E.
