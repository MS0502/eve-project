# M3-B C2 Phone Recovery-Need Witness Preflight

## Baseline

`90933501e1a3b15b4721d5ffd944c00b168daf4e` — PR #207 squash merge. The canonical retained-real-observation chain is exactly three immutable events: `prediction_error_pressure` sequence 1, `energy_budget` sequence 2, and `fatigue_pressure` sequence 3. Retained coverage is `3/37`. The M3-B observation window has not started; M3-B is incomplete; M3-C, M3-E, and cutover remain closed.

## Purpose

This artifact prepares the next independent real-phone production-origin witness for the fourth candidate axis, `recovery_need`. It does not review or retain an observation. It only creates a versioned acquisition surface capable of producing the already-governed five-field raw contract from real full-engine process/scheduler observations.

The governed `recovery_need` fields remain exactly:

```text
active_processing_ticks
cooldown_ticks
recent_overload_count
sampling_window_ticks
successful_recovery_count
```

No field is added, removed, reordered, or reinterpreted by the operational binding.

## Measurement policy

Each of three real full-engine interactions is measured in two adjacent windows.

### Active interaction window

The operator CLI measures, around the actual `engine.chat_stream(...)` interaction:

- process CPU time from `os.times()`;
- monotonic wall time;
- one-minute kernel load average before and after the interaction;
- visible CPU count.

`active_processing_ticks` is the interaction process-CPU delta normalized by visible CPU count and converted at `1,000,000 ticks/second`.

### Fixed quiet cooldown window

Immediately after the interaction, the same process enters one versioned fixed `1.0 s` quiet interval. The CLI measures process CPU, monotonic wall time, and kernel load average across that real cooldown interval. The fixed interval is part of the measurement policy, not caller-provided data.

`cooldown_ticks` is the actually observed cooldown wall duration converted at the same tick rate. `sampling_window_ticks` is the active wall window plus the cooldown wall window.

### Recent overload count

`recent_overload_count` is the count of the two active-window load-average samples that exceed visible CPU capacity. This uses the natural capacity boundary `load_average_1m > cpu_count`; it does not introduce an arbitrary learned threshold.

### Successful recovery count

`successful_recovery_count` is the count of two independently observed non-increase checks:

1. normalized process-CPU ratio during cooldown is no greater than during the active interaction;
2. the post-cooldown one-minute load average is no greater than the post-active load average.

The count therefore lies in `0..2`. It is derived only from real measurements and is not forced positive.

## Privacy boundary

Raw process CPU seconds, raw wall-clock durations, raw load-average samples, the private nonce, and the private witness mapping stay under the operator-selected private root outside the repository. The public review exposes only:

- the bounded `RegistryAxisPositiveConfidenceEvidence` mapping;
- acquisition/derivation method identifiers;
- attestation metadata approved by the trust-root contract;
- snapshot integrity digests;
- private-material digest;
- public-review digest;
- authority/window/retention flags, all false at preflight.

The repository must not receive the private witness JSON.

## Fail-closed boundaries

The preflight rejects:

- anything other than exactly three immutable snapshots with strictly increasing logical ticks and the required span;
- source-instance mismatch between attestation and snapshots;
- non-finite or negative measurements;
- active or cooldown process CPU exceeding visible CPU capacity;
- measurement/schema/policy drift;
- synthetic or fixture snapshots;
- pre-claimed production verification, verifier registration, retention, observation-window start, M3 completion, or cutover authority;
- repository-resident private root or nonce file;
- dirty or wrong-head operator checkout.

## Authority result

```text
reviewed real operator attestations:          3
registered runtime provenance verifiers:     3
registered production source verifiers:      3/37
verified positive-confidence candidates:      3/37
retained real observations:                   3/37
recovery_need witness preflight:              present after merge
recovery_need reviewed/registered:            false
recovery_need retained:                       false
M3-B observation window started:              false
M3-B complete:                                false
M3-C open:                                    false
M3-E authority open:                          false
cutover authorized:                           false
```

## Validation reuse

PR #207 is the immutable merged prerequisite and must be reused across chat/operator-session changes when its exact-head artifact, workflow scope/dependency, and merge ancestry still match:

```text
exact head:   6e81f77754d1f0d6c543cd15bac6985a2a17a0ec
exact run:    30197930139
focused:      no focused tests selected
full:         3,183 passed
artifact:     exact-head-validation-6e81f77754d1f0d6c543cd15bac6985a2a17a0ec
artifact SHA: 46e7d588a05c7edb1827253f023e284ceeff913954c7d681d0c42d6e8a232731
M2-E run:     30197930127
M2-E:         6/6 passed
merge SHA:    90933501e1a3b15b4721d5ffd944c00b168daf4e
```

A new chat, PR body edit, comment, review, or Draft/Ready transition is not a rerun trigger.

## Post-merge operator boundary

Only after this preflight itself is exact-head validated and merged may the operator execute the real phone witness on that exact merge head. A resulting public-review JSON must be independently reviewed before any production verifier registration or sequence-4 retention staging is created.

Any future `recovery_need` retained append must be **sequence 4 only** and prove exact continuity from the immutable sequence-3 `fatigue_pressure` event and chain pinned by #207.
