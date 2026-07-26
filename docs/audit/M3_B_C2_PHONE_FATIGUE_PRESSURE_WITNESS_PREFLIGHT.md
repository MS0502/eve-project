# M3-B C2 Phone Fatigue-Pressure Witness Preflight

## Baseline

`1bb0a0ad6633fce932add5929e852b3d5846cc3e` — PR #204 squash merge, with the second real retained observation receipt pinned (`2/37`).

PR #204 exact-head validation is a reusable prerequisite and must not be rerun because work moved to this later PR/chat:

```text
exact head:   5a1a6fd3c2039b7e15f3f7f3d5d45c254cf68b3e
exact run:    30194276579
focused:      no focused tests selected
full:         3,172 passed
artifact:     exact-head-validation-5a1a6fd3c2039b7e15f3f7f3d5d45c254cf68b3e
artifact SHA: b7bf28656726fe5be5632f07e08ee5c0865a8378f1a7682a94e53ca4f3e93154
M2-E run:     30194276565
M2-E:         6/6 jobs passed
merge SHA:    1bb0a0ad6633fce932add5929e852b3d5846cc3e
```

## Purpose

The retained chain currently contains exactly two immutable real observations:

```text
sequence 1  prediction_error_pressure
sequence 2  energy_budget
```

Neither may be replayed. This preflight prepares a new independent production-origin candidate for sequence 3: `fatigue_pressure`, another hardware-direct operational axis already bound by PR #179.

The existing source contract requires exactly these five raw fields:

```text
active_processing_ticks
queue_pressure
recovery_interval_ticks
sampling_window_ticks
task_switch_count
```

It requires at least three records spanning at least two logical ticks. The phone witness therefore captures exactly three full-engine interaction windows at ticks `0,1,2` from one phone process/source instance.

## Measurement boundary

Each private snapshot records only real process/kernel observations covering one actual `build_full_engine()` interaction:

- EVE witness-process CPU seconds from `os.times()`;
- monotonic wall-clock duration;
- visible CPU count;
- one-minute kernel load average before and after the interaction;
- process context-switch count delta.

Context-switch acquisition is versioned and fail-closed:

1. `getrusage_context_switch_delta_v1` from the current process;
2. if unavailable, `proc_self_status_context_switch_delta_v1` from the current process's own `/proc/self/status` counters.

No generic system-wide process list, root access, Android private API, synthetic fixture, or guessed counter is accepted.

The deterministic v1 mapping is:

```text
tick_hz                   = 1,000,000
sampling_window_ticks     = round(wall_seconds * tick_hz), minimum 1
active_processing_ticks   = floor(process_cpu_seconds * tick_hz / visible_cpu_count)
recovery_interval_ticks   = sampling_window_ticks - active_processing_ticks
queue_pressure            = clamp(mean(load_average_1m_before, load_average_1m_after)
                                  / visible_cpu_count, 0, 1)
task_switch_count         = end_context_switch_count - start_context_switch_count
```

The snapshot rejects a process-CPU observation that exceeds the visible CPU capacity of its wall window instead of clipping the observation into range. This keeps the governed `active_processing_ticks <= sampling_window_ticks` invariant without fabricating a lower value.

`queue_pressure` is explicitly the normalized real kernel runnable-load proxy for this measurement policy; it is not silently relabeled as an internal EVE queue. Any future different queue semantics require a versioned policy/schema change.

## Existing contract reuse

`PhoneFatiguePressureRuntimeSnapshot.to_operational_raw_record()` converts each private runtime snapshot into the already-merged `OperationalRegistryRawRecord` contract and preserves the canonical five-field order. Evidence is then derived by the existing `derive_operational_axis_evidence()` implementation; this preflight introduces no second fatigue scoring formula.

The historical operational raw-record type remains `runtime_polled=False` by design: phone observation occurs in the source bridge first, after which the immutable snapshot is converted into the detached caller-supplied record accepted by PR #179. No historical source-binding contract is rewritten.

## Public/private boundary

Operator-private companion only:

- raw process CPU and wall durations;
- raw load-average observations;
- raw context-switch counts/deltas;
- exact private snapshots and detached raw records;
- private nonce.

Public review output:

- operator launch attestation;
- local verification-trace digest;
- derived `RegistryAxisPositiveConfidenceEvidence` mapping and digest;
- three snapshot integrity digests;
- private-material digest;
- exact process-CPU, queue, task-switch, tick, schema, and policy method identities;
- all review/verifier/retention/window/authority flags fixed false.

The public review contains no raw CPU seconds, wall duration, load-average values, context-switch counts, or private nonce.

## Fail-closed conditions

The witness refuses:

- dirty or wrong repository head;
- private root or nonce path inside the repository;
- permissive nonce-file permissions;
- fewer/more than three operator interactions;
- missing/invalid process CPU, wall, CPU-count, or load-average observations;
- process CPU above the visible CPU capacity of the measured wall window;
- unavailable context-switch counters from both allowed methods;
- context-switch counters moving backwards;
- mixed source instances, duplicate/non-increasing ticks, or logical span below two;
- unsupported measurement method labels;
- fixture, production-verification, retention, window, M3-C/M3-E, or cutover preclaims.

## Authority boundary

This preflight does **not** advance the merged counters:

```text
reviewed real operator attestations (C2):          2
registered runtime provenance verifiers (C2):     2
verified production runtime anchors (C2):         2
registered production source verifiers (C2):      2/37
verified positive-confidence candidates:          2/37
retained real observations:                        2/37
M3-B observation window started:                   false
M3-B complete:                                     false
M3-C open:                                         false
M3-E authority open:                               false
cutover authorized:                                false
```

Only a later real phone execution can create a candidate. That public review must then be independently reviewed and pinned before a separate reviewed verifier/sequence-3 retention activation may advance any counter.

## Next exact action after merge

Run the operator CLI once on the exact merged head using the existing operator-private nonce and exactly three real interaction texts. Return only the final public-review JSON. Do not copy the private witness file, nonce, raw process timings, load averages, or context-switch observations into GitHub/chat.
