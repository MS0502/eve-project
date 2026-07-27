# M3-B C2 Phone Stress-Load Witness Preflight

## Baseline

`b613e570b4c27ed75ebdb93aaef5a4756ffb44a4` — PR #210 squash merge. The canonical retained-real-observation chain is exactly four immutable events: `prediction_error_pressure` sequence 1, `energy_budget` sequence 2, `fatigue_pressure` sequence 3, and `recovery_need` sequence 4. Retained coverage is `4/37`. Reviewed/runtime/source/candidate coverage is also `4/37`. The M3-B observation window has not started; M3-B is incomplete; M3-C, M3-E, and cutover remain closed.

## Purpose

This artifact prepares a **new**, independently witnessed real-phone production-origin acquisition surface for the fifth candidate axis, `stress_load`. It does not reuse or replay any of the three-interaction witnesses that produced sequences 1 through 4. It does not review, register, or retain a `stress_load` observation.

The merged #180 source-binding contract already defines the canonical `stress_load` appraisal fields:

```text
appraisal_version
controllability_score
demand_score
overload_score
uncertainty_score
```

No field is added, removed, reordered, or silently reclassified.

## Two-stage provenance boundary

The #180 `AppraisedSurvivalRawRecord` deliberately rejects a canonical record that claims `runtime_polled=true` or `hardware_direct_input=true`. This preflight therefore does **not** relabel raw phone metrics as a detached appraisal record.

The acquisition path is explicitly two-stage:

```text
new real full-engine interaction
        ↓
operator-private process CPU / wall / kernel load observations
        ↓
versioned deterministic phone stress appraisal bridge
        ↓
detached canonical survival-appraisal trace
        ↓
#180 AppraisedSurvivalRawRecord(runtime_polled=false,
                                hardware_direct_input=false,
                                appraisal_verified=true)
        ↓
#180 positive-confidence evidence derivation
```

The runtime measurements are genuine appraisal **inputs**. The canonical record is the bridge **output** and contains only the five governed appraisal fields. Public review exposes this distinction through an exact provenance-boundary mapping:

```text
runtime_metrics_used_as_appraisal_input:          true
runtime_input_kind:                               operator_private_real_runtime_metrics
appraisal_bridge_output_detached:                 true
appraisal_output_kind:                            detached_verified_appraisal_trace
canonical_appraised_record_runtime_polled:        false
canonical_appraised_record_hardware_direct_input: false
raw_runtime_metrics_publicly_retained:            false
```

This prevents both forbidden interpretations: raw runtime measurements cannot masquerade as detached caller fixtures, and detached appraisal records cannot falsely claim direct runtime polling.

## Measurement and appraisal policy

For each of exactly three **new** operator interactions, the CLI measures around the actual `engine.chat_stream(...)` call:

- process CPU time from `os.times()`;
- monotonic wall duration;
- one-minute kernel load average before and after the interaction;
- visible CPU count.

The versioned deterministic bridge computes only bounded `[0,1]` appraisal outputs:

```text
process_cpu_ratio = (process_cpu_seconds / cpu_count) / wall_seconds
queue_ratio       = load_average_1m / cpu_count
uncertainty       = abs(queue_ratio_after - queue_ratio_before)
demand            = mean(process_cpu_ratio, queue_ratio_after)
overload          = max(process_cpu_ratio, queue_ratio_after)
controllability   = 1 - mean(overload, uncertainty)
```

Each expression is clipped to `[0,1]` only at the normalized score boundary. There is no random sampling, learned threshold, caller-supplied appraisal score, or hidden fixture fallback.

The three detached appraisal records must preserve strictly increasing ticks `0,1,2`. #180 then derives the canonical `stress_load` value as the mean record score and the existing bounded variance-based confidence. This preflight does not change that derivation rule.

## Privacy boundary

Raw interaction text, process CPU seconds, wall durations, load-average values, CPU count, private nonce, and the private witness mapping remain under the operator-selected private root outside the repository. The public review exposes only:

- bounded `RegistryAxisPositiveConfidenceEvidence`;
- exact appraisal and measurement method identifiers;
- appraisal-input and appraisal-integrity digests;
- snapshot integrity digests;
- the explicit two-stage provenance-boundary mapping;
- approved public launch-attestation metadata;
- private-material and public-review digests;
- authority/window/retention flags, all false at preflight.

The repository must not receive the private witness JSON or raw interaction text.

## Fail-closed boundaries

The preflight rejects:

- anything other than exactly three new immutable snapshots with strictly increasing logical ticks and the required span;
- source-instance mismatch between attestation and snapshots;
- non-finite or negative measurements;
- process CPU exceeding visible CPU capacity;
- source/bridge/appraisal-policy drift;
- synthetic or fixture snapshots;
- a canonical appraisal output that becomes runtime-polled, hardware-direct, synthetic, or unverified;
- pre-claimed production verification, verifier registration, retention, observation-window start, M3 completion, or cutover authority;
- repository-resident private root or nonce file;
- dirty or wrong-head operator checkout.

## Authority result

```text
reviewed real operator attestations:          4
registered runtime provenance verifiers:     4
registered production source verifiers:      4/37
verified positive-confidence candidates:      4/37
retained real observations:                   4/37
stress_load witness preflight:                present after merge
stress_load real witness executed:            false
stress_load reviewed/registered:              false
stress_load retained:                         false
M3-B observation window started:              false
M3-B complete:                                false
M3-C open:                                    false
M3-E authority open:                          false
cutover authorized:                           false
```

## Validation reuse

PR #210 is the immutable merged prerequisite and must be reused across chat/operator-session changes when its exact-head artifact, workflow scope/dependency, and merge ancestry still match:

```text
exact head:   c653926b3dd5dfbd05a130463f3b14c165595522
exact run:    30234139943
focused:      no focused tests selected
full:         3,195 passed
artifact:     exact-head-validation-c653926b3dd5dfbd05a130463f3b14c165595522
artifact SHA: a48c9bc0aa7a9a3343d61bb14c2e19f760a251660fc2b55fa422a1cf0a8c7ba9
M2-E run:     30234139955
M2-E:         6/6 passed
merge SHA:    b613e570b4c27ed75ebdb93aaef5a4756ffb44a4
```

A new chat, PR body edit, comment, review, or Draft/Ready transition is not a rerun trigger. This preflight PR must validate its own final new head once; it must not re-run #210 as a prerequisite.

## Post-merge operator boundary

Only after this preflight itself is exact-head validated and merged may the operator execute the new `stress_load` phone witness on that exact merge head. The command must use three new interaction texts; sequences 1 through 4 and their prior three-interaction witness sessions are immutable history and must not be replayed.

A resulting public-review JSON must be independently reviewed before any `stress_load` production verifier registration or sequence-5 retention staging is created. Any future retained append must be **sequence 5 only** and prove exact continuity from `m3b:c2:retained:recovery_need:000004`, envelope digest `7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691`, and store-chain digest `16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f`.
