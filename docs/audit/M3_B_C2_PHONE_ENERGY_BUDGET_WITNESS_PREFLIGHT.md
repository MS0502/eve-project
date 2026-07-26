# M3-B C2 Phone Energy-Budget Witness Preflight

## Baseline

`f2c536e21d68bb6cd91e53748cd1063ccdc6e2e8` — PR #200 squash merge, first real retained observation receipt pinned (`1/37`).

PR #200 exact-head validation is a reusable prerequisite and must not be rerun because this work moved to a later PR/chat:

```text
exact head:   c4514a14192043ee536f84d9e25160b1142e896b
exact run:    30186490378
full:         3,155 passed
artifact SHA: 550f46a61be496b8c7d121f4caeaa39c75f8d26df61fd960f82148a059ac5f60
M2-E run:     30186490374
M2-E:         6/6 jobs passed
merge SHA:    f2c536e21d68bb6cd91e53748cd1063ccdc6e2e8
```

## Purpose

The first retained real observation covers only `prediction_error_pressure`. This preflight opens no new authority and does not reuse that event. It prepares the next independent production-origin candidate for `energy_budget`, one of the four hardware-direct operational axes already bound by PR #179.

The existing source contract requires exactly these five raw fields:

```text
available_cpu_budget
available_memory_budget
battery_governor_band
foreground_load
sampling_window_ticks
```

It also requires at least three records spanning at least two logical ticks. This preflight therefore captures exactly three full-engine interaction windows at ticks `0,1,2` from one phone process/source instance.

## Measurement boundary

Each private snapshot freezes only measurements from the real phone process/window:

- aggregate `/proc/stat` CPU total/idle deltas;
- EVE witness-process CPU time across the same interaction window;
- wall-clock duration for that interaction window;
- online CPU count;
- `/proc/meminfo` `MemTotal` and `MemAvailable`;
- integer battery capacity from `/sys/class/power_supply/battery/capacity` by default.

The derived operational fields are deterministic:

```text
available_cpu_budget    = cpu_idle_delta / cpu_total_delta
available_memory_budget = MemAvailable / MemTotal
battery_governor_band   = battery_capacity_percent / 100
foreground_load         = clamp(process_cpu_seconds / (wall_seconds * cpu_count), 0, 1)
sampling_window_ticks   = 1
```

`battery_governor_band` v1 intentionally means **directly observed battery headroom only**. Charging status, thermal state, guessed Android power modes, or undocumented vendor governor state are not silently mixed into it. A later policy change requires a versioned source-schema change rather than reinterpretation of an old witness.

The operator CLI executes `build_full_engine()` once and uses three explicit real operator `--input` values. CPU/process deltas therefore cover actual full-engine work rather than a detached synthetic fixture.

## Existing contract reuse

`PhoneEnergyBudgetRuntimeSnapshot.to_operational_raw_record()` converts the private runtime snapshot into the already-merged `OperationalRegistryRawRecord` contract. The detached record preserves the canonical five-field order and uses the existing `derive_operational_axis_evidence()` implementation; this PR does not create a second derivation formula.

The historical operational raw-record type remains `runtime_polled=False` by design: runtime polling happens in this new bridge first, then the immutable snapshot is converted to the detached caller-supplied record accepted by PR #179. No historical contract is rewritten.

## Public/private boundary

Private companion only:

- raw `/proc` counter deltas;
- memory totals;
- raw battery percentage;
- process CPU/wall timings;
- exact private snapshots and detached raw records;
- private nonce.

Public review output:

- operator launch attestation;
- local verification trace digest;
- derived `RegistryAxisPositiveConfidenceEvidence` mapping;
- evidence digest;
- three snapshot integrity digests;
- private-material digest;
- exact schema/policy/source identities;
- all authority/retention/window flags fixed false.

The public review never contains the private nonce or raw device counters.

## Fail-closed conditions

The witness refuses:

- dirty or wrong repository head;
- a private root or nonce path inside the repository;
- permissive nonce file permissions;
- fewer/more than three operator interactions;
- missing/malformed aggregate CPU counters;
- non-monotonic CPU/process/wall counters;
- missing/invalid `MemTotal` or `MemAvailable`;
- unreadable or out-of-range battery capacity;
- mixed source instances, duplicate/non-increasing ticks, or logical span below two;
- preclaimed production verification, retention, window, M3-C/M3-E, or cutover authority.

## Authority boundary

This preflight does **not** change the merged M3-B counters:

```text
reviewed real operator attestations:             1
verified production runtime anchors:             1
registered production source verifiers:          1/37
verified positive-confidence candidates:         1/37
retained real observations:                      1/37
M3-B observation window started:                 false
M3-B complete:                                   false
M3-C open:                                       false
M3-E authority open:                             false
cutover authorized:                              false
```

Only after a real phone witness is executed, its public output is independently reviewed and repository-pinned, and a later source-specific verifier/retention activation is reviewed may `energy_budget` advance those counters. CI fixtures from this PR cannot count.

## Next exact action after merge

Run the operator CLI once on the exact merged head with the existing operator-private nonce and exactly three real interaction texts. Return only the final public-review JSON. Do not copy the private witness file, nonce, `/proc` samples, or raw battery data into GitHub/chat.
