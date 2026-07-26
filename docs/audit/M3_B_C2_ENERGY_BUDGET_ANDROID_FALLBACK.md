# M3-B C2 Energy-Budget Android Fallback

## Trigger

The first real phone execution of the merged #201 witness on exact main `f178b9e0b0fbaa341776fbef66e6c5c87fe5a157` failed before any interaction or witness serialization with:

```text
cannot read CPU counters from /proc/stat
```

No `energy_budget` public review, verifier registration, retained observation, observation-window transition, or authority change occurred. Existing retained coverage therefore remains `1/37` from the earlier `prediction_error_pressure` event only.

## Root cause

The #201 collector assumed that a non-root Android/Termux process could read `/proc/stat`. That assumption is not portable across hardened Android app sandboxes. The failure is an acquisition-surface defect, not evidence that the phone lacks CPU state and not a reason to fabricate counters.

## v2 acquisition policy

The v2 witness keeps the existing `energy_budget` operational raw contract and derivation formula. It changes only how the four raw operational inputs are observed and records the method explicitly.

CPU headroom:

1. `proc_stat_idle_delta_v1` when `/proc/stat` is readable;
2. otherwise `kernel_loadavg_1m_headroom_v1`, using the mean real one-minute kernel load average across the interaction window normalized by visible CPU count.

Memory headroom:

1. `proc_meminfo_available_v1` when `/proc/meminfo` is readable;
2. otherwise `sysconf_avphys_pages_v1`, using kernel-reported physical/available page counts and page size.

Battery headroom:

1. `sysfs_capacity_v1` when the selected power-supply capacity file is readable;
2. otherwise `termux_api_battery_status_v1`, using `termux-battery-status` and its Android battery percentage.

Foreground load remains the measured witness-process CPU-seconds divided by wall time times visible CPU count.

The private snapshot includes the exact method and method-specific raw observations. The public review exposes only method identifiers, bounded derived evidence, and digests; raw device counters remain in the operator-private companion.

## Fail-closed rules

The v2 contract rejects:

- a load-average snapshot carrying fake `/proc/stat` deltas;
- a proc-stat snapshot carrying load-average observations;
- unsupported CPU/memory/battery method labels;
- missing or invalid method-specific observations;
- impossible memory or battery ranges;
- fixture/synthetic/retention/authority preclaims.

If both CPU acquisition methods fail, collection stops. If both memory methods fail, collection stops. If sysfs battery access fails and Termux:API is unavailable, collection stops with an actionable error instead of inventing a battery value.

## Authority boundary

This hotfix does not review or retain an `energy_budget` witness. It does not change any M3-B counter and does not start the observation window. M3-C, M3-E, and cutover remain closed.

## Validation reuse

#201 is the merged prerequisite and its accepted exact-head evidence must not be rerun merely because this hotfix or a new chat exists:

```text
exact head:   fce245e5c4e63f2224b6fe69d54375315896c177
exact run:    30187041821
focused:      5 passed
full:         3,160 passed
artifact SHA: ec8b3b6f045e9f39007bd98b0a0b55f680a78622038f5ef1c918abdd4457522c
M2-E run:     30187041822
M2-E:         6/6 jobs passed
merge SHA:    f178b9e0b0fbaa341776fbef66e6c5c87fe5a157
```

This hotfix receives its own validation only on its final registered head.
