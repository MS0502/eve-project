# M3-B Operational Four-Axis Source Binding

## Baseline

`0c0cfdd5a32f25cab52f62e15895475ea64afd63` — PR #178 merge.

## Purpose

PR #178 fixed the raw-source plan for every registry axis. This artifact implements the first four source bindings, limited to the axes whose registry definitions explicitly allow direct operational input:

- `energy_budget`;
- `fatigue_pressure`;
- `recovery_need`;
- `overload_risk`.

The implementation accepts only caller-supplied immutable raw records. It does not poll hardware, install a runtime observer, collect production data, or write an owner state.

## Raw-record contract

Each `OperationalRegistryRawRecord` carries:

- exact axis and logical tick;
- unique observation, source-instance, and source-snapshot identities;
- source schema version and non-placeholder source-integrity SHA-256;
- exact five-field payload from the PR #178 source manifest;
- canonical acquisition and verification methods;
- canonical model/rule version and source family;
- raw-observation SHA-256.

The raw-observation digest is recalculated from the axis, logical tick, observation ID, source instance, source snapshot, source schema, source-integrity digest, exact raw values, raw schema version, source family, acquisition method, verification method, and model/rule version. A digest copied from another identity, tick, snapshot, source, provenance contract, or payload fails closed.

The provenance fields are not caller-extensible labels. They must exactly equal:

```text
acquisition_method:    explicit_caller_supplied_immutable_operational_record
verification_method:   exact_schema_range_and_digest_verification
model_or_rule_version: eve.m3-b.operational-registry-source-binding.v1
source_family:         operational_metrics_or_appraised_load_trace
```

This prevents arbitrary strings such as `unverified` or `none` from being promoted into a `verified_current_value_observation`.

The record also rejects:

- missing, reordered, duplicated, or additional raw fields;
- non-finite or out-of-range ratios;
- tick counts outside their sampling window;
- synthetic, proposal-only, circular registry-owner, or runtime-polled inputs;
- noncanonical provenance metadata;
- malformed or placeholder digests.

## Deterministic derivation

A binding requires the manifest's minimum count and logical span. Records must:

- use one axis;
- be sorted by unique logical tick;
- have unique observation IDs and snapshots;
- share one source instance and source schema;
- retain the exact canonical acquisition, verification, source-family, and model/rule contract.

The four values are derived as follows.

### `energy_budget`

Mean of:

- available CPU budget;
- available memory budget;
- battery governor band;
- inverse foreground load.

### `fatigue_pressure`

Mean of:

- active-processing ratio;
- queue pressure;
- inverse recovery-interval ratio;
- bounded task-switch ratio.

### `recovery_need`

Mean of:

- active-processing ratio;
- inverse cooldown ratio;
- bounded recent-overload ratio;
- inverse successful-recovery ratio.

### `overload_risk`

Mean of:

- saturated concurrent-demand count;
- latency-budget ratio;
- memory-pressure ratio;
- saturated queue depth;
- thermal governor band.

The record scores are averaged. Confidence is the deterministic bounded complement of score variance, with a positive floor of `0.5`. No sampling or hidden state is used.

## Result

```text
source bindings implemented:          4
remaining source bindings:            33
production operational capture:       absent
registry owner materialized:           false
observation window started:            false
observation window satisfied:          false
M3-B complete:                         false
M3-C open:                             false
M3-E authority open:                   false
cutover authorized:                    false
```

Remaining blockers:

```text
REGISTRY_APPRAISED_33_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Authority boundary

This artifact performs no:

- hardware polling;
- runtime hook or observer installation;
- scheduler activation;
- persistence or event append;
- owner materialization;
- live affect, drive, named-state, goal, memory, self, or expression mutation;
- observation-window transition;
- M3-C, cutover, or M3-E authorization.

The deterministic audit fixture is not production evidence. Legacy runtime and legacy persistence remain authoritative.

## Next required artifact

The remaining 33 axes require appraised source bindings. Those bindings must preserve quarantine, long-horizon, AGP, identity, and social-evidence boundaries from the PR #178 manifest. Separately, a retained real operational capture package must later provide immutable raw records for these four bindings before their evidence can be accepted as production observation evidence.
