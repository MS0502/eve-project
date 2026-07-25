# M3-B Appraised Survival Two-Axis Source Binding

## Baseline

`6c89e717c8bd702b86a322d0dbec439ea1ff3944` — PR #179 squash merge.

## Purpose

The operational source-binding artifact covers the four `survival_stability` axes that explicitly permit direct operational input. This artifact covers the two remaining axes in that group:

- `stress_load`;
- `stability_need`.

Both axes require a verified appraisal. Neither axis permits direct hardware input. This artifact accepts only caller-supplied immutable appraisal records and produces detached positive-confidence observation evidence.

## Canonical appraisal boundary

Every raw record binds:

- exact axis, logical tick, observation identity, source instance, and source snapshot;
- source schema and non-placeholder source-integrity SHA-256;
- unique appraisal-trace identity;
- appraisal-input and appraisal-integrity SHA-256 values;
- exact five-field raw payload from the PR #178 source manifest;
- canonical appraisal schema, method, accepted outcome, acquisition method, verification method, source family, quarantine status, and rule version;
- canonical raw-observation SHA-256.

The provenance fields are exact, not caller-extensible labels:

```text
appraisal_schema_version: eve.m3-b.survival-appraisal-trace.v1
appraisal_method:         deterministic_bounded_survival_load_appraisal
appraisal_outcome:        accepted_bounded_survival_appraisal
acquisition_method:       explicit_caller_supplied_immutable_appraised_survival_record
verification_method:      exact_appraisal_schema_range_identity_and_digest_verification
source_family:            operational_metrics_or_appraised_load_trace
quarantine_status:        not_applicable_non_social_survival_trace
model_or_rule_version:    eve.m3-b.appraised-survival-source-binding.v1
```

The record fails closed when appraisal verification is absent or when input is raw social feedback, direct hardware input, synthetic data, proposal-only data, a circular registry-owner source, or a runtime-polled record. This artifact deliberately handles only non-social survival appraisal traces; social and identity evidence remain governed by their own quarantine-bound source families.

## Deterministic derivation

Both bindings require at least three records spanning at least two logical ticks. Records must be sorted, use unique ticks, observation IDs, source snapshots, and appraisal-trace IDs, and share one source instance and exact provenance contract.

### `stress_load`

Mean of:

- inverse controllability;
- demand score;
- overload score;
- uncertainty score.

The raw `appraisal_version` must equal the canonical appraisal schema.

### `stability_need`

Mean of:

- invariant-failure ratio over the sampling window;
- pending-migration ratio over the sampling window;
- replay-divergence ratio over the sampling window;
- inverse rollback-readiness score.

Counts cannot exceed the sampling window.

For both axes, record scores are averaged. Confidence is the deterministic bounded complement of score variance with a positive floor of `0.5`. No sampling, hidden runtime state, or owner mutation is used.

## Progress

```text
operational source bindings:          4
appraised survival bindings:          2
combined source bindings:             6/37
remaining appraised bindings:         31/37
production capture:                   absent
positive-confidence real observation: absent
observation window started:           false
M3-B complete:                         false
M3-C open:                             false
M3-E authority open:                   false
cutover authorized:                    false
```

Remaining blockers:

```text
REGISTRY_APPRAISED_31_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Authority boundary

This artifact performs no:

- hardware polling or direct hardware ingestion;
- raw social-feedback ingestion;
- runtime observer, hook, or scheduler installation;
- persistence or event append;
- registry-owner materialization;
- live affect, drive, named-state, goal, memory, self, or expression mutation;
- observation-window transition;
- M3-C, cutover, or M3-E authorization.

The deterministic audit fixture is not production evidence. Legacy runtime and legacy persistence remain authoritative.

## Next required artifact

The next bounded artifact should implement the six `risk_defense` source bindings with mandatory quarantine and verified appraisal semantics. Real retained capture remains a separate later artifact and cannot be inferred from deterministic fixtures.
