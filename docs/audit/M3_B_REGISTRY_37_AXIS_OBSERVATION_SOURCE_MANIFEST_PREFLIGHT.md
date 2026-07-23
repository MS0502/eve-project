# M3-B Registry 37-Axis Observation Source Manifest Preflight

## Baseline

`03bf2de7294ae21cef0961cd50e516be85330d13` — PR #177 merge.

## Purpose

PR #177 defined the evidence record and bundle that real positive-confidence registry observations must satisfy. This artifact defines the upstream source plan for every one of the 37 registry axes. It prevents a later capture patch from inventing values, using the registry owner as its own source, treating proposals as observations, or silently changing the required raw fields and observation span.

This preflight contains no real observations and binds no runtime source.

## Exact coverage

The manifest contains exactly 37 immutable entries in canonical registry order across six groups:

- survival/stability: 6 axes;
- risk/defense: 6 axes;
- social/relationship: 7 axes;
- learning/exploration: 6 axes;
- self/identity: 6 axes;
- expression/action: 6 axes.

Every entry has a unique source-contract ID, a source family, a versioned observation class, a sorted raw-field set, minimum record count, minimum logical span, the registry evidence requirement, derivation and confidence rule IDs, quarantine/appraisal requirements, and the hardware-direct boundary.

## Source-family boundary

The six source families are:

| Registry group | Required source family |
|---|---|
| survival/stability | `operational_metrics_or_appraised_load_trace` |
| risk/defense | `quarantined_risk_appraisal_trace` |
| social/relationship | `quarantined_social_appraisal_trace` |
| learning/exploration | `validated_learning_and_prediction_trace` |
| self/identity | `long_horizon_self_model_review_trace` |
| expression/action | `agp_bounded_expression_action_trace` |

Only `energy_budget`, `fatigue_pressure`, `recovery_need`, and `overload_risk` retain the registry's direct-hardware allowance. All other axes require appraised evidence. Direct hardware input remains operational only and cannot create existential, social, or identity affect.

## Raw-field and time requirements

Each axis has five canonical raw fields. Examples:

- `energy_budget`: CPU budget, memory budget, battery governor band, foreground load, sampling window;
- `threat_pressure`: threat probability, impact, source trust, verification status, appraisal version;
- `social_trust`: fulfilled commitments, contradictions, repairs, source trust, observation span;
- `prediction_error_pressure`: predicted and observed digests, normalized error, model version, verification status;
- `identity_integrity`: unauthorized identity writes, provenance gaps, constitutional conflicts, replay consistency, review version;
- `action_readiness`: feasible action count, selected confidence, capability, authorization, reversibility.

Operational axes require at least three raw records over at least two logical ticks. Risk and expression/action axes require at least two records. Social and self/identity axes require longer multi-record spans. Axis-specific overrides are versioned in the manifest.

## Fail-closed rules

An entry is rejected if a caller changes any canonical source family, observation class, raw-field set, minimum record count, minimum logical span, or appraisal requirement. It is also rejected if it contradicts the live registry's group, evidence, quarantine, or hardware boundary.

The following are always forbidden:

- proposal-only input;
- synthetic values presented as real observations;
- the registry owner acting as its own source;
- a claimed real binding without a later reviewed binding artifact;
- runtime capture or polling installed by this preflight;
- missing raw reference, source schema version, or source integrity digest requirements.

## Current result

```text
axis_count:                         37
structurally_complete:              true
real_source_binding_count:          0
real_observation_values_present:    false
capture_ready:                      false
observation_window_started:         false
observation_window_satisfied:       false
M3-B complete:                      false
M3-C open:                          false
M3-E authority open:                false
cutover authorized:                 false
```

Active blockers:

```text
REGISTRY_REAL_OBSERVATION_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

Legacy runtime and persistence remain authoritative.

## Next required artifact

A later PR must bind every source-contract entry to an actual immutable raw-schema producer or retained raw evidence package. The binding artifact must identify the exact code/data source, schema version, capture cadence, digest calculation, recalculation procedure, and absence of circular owner/proposal derivation. It still must not start the observation window. Real capture and the separate window-start decision remain later artifacts.
