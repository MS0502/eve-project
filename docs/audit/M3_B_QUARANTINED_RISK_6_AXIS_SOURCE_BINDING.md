# M3-B Quarantined Risk-Defense Six-Axis Source Binding

## Baseline

`c9b46e2f0d509d78b6b2802e180e7a3a4be741b3` — PR #180 squash merge.

## Purpose

This artifact binds the six `risk_defense` registry source contracts to an explicit detached raw-record schema. It advances source-binding coverage after the merged four operational axes and two appraised survival axes without installing any production capture path.

Bound axes:

1. `threat_pressure`
2. `uncertainty_pressure`
3. `self_protection`
4. `boundary_defense`
5. `trust_risk`
6. `exposure_risk`

Combined registry source-binding progress becomes `12/37`; `25/37` appraised axes remain unbound.

## Source and appraisal boundary

The canonical source family is exactly:

```text
quarantined_risk_appraisal_trace
```

Each record is caller-supplied and immutable. The module does not acquire external input. Risk evidence is admissible only after an explicit versioned quarantine trace and a separate versioned bounded appraisal trace.

The integrity chain is fail-closed:

```text
raw source evidence
  -> quarantine input digest
  -> verified quarantine integrity digest
  -> exact appraisal input digest
  -> verified appraisal integrity digest
  -> canonical raw observation digest
  -> detached registry evidence
```

`appraisal_input_digest` must equal the exact `quarantine_integrity_digest`. A caller cannot claim that appraisal occurred on a different or unquarantined input.

Raw social feedback cannot be supplied directly. Direct hardware input, synthetic records, proposal-only records, registry-owner circular sourcing, and runtime-polled records are rejected.

## Canonical raw fields

The implementation consumes the exact five fields already frozen by the merged 37-axis source manifest:

| Axis | Exact raw fields |
|---|---|
| `threat_pressure` | `appraisal_version`, `impact_score`, `source_trust`, `threat_probability`, `verification_status` |
| `uncertainty_pressure` | `appraisal_version`, `conflict_count`, `missing_evidence_ratio`, `source_reliability`, `verification_gap` |
| `self_protection` | `appraisal_version`, `capability_limit`, `exposure_scope`, `reversibility`, `threat_pressure_input` |
| `boundary_defense` | `appraisal_version`, `boundary_violation_count`, `intent_confidence`, `persistence_score`, `remedy_available` |
| `trust_risk` | `appraisal_version`, `contradiction_count`, `reversibility`, `source_reliability`, `verification_depth` |
| `exposure_risk` | `audience_scope`, `authorization_status`, `persistence_risk`, `reversibility`, `sensitivity_class` |

The risk-defense manifest requires at least two records over at least one logical tick. Records must remain sorted, unique in logical time and evidence identity, and tied to one source contract.

## Deterministic derivation

All numeric inputs are bounded or non-negative integer evidence. Derivation uses explicit deterministic transforms and arithmetic means. Count pressure uses `count / (count + 1)` rather than an unbounded raw count.

Categorical exposure inputs are versioned finite maps:

```text
authorization_status:
  authorized   -> 0.0
  restricted   -> 0.5
  unauthorized -> 1.0

sensitivity_class:
  public     -> 0.0
  internal   -> 1/3
  sensitive  -> 2/3
  restricted -> 1.0
```

Confidence is deterministic and strictly positive. It is derived from the variance of the accepted record scores and remains within `(0,1]`.

## Recalculation and identity

Each raw observation digest binds:

- axis and logical tick;
- observation identity;
- source instance, source snapshot, source schema, and source integrity digest;
- quarantine trace identity, input digest, integrity digest, schema, method, outcome, and verification state;
- appraisal trace identity, exact input digest, integrity digest, schema, method, outcome, and verification state;
- exact ordered raw values;
- canonical acquisition and verification provenance;
- the raw-record and derivation rule versions.

The final registry evidence also carries a deterministic raw-bundle digest and derived source-integrity digest, so the value and confidence can be independently recalculated from the retained record set.

## Authority boundary

This PR remains detached and `shadow_only`.

It does **not**:

- install production capture;
- ingest raw social feedback;
- poll hardware;
- install a runtime observer or scheduler;
- materialize the registry owner;
- access persistence;
- append events;
- mutate live affect, drives, named state, goals, memory, self, or expression;
- start or satisfy the observation window;
- complete M3-B;
- open M3-C;
- authorize cutover;
- open M3-E authority.

Legacy runtime and legacy persistence remain authoritative.

## Validation and cross-chat reuse

PR #180 is a merged prerequisite and must be reused by exact pin. Moving this work to another chat, editing PR metadata, changing comments/review state, or changing Draft/Ready state is not a validation invalidator.

The next PR follows the existing discovery protocol:

```text
focused / M0 / M2-B discovery
  -> full suite blocked
  -> exact append-only M2-B decisions
  -> exact forward fingerprints
  -> final registered head
  -> full suite exactly once
```

A prior pin may be invalidated only by a relevant code-head change, artifact loss/corruption or digest mismatch, validation-scope/dependency change, or loss of required merge ancestry.

## Current authority status

```text
operational source bindings:          4/37
appraised survival bindings:          2/37
quarantined risk-defense bindings:    6/37
combined source bindings:            12/37
remaining appraised bindings:        25/37
production capture:                   absent
positive-confidence real observation: absent
observation window started:           false
M3-B complete:                         false
M3-C open:                             false
M3-E authority open:                  false
cutover authorized:                    false
```

Active blockers remain:

```text
REGISTRY_APPRAISED_25_AXIS_SOURCE_BINDINGS_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

## Next required artifact

Bind the remaining 25 appraised registry axes in bounded source-family stages. Retained real production capture and any observation-window start decision remain separate later artifacts.
